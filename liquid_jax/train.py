

from collections import defaultdict
import datetime
from functools import partial
import json
import math
from pathlib import Path
from dataclasses import replace
import traceback

import operator
import jax
import numpy as np
import optax
import tqdm
from flax import linen as nn
from flax import struct
import jax.numpy as jnp

from task_base import Task
from mnist import Mnist
from cifar10 import Cifar10
from bikes import Bikes
from energy import Energy
from learner_le import LeLearner
from learner_de import DeLearner

from math_utils import ce_loss, mse_loss, ce_loss_logprobs_labels, mix_weighted_logits, mix_weighted_mean, bregman_divergence
from structs import TrainParams, TrainReturn

CurrentTask = Energy
n_delegators = 5
n_predictors = 20

g_params = TrainParams(
    batch_size=512,
    preload_batches_to_gpu=50,
    valid_batches=4,
    epochs=500,
    lr=1e-3,
    optimizer="adam",
    task=CurrentTask,
    n_predictors=n_predictors,
    n_delegators=n_delegators,
    learner=LeLearner,
    predictor=(16, 8, CurrentTask.out_dim()),
    delegator=(8, n_predictors)
)

DIVERSITY_LAMBDA = 0.0

@struct.dataclass
class InOutData:
    
    x: jax.Array
    y: jax.Array


@partial(jax.jit, static_argnames=("model", "train_params", "calc_metric"))
def loss_fn(
    network_params: dict,
    key: jax.Array,
    inout: InOutData,
    model: nn.Module,
    train_params: TrainParams,
    calc_metric: bool = False 
):
    
    k_forward, k_loss = jax.random.split(key)

    train_return: TrainReturn = train_params.learner.forward(
        key=k_forward,
        x=inout.x,
        model=model,
        params=network_params
    )

    task_type = train_params.task.task_type()
    ensemble_weights = train_return.power 

    if task_type == "classification":
        ensemble_logits = train_return.ys
        ensemble_logprobs = jax.nn.log_softmax(ensemble_logits)
        logprobs = mix_weighted_logits(ensemble_logits, train_return.power)
    elif task_type == "regression":
        ensemble_ys = train_return.ys
        yhat = mix_weighted_mean(ensemble_ys, ensemble_weights)
    
    # @jax.vmap
    # def get_optimal_weights(yhat_one: jax.Array, y_one: jax.Array, init_probs: jax.Array):    
    #     init_weights = jnp.log10(init_probs + 1e-8)
    #     solver = optax.lbfgs()

    #     def wrapped_loss_fn(weights):
    #         a = nn.softmax(weights)
    #         return ce_loss(mix_weighted_logits(yhat_one, a), y_one)
            
    #     def step_fn(state, _):
    #         weights, opt_state = state
    #         value, grad = jax.value_and_grad(wrapped_loss_fn)(weights)
    #         updates, new_opt_state = solver.update(
    #            grad, opt_state, weights, value=value, grad=grad, value_fn=wrapped_loss_fn
    #         )
    #         new_weights = optax.apply_updates(weights, updates)

    #         return (new_weights, new_opt_state), None

    #     init_state = (init_weights, solver.init(init_weights))
    #     (final_weights, _), _ = jax.lax.scan(step_fn, init_state, length=30)
        
    #     return nn.softmax(final_weights)

    def error_diversity_classification(
        ensemble_weights: jax.Array,
        logits: jax.Array, # BS, models, out
        labels: jax.Array # BS
    ):
        prob = jax.nn.softmax(logits)
        target_mask = jnp.eye(logits.shape[-1])[labels]
        
        correct_p = jnp.take_along_axis(prob, labels[:, jnp.newaxis, jnp.newaxis], axis=-1) 
        # (BS, models, 1)

        wrong_p = prob * (1 - target_mask[:, jnp.newaxis, :])
        # (BS, models, out_dim)

        weights = (1.0 - correct_p) * ensemble_weights[:, :, jnp.newaxis]
        weights = jax.lax.stop_gradient(weights)
        w_norm = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-8)
        center = jnp.sum(w_norm * wrong_p, axis=1, keepdims=True) 
        diff_sq = (wrong_p - center) ** 2
        masked_diff_sq = weights * diff_sq
        ediv = jnp.mean(masked_diff_sq)
        return ediv
    
    def error_correlation(
        ensemble_weights: jax.Array,
        y: jax.Array,
        ensemble_ys: jax.Array,
    ):
        errors = ensemble_ys - y[:, jnp.newaxis, :]
        
        cov = jnp.einsum('bmi,bni->bmn', errors, errors)
        std = jnp.sqrt(jnp.sum(errors ** 2, axis=-1)) + 1e-8
        corr = cov / jnp.einsum('bm,bn->bmn', std, std)
        
        weights_matrix = jnp.einsum('bm,bn->bmn', ensemble_weights, ensemble_weights)
        
        return jnp.mean(corr * weights_matrix)

    if task_type == "classification":
        diversity = error_diversity_classification(train_return.power, ensemble_logits, inout.y)
        loss_batch = ce_loss_logprobs_labels(logprobs, inout.y)
    elif task_type == "regression":
        diversity = error_correlation(ensemble_weights, yhat, ensemble_ys)
        loss_batch = mse_loss(yhat, inout.y)
    
    diversity_loss = -(DIVERSITY_LAMBDA * diversity)

    performance_loss = jnp.mean(loss_batch)
    auxillary_losses = train_params.learner.auxillary_losses(
        key=k_loss,
        model=model,
        params=network_params,
        train_return=train_return,
        train_params=train_params
    )

    performance_loss_name = "ce_loss" if task_type == "classification" else "mse_loss"

    metrics = {
        performance_loss_name: performance_loss,
        "div_loss": diversity_loss
    } | auxillary_losses


    aux_values = jax.tree.reduce_associative(operator.add, auxillary_losses)
    loss = aux_values + performance_loss + diversity_loss

    def calc_accuracy(yhat, y):
        hat_inds = jnp.argmax(yhat, axis=-1)
        return jnp.mean(hat_inds == y)

    def one_sample_kl(combiner: jax.Array, ensemble_weights: jax.Array, individuals: jax.Array):
        kl_per_model = jax.scipy.special.kl_div(combiner[jnp.newaxis, :], individuals).sum(axis=-1)
        return jnp.sum(kl_per_model * ensemble_weights) # Only count the models which participated in the decision

    if calc_metric:
        metrics["power_entropy_fig1"] = jnp.mean(jnp.sum(jax.scipy.special.entr(ensemble_weights + 1e-8), axis=-1)) / jnp.log(ensemble_weights.shape[-1])
        
        if task_type == "classification":

            # optimal_power = get_optimal_weights(train_return.ys, inout.y, train_return.power)
            # optimal_yhat = mix_weighted_logits(train_return.ys, optimal_power)
            metrics["accuracy_metric"] = calc_accuracy(logprobs, inout.y)
            # metrics["optimal_ce_loss"] = jnp.mean(ce_loss(optimal_yhat, inout.y))
            metrics["kl_fig2"] = jnp.mean(jax.vmap(one_sample_kl)(
                jax.nn.softmax(logprobs) + 1e-6,
                ensemble_weights,
                jax.nn.softmax(ensemble_logprobs) + 1e-6
            ))
            metrics["prediction_entropy_fig1"] = jnp.mean(jnp.sum(jax.scipy.special.entr(nn.softmax(ensemble_logits)), axis=-1)) / jnp.log(train_return.ys.shape[-1])
        
        elif task_type == "regression":
            ...

    return loss, metrics


@partial(jax.jit, static_argnames=("optimizer", "model", "train_params"))
def train_batch(
    key: jax.Array,
    inout_data: InOutData,  
    model_params: dict,
    opt_state: dict,
    optimizer: optax.GradientTransformationExtraArgs,
    model: nn.Module,
    train_params: TrainParams
):
    
    # To batches
    bs = train_params.batch_size
    assert (inout_data.x.shape[0] % bs) == 0, "Gpu batch needs to be divisible by the batch_size"
    n_batches = inout_data.x.shape[0] // bs
    inout_data = jax.tree.map(lambda x: x.reshape((n_batches, bs) + x.shape[1:]), inout_data)
    
    def train_step(carry, inout_batch: InOutData):
        
        key, network_params, opt_state = carry
        key, k_use = jax.random.split(key)


        (loss, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            network_params,
            key=k_use,
            inout=inout_batch,
            model=model,
            train_params=train_params
        )

        updates, opt_state = optimizer.update(grad, opt_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return (key, network_params, opt_state), (loss, metrics)
    

    (key, model_params, opt_state), (loss, metrics) = jax.lax.scan(train_step, (key, model_params, opt_state), inout_data)

    loss = jnp.mean(loss)
    metrics = jax.tree.map(jnp.mean, metrics)

    return (model_params, opt_state), (loss, metrics)


def train_loader(
    key: jax.Array,
    inout: InOutData,
    batch_size: int,
    desired_batches: int
):
    
    desired_size = batch_size * desired_batches
    actual_size = inout.x.shape[0]
    difference = desired_size - actual_size
    assert difference < actual_size

    k1, k2, k_next = jax.random.split(key, 3)

    proper_inds = jax.random.permutation(k1, actual_size)
    added_inds = jax.random.permutation(k2, difference) 
    
    all_inds = jnp.concatenate((proper_inds, added_inds))
    
    for i_batch in range(desired_batches):
        start = i_batch * batch_size
        end = start + batch_size

        batch_inds = all_inds[start:end]
        batch = jax.tree.map(lambda x: x[batch_inds, ...], inout)

        yield batch, k_next
    

def orthogonalize_expert_grads(last_layers: int = 1):
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        max_layer = -1
        if "predictors" in updates:
            for key in updates["predictors"].keys():
                layer = int(key.split("_")[-1])
                max_layer = max(max_layer, layer)

        target_layers = tuple([f"body_layers_{max_layer - i}" for i in range(1, last_layers + 1)])
        def process_update(key_path, update):
            path_str = "/".join(str(k) for k in key_path)

            if any(target in path_str for target in target_layers) and ("kernel" in path_str) and ("predictors" in path_str):

                E, I, O = update.shape
                flat_update = update.reshape(E, I * O)
                
                orig_norm = jnp.linalg.norm(flat_update, axis=1, keepdims=True)
                
                A = flat_update.T
                q, _ = jnp.linalg.qr(A, mode='reduced')
                
                eps = 1e-8
                q_scaled = q.T * (orig_norm + eps)
                
                return q_scaled.reshape(E, I, O)

            return update

        new_updates = jax.tree.map_with_path(process_update, updates)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)

def train(
        key: jax.Array,
        train_params: TrainParams,
        trial: object | None = None
    ):
    
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]
    

    key, k_init, k_loop, k_loader = jax.random.split(key, 4) 

    # Data
    fullData = train_params.task.load_cpu(split="train")
    x, y = train_params.task.get_xy(fullData)
    inout_data = InOutData(
        x=x, y=y
    )
    gpu_batch = train_params.batch_size * train_params.preload_batches_to_gpu
    n_data = x.shape[0]
    n_batches = math.ceil(n_data / gpu_batch)

    # Train, Valid split
    n_valid = train_params.batch_size * train_params.valid_batches
    inout_valid = jax.tree.map(lambda x: jax.device_put(x[:n_valid, ...], device=gpu), inout_data)
    inout_train = jax.tree.map(lambda x: x[n_valid:, ...], inout_data)

    # Model
    dummy_input = x[[0], ...]

    if trial is None:
        model = train_params.learner.get_model(
            train_params=train_params,
            param_budget=train_params.param_budget,
            dummy_input=dummy_input
        )
    else:
        model = train_params.learner.boot_from_trial(
            train_params=train_params,
            dummy_input=dummy_input,
            trial=trial
        )

    model_params = model.init(k_init, dummy_input)["params"]


    # Optimizer
    optimizer_pure = {
        "sgd": optax.sgd(learning_rate=train_params.lr),
        "adam": optax.adamw(learning_rate=train_params.lr, weight_decay=1e-3)
    }[train_params.optimizer]

    optimizer = optimizer_pure
    opt_state = optimizer.init(model_params)
    
    metrics = defaultdict(list)

    for i_epoch in tqdm.tqdm(range(train_params.epochs)):

        epoch_metrics = defaultdict(list) 

        for inout_batch, k_loader in train_loader(
                k_loader,
                inout_train,
                batch_size=gpu_batch,
                desired_batches=n_batches
            ):


            k_loop, k_use = jax.random.split(k_loop)
            inout_batch = jax.tree.map(lambda x: jax.device_put(x, device=gpu), inout_batch)
            
            (model_params, opt_state), (loss, tr_metrics) = train_batch(
                key=k_use,
                inout_data=inout_batch,
                model_params=model_params,
                opt_state=opt_state,
                optimizer=optimizer,
                model=model,
                train_params=train_params,
            )

            epoch_metrics["loss"].append(loss)
            for k, v in tr_metrics.items():
                epoch_metrics[k].append(v.item())

        k_loop, k_use = jax.random.split(k_loop)

        va_loss, va_metrics = loss_fn(
            model_params,
            k_use,
            inout_valid,
            model,
            train_params,
            calc_metric=True
        )
        metrics["validation_loss"].append(va_loss.item())
        for k, v in va_metrics.items():
            metrics[f"validation_{k}"].append(v.item())

        for k, v in epoch_metrics.items():
            metrics[k].append(np.mean(v).item())

    
    return metrics

def make_train_folder(experiment_name: str) -> Path:
    exp_folder = Path(__file__).parent / "runs"
    exp_folder.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = exp_folder / f"{experiment_name}_{timestamp}"
    folder.mkdir()
    return folder

def finish_run(metrics: dict[str, list[float]], folder: Path, prefix: str = ""):
    try:
        save_metrics(metrics, folder, prefix)
    except Exception:
        print("Error during metric saving")
        traceback.print_exc()

    try:
        plot_losses_and_metrics(metrics, folder, prefix)
    except Exception:
        print("Error during plotting losses")
        traceback.print_exc()

def save_metrics(metrics: dict[str, list[float]], folder: Path, prefix: str = ""):
    with open(folder / f"{prefix}_metrics.json", mode="w") as f:
        json.dump(metrics, f)
    

def plot_losses_and_metrics(metrics: dict[str, list[float]], folder: Path, prefix: str = ""):

    from matplotlib import pyplot as plt
    from copy import deepcopy

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    metrics = deepcopy(metrics)
    v_loss = metrics.pop("validation_loss")
    t_loss = metrics.pop("loss")

    n_additional_figures = 0
    for name in metrics.keys():
        suffix = name.split("_")[-1]
        if "fig" not in suffix:
            continue
        fignum = int(suffix.replace("fig", ""))
        n_additional_figures = max(fignum, n_additional_figures)

    
    n_all_figures = 2 + n_additional_figures
    cols = 2
    rows = math.ceil(n_all_figures / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.ravel()
    ax_metrics, ax_losses = axes[0], axes[1]
    rest = axes[2:]

    used = 0
    for name in metrics.keys():
        
        if not name.endswith("metric"):
            continue

        ax_metrics.plot(metrics[name], color=colors[used], label=name.removesuffix("_metric"))

        used += 1

    ax_metrics.legend()

    used = 0
    for name in metrics.keys():
        
        if name.startswith("validation") and name.endswith("loss"):
            tname = name.removeprefix("validation_")
        else:
            continue

        ax_losses.plot(metrics[name], color=colors[used], label=tname)

        if tname in metrics:
            ax_losses.plot(metrics[tname], color=colors[used], linestyle="dashed")

        used += 1
    ax_losses.legend()


    for name in metrics.keys():
        suffix = name.split("_")[-1]
        if "fig" not in suffix:
            continue
        fignum = int(suffix.replace("fig", ""))
        figindex = fignum - 1
        rest[figindex].plot(metrics[name], label=name)

    for ax in rest:
        ax.legend()

    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_losses.png")
    

if __name__ == "__main__":


    folder = make_train_folder("bikes")
    learners = [LeLearner]
    key = jax.random.key(123)
    param_budget = None
    for learner in learners:
        metrics = train(
            key=key,
            train_params=replace(
                g_params,
                learner=learner,
                param_budget=param_budget
            )
        )
        finish_run(metrics, folder, prefix=learner.__name__)