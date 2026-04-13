

from collections import defaultdict
import datetime
from functools import partial
import json
import math
from pathlib import Path
from dataclasses import replace
import traceback

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
from learner_le import LeLearner
from learner_de import DeLearner

from structs import TrainParams


g_params = TrainParams(
    batch_size=512,
    preload_batches_to_gpu=5,
    valid_batches=3,
    epochs=30,
    lr=1e-3,
    optimizer="adam",
    performance_loss="ce",
    task=Cifar10,
    n_models_in_ensemble=25,
    learner=LeLearner
)

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

    yhat, train_return, cos_sim = train_params.learner.forward(
        key=k_forward,
        x=inout.x,
        model=model,
        params=network_params
    )

    if train_params.performance_loss == "ce":
        loss_batch = optax.softmax_cross_entropy_with_integer_labels(
            yhat, inout.y
        )
    else:
        raise ValueError(f"loss={train_params.performance_loss} is not implemented")
    
    
    performance_loss = jnp.mean(loss_batch)
    auxillary_losses = train_params.learner.auxillary_losses(
        key=k_loss,
        model=model,
        params=network_params,
        train_return=train_return
    )
    metrics = {f"{train_params.performance_loss}_loss": performance_loss} | auxillary_losses

    aux_values = jax.tree.reduce(lambda accum, aux_loss: accum + aux_loss, auxillary_losses, initializer=0.0)
    loss = aux_values + performance_loss

    metrics["cosine_sim_metric"] = cos_sim

    if calc_metric:
        if (tt := train_params.task.task_type()) == "classification":
            hat_inds = jnp.argmax(yhat, axis=-1)
            assert hat_inds.shape == inout.y.shape
            metrics["accuracy_metric"] = jnp.mean(hat_inds == inout.y)

        elif tt == "regression":
            assert yhat.shape == inout.y.shape
            metrics["mae_metric"] = jnp.mean(jnp.abs(yhat - inout.y))
        

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
    optimizer = {
        "sgd": optax.sgd(learning_rate=train_params.lr),
        "adam": optax.adamw(learning_rate=train_params.lr)
    }[train_params.optimizer]
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
    
    fig, (ax_metrics, ax_losses) = plt.subplots(1, 2, figsize=(14, 5))
    
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
        ax_losses.plot(metrics[tname], color=colors[used], linestyle="dashed")

        used += 1
    
    ax_losses.legend()
    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_losses.png")
    

if __name__ == "__main__":


    folder = make_train_folder("check_orthogonality")
    learners = [LeLearner]
    key = jax.random.key(123)
    param_budget = 200_000
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