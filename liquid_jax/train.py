

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
from architectures import Ensemble, split_ensemble

from structs import InOutData, TrainParams, ForwardArgs
from utils import train_loader
from math_utils import loss as loss_fn
from evaluation import get_evaluation_metrics

from atomic_networks import three_layer_mlp, two_layer_mlp, small_cnn, big_cnn

PROFILER = False

CurrentTask = Cifar10
n_delegators = 5
n_predictors = 10

g_params = TrainParams(
    batch_size=64,
    preload_batches_to_gpu=50,
    valid_batches=10,
    epochs=50,
    lr=1e-3,
    task=CurrentTask,
    n_predictors=n_predictors,
    n_delegators=n_delegators,
    delegators_mixing="sum",
    ambiguity_gradient="delegators",
    architecture=small_cnn.determine_size(
        predictor_base=4,
        delegator_base=2,
        out_dim=CurrentTask.out_dim(),
        n_predictors=n_predictors
    )
)

@partial(jax.jit, static_argnames=("optimizer", "ensemble", "train_params"))
def train_batch(
    key: jax.Array,
    inout_data: InOutData,
    ensemble_params: dict,
    opt_state: dict, 
    optimizer: optax.GradientTransformationExtraArgs,
    ensemble: Ensemble,
    train_params: TrainParams,
):
    bs = train_params.batch_size
    assert (inout_data.x.shape[0] % bs) == 0, (
        "Gpu batch needs to be divisible by the batch_size"
    )

    n_batches = inout_data.x.shape[0] // bs
    inout_data = jax.tree.map(
        lambda x: x.reshape((n_batches, bs) + x.shape[1:]),
        inout_data,
    )

    def one_seed_loss(
        params: dict,
        seed_key: jax.Array,
        inout_batch: InOutData,
    ):
        return loss_fn(
            params,
            key=seed_key,
            x=inout_batch.x,
            y=inout_batch.y,
            ensemble_model=ensemble,
            train_params=train_params,
        )

    one_seed_value_and_grad = jax.value_and_grad(
        one_seed_loss,
        has_aux=True,
    )

    def one_seed_update(grad, state, params):
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    def train_step(carry, inout_batch: InOutData):
        keys, network_params, opt_state = carry
        keys, k_use = split_seed_keys(keys)

        (loss, metrics), grad = jax.vmap(
            one_seed_value_and_grad,
            in_axes=(0, 0, None),
        )(
            network_params,
            k_use,
            inout_batch,
        )

        network_params, opt_state = jax.vmap(one_seed_update)(
            grad,
            opt_state,
            network_params,
        )

        return (
            keys,
            network_params,
            opt_state,
        ), (
            loss,
            metrics,
        )

    (_, ensemble_params, opt_state), (loss, metrics) = jax.lax.scan(
        train_step,
        (key, ensemble_params, opt_state),
        inout_data,
    )

    # Before reduction (n_batches, n_seeds)
    loss = jnp.mean(loss, axis=0)
    metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)

    return (ensemble_params, opt_state), (loss, metrics)



def split_seed_keys(keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    pairs = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
    return pairs[:, 0], pairs[:, 1]


def train(
        key: jax.Array,
        train_params: TrainParams,
        profile_dir: Path,
        profile_batches: int = 5,
        n_seeds: int = 5,
    ):
    
    gpu = jax.devices("gpu")[0]

    key, k_loop, k_loader, k_seeds, k_eval = jax.random.split(key, 5) 

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
    inout_valid: InOutData = jax.tree.map(lambda x: jax.device_put(x[:n_valid, ...], device=gpu), inout_data)
    inout_train: InOutData = jax.tree.map(lambda x: x[n_valid:, ...], inout_data)

    # Model
    dummy_input = ForwardArgs(x=x[[0], ...])

    ensemble = Ensemble(
        n_predictors=train_params.n_predictors,
        n_delegators=train_params.n_delegators,
        predictor=train_params.architecture.predictor,
        delegator=train_params.architecture.delegator,
        n_cnn_layers=train_params.architecture.cnn   
    )

    optimizer = optax.adamw(learning_rate=train_params.lr)
    
    def init_one(init_key):
        return ensemble.init(init_key, dummy_input)["params"]

    seed_keys = jax.random.split(k_seeds, n_seeds)
    k_init, k_loop = split_seed_keys(seed_keys)
    ensemble_params = jax.vmap(init_one)(k_init)
    opt_state = jax.vmap(optimizer.init)(ensemble_params)
    
    metrics = defaultdict(list)

    global_step = 0

    def validate_one(params, seed_key):
        return loss_fn(
            ensemble_params=params,
            key=seed_key,
            train_params=train_params,
            ensemble_model=ensemble,
            x=inout_valid.x,
            y=inout_valid.y,
        )
    validate_seeds = jax.vmap(validate_one)


    for i_epoch in tqdm.tqdm(range(train_params.epochs)):

        epoch_metrics = defaultdict(list) 

        for inout_batch, k_loader in train_loader(
                k_loader,
                inout_train,
                batch_size=gpu_batch,
                desired_batches=n_batches
            ):


            k_loop, k_use = split_seed_keys(k_loop)
            inout_batch = jax.tree.map(lambda x: jax.device_put(x, device=gpu), inout_batch)


            start_profile = profile_dir is not None and global_step == 1
            stop_profile = (
                profile_dir is not None
                and global_step == 1 + profile_batches
            )

            if PROFILER and start_profile:
                jax.profiler.start_trace(str(profile_dir))

            if PROFILER:
                with jax.profiler.StepTraceAnnotation(
                    "train_step",
                    step_num=global_step,
                ):
            
                    (ensemble_params, opt_state), (loss, tr_metrics) = train_batch(
                        key=k_use,
                        inout_data=inout_batch,
                        ensemble_params=ensemble_params,
                        opt_state=opt_state,
                        optimizer=optimizer,
                        ensemble=ensemble,
                        train_params=train_params,
                    )
                    jax.block_until_ready(loss)
            else:
                (ensemble_params, opt_state), (loss, tr_metrics) = train_batch(
                    key=k_use,
                    inout_data=inout_batch,
                    ensemble_params=ensemble_params,
                    opt_state=opt_state,
                    optimizer=optimizer,
                    ensemble=ensemble,
                    train_params=train_params,
                )

            if PROFILER and stop_profile:
                jax.profiler.stop_trace()
            
            global_step += 1

            epoch_metrics["loss"].append(np.asarray(loss))

            for k, v in tr_metrics.items():
                epoch_metrics[k].append(np.asarray(v))
            
     
        k_loop, k_use = split_seed_keys(k_loop)

        va_loss, va_metrics = validate_seeds(
            ensemble_params,
            k_use,
        )
        metrics["validation_loss"].append(np.asarray(va_loss))

        for k, v in va_metrics.items():
            metrics[f"validation_{k}"].append(np.asarray(v))

        for k, values in epoch_metrics.items():
            values = np.stack(values, axis=0)
            metrics[k].append(values.mean(axis=0))

    metrics = {
        name: np.stack(values, axis=0)
        for name, values in metrics.items()
    }
    

    (predictors, predictors_params), (delegators, delegators_params) = split_ensemble(ensemble, ensemble_params)

    get_evaluation_metrics(
        key=k_eval,
        delegators=delegators,
        delegators_params=delegators_params,
        predictors=predictors,
        predictors_params=predictors_params,
        inout_train_predictions=inout_train,
        inout_valid_predictions=inout_valid,
        train_params=train_params,
        use_seed=0
    )

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
    json_metrics = {
        name: values.tolist()
        for name, values in metrics.items()
    }

    with open(folder / f"{prefix}_metrics.json", mode="w") as f:
        json.dump(json_metrics, f)
    

def plot_seed_summary(
    ax,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = "solid",
):
    values = np.asarray(values)

    mean = values.mean(axis=1)
    lower, upper = np.quantile(
        values,
        [0.025, 0.975],
        axis=1,
    )

    epochs = np.arange(len(mean))

    ax.plot(
        epochs,
        mean,
        color=color,
        linestyle=linestyle,
        label=label,
    )
    ax.fill_between(
        epochs,
        lower,
        upper,
        color=color,
        alpha=0.2,
    )

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

        if name.startswith("validation") and name.endswith("metric"):
            tname = name.removeprefix("validation_")
        else:
            continue

        color = colors[used % len(colors)]
        label = name.removeprefix("validation_").removesuffix("_metric")

        plot_seed_summary(
            ax_metrics,
            metrics[name],
            label=f"validation {label}",
            color=color,
        )

        if tname in metrics:
            plot_seed_summary(
                ax_metrics,
                metrics[tname],
                label=f"train {label}",
                color=color,
                linestyle="dashed",
            )

        used += 1

    ax_metrics.legend()

    used = 0
    for name in metrics.keys():

        if name.startswith("validation") and name.endswith("loss"):
            tname = name.removeprefix("validation_")
        else:
            continue

        color = colors[used % len(colors)]

        plot_seed_summary(
            ax_losses,
            metrics[name],
            label=f"validation {tname}",
            color=color,
        )

        if tname in metrics:
            plot_seed_summary(
                ax_losses,
                metrics[tname],
                label=f"train {tname}",
                color=color,
                linestyle="dashed",
            )

        used += 1

    ax_losses.legend()


    for name in metrics.keys():
        suffix = name.split("_")[-1]
        if "fig" not in suffix:
            continue
        fignum = int(suffix.replace("fig", ""))
        figindex = fignum - 1
        plot_seed_summary(
            rest[figindex],
            metrics[name],
            label=name,
            color=colors[figindex % len(colors)],
        )

    for ax in rest:
        ax.legend()

    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_losses.png")
    

if __name__ == "__main__":


    folder = make_train_folder("profile_single_seed")
    key = jax.random.key(123)
    metrics = train(
        key=key,
        train_params=g_params,
        profile_dir=folder
    )
    finish_run(metrics, folder, prefix="paper")