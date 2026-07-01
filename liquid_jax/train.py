

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
from architectures import Ensemble

from structs import InOutData, TrainParams, ForwardArgs
from utils import train_loader
from math_utils import loss as loss_fn

from atomic_networks import three_layer_mlp, two_layer_mlp, small_cnn, big_cnn

CurrentTask = Cifar10
n_delegators = 5
n_predictors = 10

g_params = TrainParams(
    batch_size=64,
    preload_batches_to_gpu=20,
    valid_batches=4,
    epochs=20,
    lr=1e-3,
    task=CurrentTask,
    n_predictors=n_predictors,
    n_delegators=n_delegators,
    delegators_mixing="sum",
    ambiguity_gradient="delegators",
    architecture=small_cnn.determine_size(
        predictor_base=2,
        delegator_base=1,
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
            x=inout_batch.x,
            y=inout_batch.y,
            ensemble_model=ensemble,
            train_params=train_params
        )

        updates, opt_state = optimizer.update(grad, opt_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return (key, network_params, opt_state), (loss, metrics)
    

    (key, ensemble_params, opt_state), (loss, metrics) = jax.lax.scan(train_step, (key, ensemble_params, opt_state), inout_data)

    loss = jnp.mean(loss)
    metrics = jax.tree.map(jnp.mean, metrics)

    return (ensemble_params, opt_state), (loss, metrics)

    

def train(
        key: jax.Array,
        train_params: TrainParams,
        trial: object | None = None
    ):
    
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
    
    ensemble_params = ensemble.init(k_init, dummy_input)["params"]
    opt_state = optimizer.init(ensemble_params)
    
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
            
            (ensemble_params, opt_state), (loss, tr_metrics) = train_batch(
                key=k_use,
                inout_data=inout_batch,
                ensemble_params=ensemble_params,
                opt_state=opt_state,
                optimizer=optimizer,
                ensemble=ensemble,
                train_params=train_params,
            )

            epoch_metrics["loss"].append(loss)
            for k, v in tr_metrics.items():
                epoch_metrics[k].append(v.item())
     
        k_loop, k_use = jax.random.split(k_loop)

        va_loss, va_metrics = loss_fn(
            ensemble_params=ensemble_params,
            key=k_use,
            train_params=train_params,
            ensemble_model=ensemble,
            x=inout_valid.x,
            y=inout_valid.y
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
        
        if name.startswith("validation") and name.endswith("metric"):
            tname = name.removeprefix("validation_")
        else:
            continue

        ax_metrics.plot(metrics[name], color=colors[used], label=name.removesuffix("_metric"))

        if tname in metrics:
            ax_metrics.plot(metrics[tname], color=colors[used], linestyle="dashed")


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


    folder = make_train_folder("paper")
    key = jax.random.key(123)
    metrics = train(
        key=key,
        train_params=g_params
    )
    finish_run(metrics, folder, prefix="paper")