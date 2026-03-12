"""
Ablation experiment: fixed parameter budget per expert, sweep body hidden width.

Tests:
    1. With vs without body (h_body=0 means no body, heads take raw input)
    2. With vs without skip connection (concatenate raw input to body output before heads)
    3. Full retraining for each configuration
    4. Each component has: input → Dense(hidden) → ReLU → Dense(output)
    5. Fixed total param budget: shrinking body redistributes params to heads uniformly

Final plot: x=body hidden width, y=performance (validation loss)
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax import linen as nn

from mnist import Mnist
from liquid_solver import LEsolver, LEInfo
from learner_base import Learner
from learner_mnist_le import get_layers, forward as fwd_layers
from structs import TrainParams
from train import train, plot_losses


# ---------------------------------------------------------------------------
# Model with optional skip connection
# ---------------------------------------------------------------------------

class DeModelMlpSkip(nn.Module):
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None
    skip: bool = False

    def setup(self):
        self.body_layers = get_layers(self.body)
        self.out_layers = get_layers(self.out)
        self.delegation_layers = get_layers(self.delegation)

    def __call__(self, x: jax.Array):
        h_body = fwd_layers(x, last_linear=False, layers=self.body_layers)

        if self.skip and self.body_layers:
            h_body = jnp.concatenate([h_body, x], axis=-1)

        y = fwd_layers(h_body, last_linear=True, layers=self.out_layers)
        d = fwd_layers(h_body, last_linear=True, layers=self.delegation_layers)
        return y, d


class LeMlpSkip(nn.Module):
    n_models: int
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None
    skip: bool = False

    def setup(self):
        assert self.delegation[-1] == self.n_models
        Vmapped = nn.vmap(
            DeModelMlpSkip,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            axis_size=self.n_models,
            out_axes=1,
        )
        self.ensemble = Vmapped(
            out=self.out, delegation=self.delegation,
            body=self.body, skip=self.skip,
        )

    def __call__(self, x: jax.Array):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        y, d = self.ensemble(x)
        d = nn.softmax(d, axis=-1)
        return y, d


# ---------------------------------------------------------------------------
# Parameter budget helpers
# ---------------------------------------------------------------------------

def count_component_params(in_dim: int, hidden: int, out_dim: int) -> int:
    """Params for: in_dim → Dense(hidden) → ReLU → Dense(out_dim)"""
    return hidden * (in_dim + 1) + out_dim * (hidden + 1)


def solve_head_hidden(budget: float, in_dim: int, out_dim: int) -> int:
    """Solve for hidden dim: h = (budget - out_dim) / (in_dim + 1 + out_dim)"""
    h = (budget - out_dim) / (in_dim + 1 + out_dim)
    return max(1, int(h))


def max_body_hidden(total_budget: int, input_dim: int, body_out_dim: int) -> int:
    """Largest h_body that fits entirely in the budget (leaving 0 for heads)."""
    return int(total_budget / (input_dim + 1 + body_out_dim))


def solve_config(
    h_body: int,
    total_budget: int,
    input_dim: int = 784,
    body_out_dim: int = 64,
    n_classes: int = 10,
    n_models: int = 10,
    skip: bool = False,
):
    """Given body hidden width, solve head widths from remaining budget.

    Returns dict with architecture tuples and param counts, or None if infeasible.
    """
    if h_body == 0:
        body = None
        body_params = 0
        head_in = input_dim
        use_skip = False
    else:
        body = (h_body, body_out_dim)
        body_params = count_component_params(input_dim, h_body, body_out_dim)
        head_in = (body_out_dim + input_dim) if skip else body_out_dim
        use_skip = skip

    remaining = total_budget - body_params
    if remaining <= 0:
        return None

    head_budget = remaining / 2
    h_out = solve_head_hidden(head_budget, head_in, n_classes)
    h_del = solve_head_hidden(head_budget, head_in, n_models)

    if h_out < 1 or h_del < 1:
        return None

    out = (h_out, n_classes)
    delegation = (h_del, n_models)

    actual = (
        body_params
        + count_component_params(head_in, h_out, n_classes)
        + count_component_params(head_in, h_del, n_models)
    )

    return {
        "h_body": h_body,
        "h_out": h_out,
        "h_del": h_del,
        "body": body,
        "out": out,
        "delegation": delegation,
        "skip": use_skip,
        "actual_params": actual,
        "body_params": body_params,
        "head_in": head_in,
    }


# ---------------------------------------------------------------------------
# Learner factory
# ---------------------------------------------------------------------------

solver = LEsolver()


def make_learner(n_models, body, out, delegation, skip=False):
    """Return a Learner class configured with the given architecture."""

    class _Learner(Learner[LEInfo]):
        @staticmethod
        def get_model():
            return LeMlpSkip(
                n_models=n_models, body=body, out=out,
                delegation=delegation, skip=skip,
            )

        @staticmethod
        def forward(key, x, model, params):
            ys, deleg = model.apply({"params": params}, x)
            leinfo = solver.solve_power(deleg)
            y = solver.mix_power_logits(ys, leinfo.power)
            return y, leinfo

        @staticmethod
        def auxillary_losses(key, train_return):
            return {
                "load_distribution_loss": solver.load_distribution_loss(train_return),
                "specialization_losss": solver.specialization_loss(train_return),
            }

    return _Learner


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_ablation(
    total_budget: int = 50_000,
    body_out_dim: int = 64,
    n_models: int = 10,
    skip: bool = False,
    h_body_values: list[int] | None = None,
    n_points: int = 10,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 42,
):
    """Sweep body hidden width, train each config, return results.

    Args:
        total_budget: total parameter budget per expert
        body_out_dim: fixed output dim of body (interface to heads)
        n_models: number of experts in the ensemble
        skip: whether to use skip connections (concat input to body output)
        h_body_values: explicit list of body hidden widths to test.
            If None, generates n_points values from 0 to max.
        n_points: number of sweep points (used when h_body_values is None)
        epochs: training epochs per configuration
        lr: learning rate
        batch_size: batch size
        seed: random seed
    """
    input_dim = 784
    n_classes = 10

    # Build sweep values
    if h_body_values is None:
        h_max = max_body_hidden(total_budget, input_dim, body_out_dim)
        step = max(1, h_max // n_points)
        h_body_values = [0] + list(range(1, h_max + 1, step))
        if h_max not in h_body_values:
            h_body_values.append(h_max)

    # Solve configs
    configs = []
    for h in h_body_values:
        c = solve_config(
            h, total_budget, input_dim, body_out_dim,
            n_classes, n_models, skip,
        )
        if c is not None:
            configs.append(c)

    print(f"Running {len(configs)} configurations (budget={total_budget}, skip={skip}):")
    for c in configs:
        print(
            f"  h_body={c['h_body']:4d} | h_out={c['h_out']:4d} | h_del={c['h_del']:4d} | "
            f"params={c['actual_params']:6d} (body={c['body_params']})"
        )

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{timestamp}"
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)

    # Run each config
    results = []
    key = jax.random.key(seed)

    for c in tqdm.tqdm(configs, desc="Ablation sweep"):
        key, k_train = jax.random.split(key)

        learner_cls = make_learner(
            n_models=n_models,
            body=c["body"],
            out=c["out"],
            delegation=c["delegation"],
            skip=c["skip"],
        )

        params = TrainParams(
            batch_size=batch_size,
            preload_batches_to_gpu=5,
            valid_batches=2,
            epochs=epochs,
            lr=lr,
            optimizer="adam",
            performance_loss="ce",
            task=Mnist,
            learner=learner_cls,
        )

        metrics = train(k_train, params)

        result = {
            **c,
            "final_val_loss": metrics["validation_loss"][-1],
            "final_train_loss": float(metrics["loss"][-1]),
            "best_val_loss": min(metrics["validation_loss"]),
            "metrics": metrics,
        }
        results.append(result)

        # Save after each run in case of crash
        _save_results(results, exp_dir)

    plot_ablation(results, exp_dir)
    return results


# ---------------------------------------------------------------------------
# Saving & plotting
# ---------------------------------------------------------------------------

def _save_results(results, exp_dir):
    summary = [{k: v for k, v in r.items() if k != "metrics"} for r in results]
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def plot_ablation(results, exp_dir):
    from matplotlib import pyplot as plt

    h_body = [r["h_body"] for r in results]
    val_loss = [r["best_val_loss"] for r in results]
    train_loss = [r["final_train_loss"] for r in results]

    # Main plot: body width vs performance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_body, val_loss, "o-", label="Best validation loss", color="black")
    ax.plot(h_body, train_loss, "s--", label="Final train loss", color="gray")
    ax.set_xlabel("Body hidden width")
    ax.set_ylabel("Loss")
    ax.set_title("Ablation: body width vs performance (fixed param budget)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/plots/ablation.png", dpi=150)
    plt.show()

    # Training curves per config
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(r["metrics"]["validation_loss"], label=f"h_body={r['h_body']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Training curves per configuration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/plots/training_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    results = run_ablation(
        total_budget=50_000,
        body_out_dim=64,
        n_models=10,
        skip=False,
        epochs=20,
        lr=1e-3,
    )
