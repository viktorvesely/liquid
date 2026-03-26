from datetime import datetime
import os

import jax
import jax.numpy as jnp
import tqdm
from flax import linen as nn

from mnist import Mnist
from liquid_solver import LEsolver, LEInfo
from learner_base import Learner
from learner_mnist_le import get_layers, forward as fwd_layers
from structs import TrainParams
from train import train

import yaml
import pandas as pd
from plotting import plot_ablation

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


def count_component_params(in_dim: int, hidden: int, out_dim: int) -> int:
    """Params for: in_dim → Dense(hidden) → ReLU → Dense(out_dim)"""
    return hidden * (in_dim + 1) + out_dim * (hidden + 1)


def solve_head_hidden(budget: float, in_dim: int, out_dim: int) -> int:
    """Solve for hidden dim: h = (budget - out_dim) / (in_dim + 1 + out_dim)"""
    h = round((budget - out_dim) / (in_dim + 1 + out_dim))
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


solver = LEsolver(
    load_distribution_lambda=0.1,
    specialization_lambda=0,
)


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
    exp_dir = f"experiments/ablation/{timestamp}"
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)

    # Save experiment config
    exp_config = {
        "total_budget": total_budget,
        "body_out_dim": body_out_dim,
        "n_models": n_models,
        "skip": skip,
        "h_body_values": [c["h_body"] for c in configs],
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "n_configs": len(configs),
    }
    with open(f"{exp_dir}/config.yaml", "w") as f:
        yaml.dump(exp_config, f, default_flow_style=False)

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

        # Verify param count: expected (from solve_config) vs actual (from model init)
        model = learner_cls.get_model()
        dummy = jnp.zeros((1, input_dim))
        init_params = model.init(jax.random.key(0), dummy)["params"]
        # Each leaf has shape (n_models, ...) from vmap — first dim is the expert axis
        per_expert = sum(p[0].size for p in jax.tree.leaves(init_params))
        expected = c["actual_params"]
        print(
            f"  h_body={c['h_body']:4d} | "
            f"expected={expected:6d} | actual={per_expert:6d} | "
            f"match={'OK' if per_expert == expected else 'MISMATCH'}"
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
            "best_val_ce_loss": min(metrics["validation_ce_loss"]),
            "final_val_ce_loss": metrics["validation_ce_loss"][-1],
            "metrics": metrics,
        }
        results.append(result)

        # Save after each run in case of crash
        _save_results(results, exp_dir)

    plot_ablation(results, exp_dir)
    return results

def _save_results(results, exp_dir):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "metrics"} for r in results])
    df.to_parquet(f"{exp_dir}/results.parquet")

if __name__ == "__main__":
    results = run_ablation(
        total_budget=50_000,
        body_out_dim=64,
        n_models=10,
        skip=True,
        epochs=20,
        lr=1e-3,
    )
