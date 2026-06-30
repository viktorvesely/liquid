from __future__ import annotations
from dataclasses import asdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct, serialization
from flax.training import train_state
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go
from tqdm import tqdm


class LinearPredictor(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(1)(x)


class PredictorEnsemble(nn.Module):
    n_models: int

    @nn.compact
    def __call__(self, x: jax.Array):
        ensemble = nn.vmap(
            LinearPredictor,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_models
        )
        return ensemble()(x)


class Delegator(nn.Module):
    n_predictors: int
    hidden_width: int = 16

    @nn.compact
    def __call__(self, x: jax.Array):
        h = nn.Dense(self.hidden_width)(x)
        h = nn.relu(h)
        return nn.Dense(self.n_predictors)(h)


@struct.dataclass
class System:
    x: jax.Array
    y: jax.Array
    val_mask: jax.Array
    regions: jax.Array
    boundary_x: jax.Array
    boundary_y: jax.Array
    true_weights: jax.Array
    true_biases: jax.Array
    true_function_vectors: jax.Array
    function_cosine_distance: jax.Array
    n_predictors: int



def true_function(system: System, x: jax.Array):
    boundary_at_x = jnp.interp(x[:, 0], system.boundary_x, system.boundary_y)
    regions = (x[:, 1] >= boundary_at_x).astype(jnp.int32)
    weights = system.true_weights[regions]
    biases = system.true_biases[regions]
    y = jnp.sum(x * weights, axis=-1) + biases
    return y, regions


def display_disagreement_system(system: System, output_path: str | Path = "system_disagreement.png"):
    fig, ax = plt.subplots(figsize=(7, 6))

    # grid_axis = jnp.linspace(-1, 1, 300)
    # gx, gy = jnp.meshgrid(grid_axis, grid_axis)
    # grid = jnp.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1)
    # _, grid_regions = true_function(system, grid)
    disagreement = system.function_cosine_distance
    # signed_disagreement = jnp.where(grid_regions == 0, -disagreement, disagreement).reshape(gx.shape)

    # cmap = ListedColormap(["#2166ac", "#b2182b"])
    # norm = BoundaryNorm([-1.0, 0.0, 1.0], cmap.N)

    # ax.contourf(gx, gy, signed_disagreement, levels=[-1.0, 0.0, 1.0], cmap=cmap, norm=norm, alpha=0.55)
    ax.scatter(system.x[~system.val_mask, 0], system.x[~system.val_mask, 1], s=25)
    ax.scatter(system.x[system.val_mask, 0], system.x[system.val_mask, 1], marker="x", s=65)
    ax.plot(system.boundary_x, system.boundary_y, color="black", linewidth=2.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Two Linear Regions, Absolute Cosine Distance: {float(disagreement):.4f}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def display_true_surface_html(system: System, output_path: str | Path = "true_surface.html"):
    grid_axis = jnp.linspace(-1, 1, 180)
    gx, gy = jnp.meshgrid(grid_axis, grid_axis)
    grid = jnp.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1)
    z, regions = true_function(system, grid)
    z = z.reshape(gx.shape)
    regions = regions.reshape(gx.shape)
    boundary_grid = jnp.stack((system.boundary_x, system.boundary_y), axis=-1)
    boundary_z, _ = true_function(system, boundary_grid)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=gx,
            y=gy,
            z=z,
            surfacecolor=regions,
            colorscale=[[0.0, "#2166ac"], [0.499, "#2166ac"], [0.5, "#b2182b"], [1.0, "#b2182b"]],
            showscale=False,
            opacity=0.9
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=system.boundary_x,
            y=system.boundary_y,
            z=boundary_z,
            mode="lines",
            line=dict(color="black", width=8),
            name="boundary"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=system.x[:, 0],
            y=system.x[:, 1],
            z=system.y,
            mode="markers",
            marker=dict(size=3, color=system.y, colorscale="Viridis", opacity=0.75),
            name="samples"
        )
    )

    fig.update_layout(
        title="True Piecewise Linear Regression Surface",
        scene=dict(
            xaxis_title="x0",
            yaxis_title="x1",
            zaxis_title="y"
        ),
        width=900,
        height=750
    )

    fig.write_html(output_path, include_plotlyjs="cdn")


def display_system(system: System, output_dir: str | Path = "."):
    output_dir = Path(output_dir)
    display_disagreement_system(system, output_dir / "system_disagreement.png")
    display_true_surface_html(system, output_dir / "true_surface.html")

@struct.dataclass
class Predictions:
    predictions: jax.Array
    aggregated_y: jax.Array
    delegations: jax.Array
    aggregated_d: jax.Array

@struct.dataclass
class EpochSnapshot:
    predictions: Predictions
    loss: jax.Array

class MixtureEnsemble(nn.Module):
    n_predictors: int
    n_delegators: int
    delegator_hidden_width: int = 16
    agg: Literal["sum", "product"] = "sum"

    def setup(self):
        self.predictors = PredictorEnsemble(n_models=self.n_predictors)

        if self.n_delegators > 0:
            Delegators = nn.vmap(
                Delegator,
                variable_axes={'params': 0},
                split_rngs={'params': True},
                in_axes=None,
                axis_size=self.n_delegators,
                out_axes=1
            )
            self.delegators = Delegators(
                n_predictors=self.n_predictors,
                hidden_width=self.delegator_hidden_width
            )
        else:
            self.delegators = lambda x: jnp.zeros((x.shape[0], 1, self.n_predictors)) - jnp.log(self.n_predictors)

    def __call__(self, x: jax.Array):
        x_in = x.reshape((-1, x.shape[-1]))
        preds = self.predictors(x_in)

        if self.n_delegators > 0:
            del_logits = self.delegators(x_in)
            del_logprobs = jax.nn.log_softmax(del_logits, axis=-1)
            del_probs = jax.nn.softmax(del_logits, axis=-1)
        else:
            del_logprobs = self.delegators(x_in)
            del_logits = del_logprobs
            del_probs = jax.nn.softmax(del_logits, axis=-1)

        combiner_logprobs = del_logprobs.sum(axis=1)
        combiner_probs = del_probs.mean(axis=1)

        if self.agg == "product":
            weights = jax.nn.softmax(combiner_logprobs, axis=-1)
        elif self.agg == "sum":
            weights = combiner_probs / combiner_probs.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(self.agg)

        y_hat = jnp.sum(weights[:, :, jnp.newaxis] * preds, axis=-2)

        return Predictions(
            predictions=preds,
            aggregated_y=y_hat,
            delegations=del_probs,
            aggregated_d=weights
        )


def snapshot_to_dict(s: EpochSnapshot):
    return {
        "loss": s.loss,
        "predictions.predictions": s.predictions.predictions,
        "predictions.aggregated_y": s.predictions.aggregated_y,
        "predictions.delegations": s.predictions.delegations,
        "predictions.aggregated_d": s.predictions.aggregated_d,
    }


def make_problem(
    key: jax.Array,
    n_linear_predictors: int,
    n_boundary_segments: int,
    n_samples: int,
    x_range: tuple[float, float] = (-1, 1),
    boundary_spread: float = 0.35,
    val_fraction: float = 0.5,
    function_weight_scale: float = 1.0,
    function_bias_scale: float = 0.3
):
    if n_boundary_segments < 1:
        raise ValueError("n_boundary_segments must be at least 1")

    if n_linear_predictors < 2:
        raise ValueError("n_linear_predictors must be at least 2")

    k_x, k_boundary, k_val, k_weights, k_biases = jax.random.split(key, 5)

    x = jax.random.uniform(
        k_x,
        shape=(n_samples, 2),
        minval=x_range[0],
        maxval=x_range[1]
    )

    boundary_x = jnp.linspace(x_range[0], x_range[1], n_boundary_segments + 1)
    interior_count = n_boundary_segments - 1

    if interior_count > 0:
        interior_offsets = jax.random.uniform(
            k_boundary,
            shape=(interior_count,),
            minval=-boundary_spread,
            maxval=boundary_spread
        )
        offsets = jnp.concatenate((jnp.zeros((1,)), interior_offsets, jnp.zeros((1,))))
    else:
        offsets = jnp.zeros((2,))

    boundary_y = jnp.clip(boundary_x + offsets, x_range[0], x_range[1])
    boundary_at_x = jnp.interp(x[:, 0], boundary_x, boundary_y)
    regions = (x[:, 1] >= boundary_at_x).astype(jnp.int32)

    true_weights = jax.random.normal(k_weights, shape=(2, 2)) * function_weight_scale
    true_biases = jax.random.normal(k_biases, shape=(2,)) * function_bias_scale
    selected_weights = true_weights[regions]
    selected_biases = true_biases[regions]
    y = jnp.sum(x * selected_weights, axis=-1) + selected_biases

    true_function_vectors = jnp.concatenate((true_weights, true_biases[:, None]), axis=-1)
    v0 = true_function_vectors[0]
    v1 = true_function_vectors[1]
    cosine_similarity = jnp.sum(v0 * v1) / ((jnp.linalg.norm(v0) * jnp.linalg.norm(v1)) + 1e-8)
    function_cosine_distance = 1.0 - jnp.abs(cosine_similarity)

    val_mask = jax.random.bernoulli(k_val, p=val_fraction, shape=(n_samples,))

    return System(
        x=x,
        y=y[:, jnp.newaxis],
        regions=regions,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        true_weights=true_weights,
        true_biases=true_biases,
        true_function_vectors=true_function_vectors,
        function_cosine_distance=function_cosine_distance,
        n_predictors=n_linear_predictors,
        val_mask=val_mask.astype(bool)
    )


def one_sample_kl(combiner: jax.Array, individuals: jax.Array):
    kl_per_model = jax.scipy.special.kl_div(combiner[jnp.newaxis, :], individuals).sum(axis=-1)
    return jnp.mean(kl_per_model)


def calculate_kl(combiner_logprobs: jax.Array, delegator_logprobs: jax.Array):
    return jnp.mean(jax.vmap(one_sample_kl)(
        jax.nn.softmax(combiner_logprobs, axis=-1) + 1e-6,
        jax.nn.softmax(delegator_logprobs, axis=-1) + 1e-6
    ))

def mse_ensemble_loss(
    individual_predictions: jax.Array,
    y: jax.Array,
    weights: jax.Array
):
    
    y_expanded = y[:, jnp.newaxis, :]
    assert (
            y_expanded.ndim == individual_predictions.ndim
        ) and (
            y_expanded.shape[-1] ==  individual_predictions.shape[-1]
        ) and (
            y_expanded.shape[0] ==  individual_predictions.shape[0]
        )

    sq_error = (y_expanded - individual_predictions) ** 2
    weighted_loss_per_sample = jnp.sum(
        sq_error * weights[:, :, jnp.newaxis],
        axis=1,
    )
    return weighted_loss_per_sample

def mse_ensemble_eval(
    individual_predictions: jax.Array,
    y: jax.Array,
    weights: jax.Array
):
    
    assert (
            y.ndim == (individual_predictions.ndim - 1)
        ) and (
            y.shape[-1] == individual_predictions.shape[-1]
        ) and (
            y.shape[0] ==  individual_predictions.shape[0]
        )

    y_combiner = jnp.sum(individual_predictions * weights[:, :, jnp.newaxis], axis=-2)
    per_sample_error = jnp.mean((y - y_combiner) ** 2, axis=-1)
    return per_sample_error

def train_ensemble(
    system: System,
    key: jax.Array,
    n_seeds: int = 10,
    n_delegators: int = 5,
    n_chunks: int = 750,
    n_chunk_epochs: int = 50,
    lr: float = 1e-3,
    delegator_hidden_width: int = 16,
    agg: Literal["sum", "product"] = "sum",
):
    model = MixtureEnsemble(
        n_predictors=system.n_predictors,
        n_delegators=n_delegators,
        delegator_hidden_width=delegator_hidden_width,
        agg=agg
    )

    keys = jax.random.split(key, n_seeds)
    init_params = jax.vmap(lambda k: model.init(k, system.x)['params'])(keys)
    tx = optax.adamw(lr)
    create_fn = lambda p: train_state.TrainState.create(apply_fn=model.apply, params=p, tx=tx)
    state = jax.vmap(create_fn)(init_params)

    x_train, y_train = system.x[~system.val_mask], system.y[~system.val_mask]

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, None))
    def train_chunk(state, x: jax.Array, y: jax.Array):
        def step_fn(state, _):
            def loss_fn(params):
                predictions: Predictions = model.apply({'params': params}, x)
        
                loss = mse_ensemble_loss(predictions.predictions, y, predictions.aggregated_d)
                loss = jnp.mean(loss)
                
                mean_weights = jnp.mean(predictions.aggregated_d, axis=0)
                across_sample_gini = 1 - jnp.sum(mean_weights ** 2)

                return loss - 0.5 * across_sample_gini

            grads = jax.grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, None

        state, _ = jax.lax.scan(step_fn, state, None, length=n_chunk_epochs)

        predictions: Predictions = model.apply({"params": state.params}, system.x)
        eval_loss = mse_ensemble_eval(predictions.predictions, system.y, predictions.aggregated_d)
        snapshot = EpochSnapshot(
            predictions=predictions,
            loss=eval_loss
        )

        return state, snapshot

    snapshots = []

    for chunk in tqdm(range(n_chunks)):
        state, snapshot = train_chunk(
            state,
            x_train,
            y_train,
        )

        snapshots.append(snapshot)


    snapshots = jax.tree.map(lambda *xs: jnp.stack(xs), *snapshots)
 
    return snapshots


if __name__ == "__main__":
    base = Path(__file__).parent / "synthetic_runs"
    base.mkdir(exist_ok=True)

    train_seed = 123
    system_seed = 7878
    n_predictors = 2
    n_boundary_segments = 8
    n_samples = 300
    boundary_spread = 0.5
    function_weight_scale = 1.0
    function_bias_scale = 0.3
    delegator_hidden_width = 4
    only_display = False

    for agg in ("sum", "product"):
        folder = base / f"ts_{train_seed}_ss_{system_seed}_predictors_{n_predictors}_segments_{n_boundary_segments}_samples_{n_samples}_spread_{boundary_spread}_agg_{agg}"
        folder.mkdir(exist_ok=True)
        for ent in folder.iterdir():
            ent.unlink()

        k_train = jax.random.key(train_seed)

        system = make_problem(
            jax.random.key(system_seed),
            n_linear_predictors=n_predictors,
            n_boundary_segments=n_boundary_segments,
            n_samples=n_samples,
            boundary_spread=boundary_spread,
            function_weight_scale=function_weight_scale,
            function_bias_scale=function_bias_scale
        )

        if only_display:
            display_system(system, Path("."))
            exit(0)

        display_system(system, folder)

        delegators_specs = (1, 2, 4, 8, 16, 32, 64)
        specs = tuple(product(delegators_specs))
        finished = 0

        for (n_delegators,) in specs:
            jax.clear_caches()
            plt.close("all")

            snapshots: EpochSnapshot = train_ensemble(
                system,
                key=k_train,
                n_delegators=n_delegators,
                delegator_hidden_width=delegator_hidden_width,
                agg=agg
            )   

        
            jnp.savez(folder / f"delegators_{n_delegators}.npz", **snapshot_to_dict(snapshots))

            finished += 1
            print(f"{finished} / {len(specs)}")