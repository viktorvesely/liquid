from __future__ import annotations
from dataclasses import asdict
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.core import unfreeze, freeze
from typing import Literal
import matplotlib.pyplot as plt
from tqdm import tqdm

class LinearPredictor(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        weight = self.param('weight', jax.nn.initializers.normal(), tuple())
        bias = self.param('bias', jax.nn.initializers.normal(), tuple())
        return weight * x + bias


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

    @nn.compact
    def __call__(self, x: jax.Array):

        fw = 16
        sw = 8

        if activation_function == "cos":
            h = nn.Dense(fw)(x)
            h = jnp.cos(h)
            h = nn.Dense(sw)(h)
            h = jnp.cos(h)
        elif activation_function == "relu":
            h = nn.Dense(fw)(h)
            h = nn.relu(h)
            h = nn.Dense(sw)(h)
            h = nn.relu(h)
        else:
            raise ValueError(activation_function)

        return nn.Dense(self.n_predictors)(h)

def display_system(system: System):

    from matplotlib import pyplot as plt

    n_groups = int(system.regions.max()) + 1
    fig, ax = plt.subplots()

    for i_group in range(n_groups):
        mask = system.regions == i_group
        ax.scatter(system.x[mask], system.y[mask], label=str(i_group))


    ax.scatter(system.x[system.val_mask], system.y[system.val_mask], color="black", marker="x")
    fig.savefig("system.png")


class MixtureEnsemble(nn.Module):
    n_predictors: int
    n_delegators: int
    preset_predictors: PredictorEnsemble | None = None

    def setup(self):

        if self.preset_predictors is None:
            self.predictors = PredictorEnsemble(n_models=self.n_predictors)
        else:
            self.predictors = self.preset_predictors.clone(name="predictors")
        
        if self.n_delegators > 0:
            Delegators = nn.vmap(
                Delegator,
                variable_axes={'params': 0},
                split_rngs={'params': True},
                in_axes=None,
                axis_size=self.n_delegators,
                out_axes=1
            )
            self.delegators = Delegators(n_predictors=self.n_predictors)
        else:
            self.delegators = lambda x: jnp.zeros((x.shape[0], 1, self.n_predictors)) - jnp.log(self.n_predictors)

    def __call__(self, x: jax.Array):
        x_in = x.reshape((-1, 1))
        
        preds = self.predictors(x)
        
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

        if agg == "product":
            weights = jax.nn.softmax(combiner_logprobs, axis=-1)
        elif agg == "sum":
            weights = combiner_probs / combiner_probs.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(agg)
        
        y_hat = jnp.sum(weights * preds, axis=-1)
        
        return y_hat, weights, preds, del_logprobs, combiner_logprobs


@struct.dataclass
class System:
    x: jax.Array
    y: jax.Array
    val_mask: jax.Array
    regions: jax.Array
    predictors: nn.Module
    params: dict

@struct.dataclass
class Predictions:
    x: jax.Array
    joint_y: jax.Array
    predictors: jax.Array
    effective_predictors: jax.Array

def make_problem(
    key: jax.Array,
    n_linear_predictors: int,
    n_regions: int,
    n_samples: int,
    n_validation_regions: int = 1,
    x_range: tuple[float, float] = (-1, 1),
    x_gap: float = 0.3
):
    k_init, k_regions = jax.random.split(key)
    k_stops, k_ids, k_val = jax.random.split(k_regions, 3)

    region_counts = jnp.full((n_regions,), fill_value=(n_samples // n_regions))
    region_counts = region_counts.at[-1].set(n_samples - region_counts[:-1].sum())
    
    region_indices = jnp.repeat(jnp.arange(n_regions), region_counts)
    x = jnp.linspace(x_range[0], x_range[1], num=n_samples) + (region_indices * x_gap)
    x = x - jnp.mean(x)

    predictors = PredictorEnsemble(n_models=n_linear_predictors)
    params = predictors.init(k_init, x)
    y_predictors = predictors.apply(params, x)
    
    region_ids = jnp.arange(n_regions) % n_linear_predictors
    regions = jnp.repeat(region_ids, region_counts)
    
    y = jnp.take_along_axis(y_predictors, regions[:, None], axis=1).squeeze(-1)

    inds = jnp.where(jnp.diff(regions) != 0)[0] + 1
    inds = jnp.concatenate((jnp.array([0]), inds, jnp.array([regions.size])))
    val_region = (inds.size // 2) - 9
    mask = jnp.zeros((regions.size,)).astype(jnp.int32)
    mask = mask.at[inds[val_region]:inds[val_region + 1]].set(1)

    # mask = jax.random.bernoulli(k_val, p=0.3, shape=(n_samples,))

    return System(
        x=x,
        y=y,
        regions=regions,
        predictors=predictors,
        params=params,
        val_mask=mask.astype(jnp.bool)
    )

def one_sample_kl(combiner: jax.Array, individuals: jax.Array):
    kl_per_model = jax.scipy.special.kl_div(combiner[jnp.newaxis, :], individuals).sum(axis=-1)
    return jnp.mean(kl_per_model)


def calculate_kl(combiner_logprobs: jax.Array, delegator_logprobs: jax.Array):
    return jnp.mean(jax.vmap(one_sample_kl)(
        jax.nn.softmax(combiner_logprobs, axis=-1) + 1e-6,
        jax.nn.softmax(delegator_logprobs, axis=-1) + 1e-6
    ))


def train_ensemble(
    system: System, 
    key: jax.Array,
    n_seeds: int = 10,
    n_delegators: int = 5,   # how does this interact with the validation loss here
    preset_predictors: PredictorEnsemble | None = None,
    preset_predictors_params: dict | None = None, 
    n_chunks: int = 300,
    n_chunk_epochs: int = 200,
    lr: float = 1e-3,
    kl_weight: float =  0.0000,
    output_fig: bool = False,
    display_seed: int = 0
):
    model = MixtureEnsemble(n_predictors=system.predictors.n_models, n_delegators=n_delegators, preset_predictors=preset_predictors)
    
    keys = jax.random.split(key, n_seeds)
    init_params = jax.vmap(lambda k: model.init(k, system.x)['params'])(keys)
    
    freeze_predictors = (preset_predictors is not None) and (preset_predictors_params is not None)

    if freeze_predictors:
        
        # 1. Inject the pretrained parameters
        init_params = unfreeze(init_params)
        init_params["predictors"] = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_seeds,) + x.shape), 
            preset_predictors_params["params"]
        )
        init_params = freeze(init_params)

        # 2. Create partition labels matching the param tree structure
        labels = jax.tree.map(lambda _: "train", init_params)
        labels = unfreeze(labels)
        labels["predictors"] = jax.tree.map(lambda _: "freeze", labels["predictors"])
        labels = freeze(labels)

        # 3. Route 'train' to AdamW, and 'freeze' to zero updates
        tx = optax.multi_transform(
            {"train": optax.adamw(lr), "freeze": optax.set_to_zero()},
            labels
        )
    else:
        tx = optax.adamw(lr)

    create_fn = lambda p: train_state.TrainState.create(apply_fn=model.apply, params=p, tx=tx)
    state = jax.vmap(create_fn)(init_params)
    
    x_train, y_train = system.x[~system.val_mask], system.y[~system.val_mask]
    x_val, y_val = system.x[system.val_mask], system.y[system.val_mask]

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, None, None, None))
    def train_chunk(state, x, y, x_val, y_val):
        def step_fn(state, _):
            def loss_fn(params):
                y_hat, weights, preds, d_lp, c_lp = model.apply({'params': params}, x)
                loss = jnp.mean((y_hat - y)**2)
                var_preds = jnp.mean(jnp.var(preds, axis=-1))
                kl = calculate_kl(c_lp, d_lp)
                return loss - kl_weight * kl, (loss, var_preds, kl)
            
            (_, (loss, variance, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            
            y_hat, _, _, _, _ = model.apply({'params': state.params}, x_val)
            val_loss = jnp.mean((y_hat - y_val)**2)

            return state, (loss, val_loss, variance, kl)
            
        
            
        return jax.lax.scan(step_fn, state, None, length=n_chunk_epochs)

    @jax.jit
    def eval_step(params, x_val, y_val):
        y_hat, _, _, _, _ = model.apply({'params': params}, x_val)
        return jnp.mean((y_hat - y_val)**2)

    hist_tr_loss, hist_v_loss, hist_var, hist_kl = [], [], [], []

    for chunk in tqdm(range(n_chunks)):
        state, (tr_loss_arr, val_loss_arr, var_preds_arr, kl_arr) = train_chunk(state, x_train, y_train, x_val, y_val)

        tr_loss_arr = jnp.mean(tr_loss_arr, axis=0)
        val_loss_arr = jnp.mean(val_loss_arr, axis=0)
        var_preds_arr = jnp.mean(var_preds_arr, axis=0)
        kl_arr = jnp.mean(kl_arr, axis=0)
        
        hist_tr_loss.append(tr_loss_arr)
        hist_v_loss.append(val_loss_arr)
        hist_var.append(var_preds_arr)
        hist_kl.append(kl_arr)

    hist_tr_loss = jnp.concatenate(hist_tr_loss, axis=0)
    hist_v_loss = jnp.concatenate(hist_v_loss, axis=0)
    hist_var = jnp.concatenate(hist_var, axis=0)
    hist_kl = jnp.concatenate(hist_kl, axis=0)

    final_y_hat, final_weights, linear_preds, _, _ = jax.vmap(
        lambda p: model.apply({'params': p}, system.x)
    )(state.params)
    
    def calc_soft(weights: jax.Array):
        return jnp.exp(-jnp.sum(weights * jnp.log(weights + 1e-8), axis=-1))

    soft_models = jax.vmap(calc_soft)(final_weights)


    if output_fig:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        axs[0, 0].plot(hist_tr_loss, label="Train Loss")
        axs[0, 0].plot(hist_v_loss, label="Val Loss")
        axs[0, 0].set_yscale('log')
        axs[0, 0].legend()
        
        axs[0, 1].plot(hist_var, color='purple')
        axs[0, 1].set_title("Predictor Variance (Diversity)")
        
        axs[1, 0].plot(hist_kl, color='green')
        axs[1, 0].set_title("KL Divergence (Combiner vs Delegators)")
        
        ax_fin = axs[1, 1]
        ax_fin.scatter(x_train, y_train, color='gray', alpha=0.5, label='Train')
        ax_fin.scatter(x_val, y_val, color='blue', marker='*', s=150, edgecolor='black', label='Val')
        ax_fin.plot(system.x, final_y_hat[display_seed], color='black', linewidth=2.5, label='Ensemble Pred')
        
        for i_linear in range(linear_preds.shape[2]):
            ax_fin.plot(system.x, linear_preds[display_seed, :, i_linear], label=f"lm-{i_linear}")

        ax_fin.legend()
        
        ax_twin = ax_fin.twinx()
      
        ax_twin.plot(system.x, soft_models[display_seed], color='red', linestyle='--', alpha=0.7, label='Soft Models')
        ax_twin.set_ylabel("Effective Active Models", color='red')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        plt.savefig("ensemble_training.png")


    return (
        hist_tr_loss,
        hist_v_loss,
        hist_var,
        hist_kl,
    ), Predictions(
        x=system.x,
        joint_y=final_y_hat,
        predictors=linear_preds,
        effective_predictors=soft_models
    )


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).parent / "synthetic_runs"
    base.mkdir(exist_ok=True)

    train_seed = 123
    system_seed = 129321093809
    n_predictors = 2
    n_regions = 29
    n_samples = 150
    activation_function = "cos" # "cos" "relu"
    agg: Literal["sum", "product"] = "sum" # ""

    folder = base / f"ts_{train_seed}_ss_{system_seed}_predictors_{n_predictors}_regions_{n_regions}_samples_{n_samples}_activation_{activation_function}_agg_{agg}"
    folder.mkdir(exist_ok=True)
    for ent in folder.iterdir(): ent.unlink()

    k_train = jax.random.key(train_seed)

    system = make_problem(
        jax.random.key(system_seed),
        n_linear_predictors=n_predictors,
        n_regions=n_regions,
        n_samples=n_samples
    )

    display_system(system)
    
    
    
    freeze_specs = (True, False)
    delegators_specs = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16)

    specs = tuple(product(freeze_specs, delegators_specs))
    finished = 0
    for frozen_predictors, n_delegators in specs:
        jax.clear_caches()
        plt.close("all")


        
        (train_loss, val_loss, var, kl), predictions = train_ensemble(
            system,
            key=k_train,
            n_delegators=n_delegators,
            preset_predictors_params=system.params if frozen_predictors else None,
            preset_predictors=system.predictors if frozen_predictors else None
        )

        predictions_data = {f"{k}_predictions": v for k, v in asdict(predictions).items()}
        data = dict(
            train_loss=train_loss,
            val_loss=val_loss,
            predictor_variance=var,
            delegator_kl=kl,
        )

        jnp.savez(
            folder / f"frozen_{frozen_predictors}_delegators_{n_delegators}.npz",
            **(data | predictions_data)
        )

        finished += 1
        print(f"{finished} / {len(specs)}")

    