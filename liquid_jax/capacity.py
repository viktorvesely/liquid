from typing import Literal

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from pathlib import Path

from tqdm import tqdm

DE_MODE = True

EXP_NAME = "deep_ensembles"
N_DELEGATORS_LIST = (1, 2, 4, 8, 16, 32)
N_SEEDS = 10

AGG: Literal["sum", "product"] = "sum" # ""

PROB_SEED = 123
TRAIN_SEED = 0

W_PRED = 8 * (2 if DE_MODE else 1)
W_DEL = 16
N_PRED = 10

T = 50_000
LR = 1e-3

N_SAMPLES = 40
VFRAC = 0.2
SIGMA = 8.0

class Predictor(nn.Module):
    width: int
    out_dim: int = 1
    
    @nn.compact
    def __call__(self, x: jax.Array):
        h = nn.Dense(self.width)(x)
        h = nn.relu(h)
        return nn.Dense(self.out_dim)(h)

class PredictorEnsemble(nn.Module):
    n_models: int
    width: int
    out_dim: int = 1
    
    @nn.compact
    def __call__(self, x: jax.Array):
        ensemble = nn.vmap(
            Predictor,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_models
        )
        return ensemble(width=self.width, out_dim=self.out_dim)(x) 

class Delegator(nn.Module):
    n_predictors: int
    width: int

    @nn.compact
    def __call__(self, x: jax.Array):
        h = nn.Dense(self.width)(x)
        h = nn.relu(h)
        h = nn.Dense(self.width)(h)
        h = nn.relu(h)
        return nn.Dense(self.n_predictors)(h)

class MixtureEnsemble(nn.Module):
    n_predictors: int
    n_delegators: int
    w_pred: int
    w_del: int
    out_dim: int = 1

    def setup(self):
        self.predictors = PredictorEnsemble(
            n_models=self.n_predictors, width=self.w_pred, out_dim=self.out_dim
        )
        
        if self.n_delegators > 0:
            Delegators = nn.vmap(
                Delegator,
                variable_axes={'params': 0},
                split_rngs={'params': True},
                in_axes=None,
                axis_size=self.n_delegators,
                out_axes=1
            )
            self.delegators = Delegators(n_predictors=self.n_predictors, width=self.w_del)
        else:
            self.delegators = lambda x: jnp.zeros((x.shape[0], 1, self.n_predictors)) - jnp.log(self.n_predictors)

    def __call__(self, x: jax.Array):
        x_in = x.reshape((x.shape[0], -1))
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

        if AGG == "product":
            weights = jax.nn.softmax(combiner_logprobs, axis=-1)
        elif AGG == "sum":
            weights = combiner_probs / combiner_probs.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(AGG)
        
        weights_exp = jnp.expand_dims(weights, axis=-1)
        y_hat = jnp.sum(weights_exp * preds, axis=1)
        
        return y_hat

def generate_data(n, p_val, sigma, seed):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = jnp.linspace(-1.0, 1.0, n)
    norm = jax.random.normal(k1, (100,)) * sigma
    unif = jax.random.uniform(k2, (100,), maxval=2*jnp.pi)
    y = jnp.sum(jnp.cos(jnp.outer(x, norm) + unif), axis=1)
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    idx = jax.random.permutation(k3, n)
    nv = int(p_val * n)
    iv, it = idx[:nv], idx[nv:]
    
    return x[it], y[it], x[iv], y[iv]

def get_vmapped_train_fn(n_del):

    if DE_MODE:
        model = MixtureEnsemble(n_predictors=n_del, n_delegators=0, w_pred=W_PRED, w_del=W_DEL)
    else:
        model = MixtureEnsemble(n_predictors=N_PRED, n_delegators=n_del, w_pred=W_PRED, w_del=W_DEL)
    
    def train_single_seed(seed_key, xtr, ytr, xva, yva):
        params = model.init(seed_key, xtr)
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
        opt_state = tx.init(params)
        
        def step(carry, _):
            p, opt_st = carry
            
            def loss_fn(weights):
                preds = model.apply(weights, xtr)
                return jnp.mean((preds - ytr) ** 2)
            
            loss_tr, grads = jax.value_and_grad(loss_fn)(p)
            updates, opt_st = tx.update(grads, opt_st, p)
            p = optax.apply_updates(p, updates)
            
            preds_va = model.apply(p, xva)
            loss_va = jnp.mean((preds_va - yva) ** 2)
            
            return (p, opt_st), (loss_tr, loss_va)
            
        _, (losses_tr, losses_va) = jax.lax.scan(step, (params, opt_state), None, length=T)
        return losses_tr, losses_va

    return jax.jit(jax.vmap(train_single_seed, in_axes=(0, None, None, None, None)))

def run_experiment():
    save_dir = Path(__file__).parent / "synthetic_runs" / EXP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    
    xtr, ytr, xva, yva = generate_data(N_SAMPLES, VFRAC, SIGMA, PROB_SEED)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(xtr, ytr, label="Train", alpha=0.7)
    plt.scatter(xva, yva, label="Validation", alpha=0.7)
    plt.legend()
    plt.savefig(save_dir / "data_scatter.png")
    plt.close()
    
    seed_keys = jax.random.split(jax.random.PRNGKey(TRAIN_SEED), N_SEEDS)
    
    for n_del in tqdm(N_DELEGATORS_LIST):
        train_fn = get_vmapped_train_fn(n_del)
        train_mse, val_mse = train_fn(seed_keys, xtr, ytr, xva, yva)
        
        jnp.save(save_dir / f"ndel_{n_del}_train_mse.npy", train_mse)
        jnp.save(save_dir / f"ndel_{n_del}_val_mse.npy", val_mse)

if __name__ == "__main__":
    run_experiment()