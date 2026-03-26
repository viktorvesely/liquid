from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import optax
from tqdm import tqdm

from liquid_solver import LEsolver, LEInfo

import jax 
import jax.numpy as jnp
from flax import linen as nn


def get_solver():
    return LEsolver()

def get_layers(neurons: tuple[int, ...]):

    if not neurons:
        return None
    
    layers = []
    for n in neurons:
        layers.append(
            nn.Dense(n)
        )
    return tuple(layers)

def forward(h, last_linear, layers):
    if not layers: 
        return h
    
    for layer in layers[:-1]:
        h = nn.relu(layer(h))
    
    final_layer = layers[-1]
    h = final_layer(h)
    
    return h if last_linear else nn.relu(h)   

class LeModelMlp(nn.Module):
    
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None
    
    def setup(self):
        self.body_layers = get_layers(self.body)
        self.out_layers = get_layers(self.out)
        self.delegation_layers = get_layers(self.delegation)

    def __call__(self, x: jax.Array):
        h = x
        h_body = forward(h, last_linear=False, layers=self.body_layers)
        y = forward(h_body, last_linear=True, layers=self.out_layers)
        d = forward(h_body, last_linear=True, layers=self.delegation_layers)

        return y, d
    

class LeMlp(nn.Module):
    n_models: int
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None

    def setup(self):
        
        assert self.delegation[-1] == self.n_models, "Last dim of delegation needs to equal to n_models"

        VmappedLeModelMlp = nn.vmap(
            LeModelMlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.ensemble = VmappedLeModelMlp(
            out=self.out, 
            delegation=self.delegation, 
            body=self.body
        )

    def __call__(self, x: jax.Array, solver: LEsolver):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        ys, d = self.ensemble(x)
        d = nn.softmax(d, axis=-1)
        leinfo = solver.solve_power(d)
        y = solver.mix_power_mean(ys, leinfo.power)
        return y, leinfo

def strict_fourier_complexity(k: jax.Array, x: jax.Array, sigma: float=1.0, n_features: int=100):

    k1, k2 = jax.random.split(k)
    omega = jax.random.normal(k1, shape=(n_features, 1)) * sigma
    phi = jax.random.uniform(k2, minval=0, maxval=2 * jnp.pi, shape=(n_features, 1))
    
    # Compute Raw Fourier Sum
    raw_sum = jnp.sum(jnp.cos(omega @ x.reshape(1, -1) + phi), axis=0)
    
    # Scale by 1/sqrt(N) to keep the input to tanh in a sensitive range
    # then squash to exactly (-1, 1)
    return jnp.tanh(raw_sum / jnp.sqrt(n_features))

class FuncComposer:

    ybound: tuple[float, float] = (-1, 1)

    def __init__(self):
        self.regions = []
        self.x = None

    def add(self, name: Literal["linear", "fourier"], xbound: tuple[float, float], num_samples: int, sigma: float = 1.0):
        self.regions.append((name, xbound, num_samples, sigma))
        return self

    def compose(self, key: jax.Array) -> jax.Array:
        
        x = []
        y = []

        for name, (x1, x2), num_samples, sigma in self.regions:

            x_sub = jnp.linspace(x1, x2, num_samples, endpoint=False)
            x.append(x_sub)
            
            if name == "linear":
                y.append(None)
                continue

            key, k_use = jax.random.split(key)
            y_sub = strict_fourier_complexity(k_use,  x_sub, sigma)
            y.append(y_sub)

        x = jnp.concatenate(x)
        self.x = x
        self.point_sigmas = jnp.concatenate([
            jnp.full(num_samples, -1.0 if name == "linear" else sigma) 
            for name, _, num_samples, sigma in self.regions
        ])
        
        for i, (name, (x1, x2), _,  _) in enumerate(self.regions):
            
            if name == "fourier":
                continue

            x_sub = x[(x >= x1) & (x < x2)]

            key, k1, k2 = jax.random.split(key, 3)

            if (i == 0):
                y_start = jax.random.uniform(k1, minval=self .ybound[0], maxval=self.ybound[1])
            else:
                y_start = y[i - 1][-1]
            
            if ((i + 1) == len(self.regions)) or (y[i + 1] == None):
                y_end = jax.random.uniform(k2, minval=self .ybound[0], maxval=self.ybound[1])
            else:
                y_end = y[i + 1][0]

            y_sub = jnp.linspace(y_start, y_end, num=x_sub.size, endpoint=True)
            y[i] = y_sub
        
        return jnp.concatenate(y)





def auxillary_losses(
        train_return: LEInfo,
        solver: LEsolver,
    ) -> dict[str, jax.Array]:

        return {
            "load_distribution_loss": solver.load_distribution_loss(train_return),
            "specialization_losss": solver.specialization_loss(train_return)
        }


adam = optax.adam(5e-4)

@partial(jax.jit, static_argnames=("n_experts", "n_train_steps"))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def train_le(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    solver: LEsolver,
    n_experts: int,
    n_train_steps: int,
):
    
    print("Compiling")
    model = LeMlp(
        n_models=n_experts,
        body=(8,),
        delegation=(n_experts,),
        out=(1,)
    )
    params = model.init(key, x[[0], :], solver)["params"]
    opt_state = adam.init(params)

    def train_step(carry, _):
        params, opt_state = carry

        def loss_fn(params):
            yhat, leinfo = model.apply({"params": params}, x, solver)
            aux_loss = auxillary_losses(leinfo, solver)
            aux_values = jax.tree.reduce(lambda accum, aux_loss: accum + aux_loss, aux_loss, initializer=0.0)
            perf_loss = jnp.mean((yhat - y) ** 2)
            perf_metric = jnp.quantile(jnp.abs(yhat - y), 0.95)
            return perf_loss + aux_values, (perf_metric, perf_loss)
        
        (loss, (perf_metric, perf_loss)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = adam.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), (perf_metric, perf_loss)

    (params, opt_state), (perf_metric, perf_loss) = jax.lax.scan(train_step, (params, opt_state), length=n_train_steps)
    return perf_metric, params, perf_loss


@partial(jax.vmap, in_axes=(None, None, 0, None))
def eval_le(
    x: jax.Array,
    model: LeMlp,
    params: dict,
    solver: LEsolver
):
    
    yhat, leinfo = model.apply({"params": params}, x, solver)
    
    power: jax.Array = leinfo.power
    power = power / power.sum(-1, keepdims=True)
    power_entropy = jax.vmap(lambda p: jnp.sum(jax.scipy.special.entr(p)))(power)
    power_perplexity = jnp.exp(power_entropy)

    return yhat, power_perplexity
    
def run_gridsearch(key: jax.Array, n_models: int = 10, steps: int = 15_000):
    import numpy as np

    save_dir = Path(__file__).parent / "exp_composition"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    n_lambdas = 9
    lambdas = jnp.logspace(jnp.log10(1e-4), jnp.log10(1), n_lambdas, endpoint=True)
    lambdas = jnp.concatenate((jnp.array([0.0]), lambdas))
    n_lambdas += 1
    x_s, y_s, sigmas = composite_function()
    x, y = jnp.expand_dims(x_s, -1), jnp.expand_dims(y_s, -1)

    loss_grid = np.zeros((n_lambdas, n_lambdas))
    perf_grid = np.zeros((n_lambdas, n_lambdas))
    corr_grid = np.zeros((n_lambdas, n_lambdas))

    model = LeMlp(n_models=n_models, body=(8,), delegation=(n_models,), out=(1,))

    with tqdm(total=n_lambdas**2, desc="Lambdas") as pbar:
        for i, load_l in enumerate(lambdas):
            for j, spec_l in enumerate(lambdas):
                
        
                k_train = jax.random.split(key, 5) 
                solver = LEsolver(load_distribution_lambda=load_l, specialization_lambda=spec_l)
                perf_metric, params, perf_loss = train_le(k_train, x, y, solver, n_models, steps)
                
                loss_grid[i, j] = jnp.mean(perf_loss[:, -1])
                perf_grid[i, j] = jnp.mean(perf_metric[:, -1])
                
                _, perplexity = eval_le(x, model, params, solver)
                mean_perp = jnp.mean(perplexity, axis=0)
                corr_grid[i, j] = jnp.corrcoef(mean_perp, sigmas)[0, 1]
                
                pbar.update(1)

    np.savez(save_dir / "gridsearch.npz", loss_grid=loss_grid, corr_grid=corr_grid, perf_grid=perf_grid, lambdas=lambdas)


def powspace(start, stop, num, power=2):
    steps = jnp.linspace(0, 1, num)
    return start + (stop - start) * (steps ** power)


def train_le_on_composite(
    key: jax.Array,
    n_models: int,
    n_marginilize: int = 5,
    steps: int = 15_000
):
    save_dir = Path(__file__).parent / "exp_composition"
    save_dir.mkdir(exist_ok=True, parents=True)

        
    model = LeMlp(
        n_models=n_models,
        body=(8,),
        delegation=(n_models,),
        out=(1,)
    )

    solver = get_solver()

    x_s, y_s, _ = composite_function()
    x = jnp.expand_dims(x_s, -1)
    y = jnp.expand_dims(y_s, -1)

    key, k_train = jax.random.split(key)
    
    keys = jax.random.split(k_train, n_marginilize)
    perf_history, params, perf_loss = train_le(keys, x, y, n_models, steps)
    print(jnp.mean(perf_loss, axis=-1))
    
    yhat, perplexity = eval_le(x, model, params, solver)

    yhat = yhat[0, :]
    perplexity = jnp.mean(perplexity, axis=0)
    perplexitystd = jnp.std(perplexity, axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax2 = ax.twinx()

    l1, = ax.plot(x_s, y_s, color="black", label="truth")
    l2, = ax.plot(x_s, jnp.squeeze(yhat), color="black", linestyle="dashed", label="yhat")

    l3, = ax2.plot(x_s, perplexity, label="perplexity")
    ax2.fill_between(
        x_s,
        perplexity - perplexitystd,
        perplexity + perplexitystd,
        alpha=0.2
    )
    ax2.set_ylabel("Effective expert usage")

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    fig.savefig(save_dir / "expert_usage.png", dpi=500)

def train_experts_on_composite(
    key: jax.Array,
    n_marginilize: int = 4,
    steps: int = 15_000
):
    save_dir = Path(__file__).parent / "exp_composition"
    save_dir.mkdir(exist_ok=True, parents=True)

    x, y, _ = composite_function()
    x = jnp.expand_dims(x, -1)
    y = jnp.expand_dims(y, -1)

    experts = jnp.array([1, 2, 3, 5, 10, 20], dtype=jnp.int32)
    key, k_loop = jax.random.split(key)

    fig, ax = plt.subplots(figsize=(10, 6))

    xsteps = jnp.arange(steps)
    skip_mask = xsteps > 1_500

    solver = get_solver()

    for n_exp in experts.tolist():
        keys = jax.random.split(k_loop, n_marginilize)
        perf_history, _, _ = train_le(keys, x, y, solver, n_exp, steps)
        mean_traj = jnp.mean(perf_history, axis=0)
        std_traj = jnp.std(perf_history, axis=0)
        ax.plot(xsteps[skip_mask], mean_traj[skip_mask], label=f"Experts: {n_exp}")
        # ax.fill_between(
        #     xsteps[skip_mask], 
        #     mean_traj[skip_mask] - std_traj[skip_mask], 
        #     mean_traj[skip_mask] + std_traj[skip_mask], 
        #     alpha=0.2
        # )
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_yscale("log")
    ax.legend()
    
    import matplotlib.ticker as ticker
    ymin, ymax = ax.get_ylim()
    ticks = jnp.logspace(jnp.log10(ymin), jnp.log10(ymax), num=10)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='y', which='both', labelsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / f"composite.png", dpi=500)

    plt.close(fig)

def composite_function(plot=False):
    key = jax.random.key(123)
    n_linear_samples = 10
    composer = FuncComposer()
    y = composer.add(
        "linear", (-1, -0.6), n_linear_samples
    ).add(
        "fourier", (-0.6, 0), sigma=15.0, num_samples=40
    ).add(
        "linear", (0, 0.3), n_linear_samples
    ).add(
        "fourier", (0.3, 0.8), sigma=5, num_samples=20
    ).add(
        "linear", (0.8, 1), n_linear_samples
    ).compose(key)

    if plot:
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(composer.x, y, marker="o", color="black")
        fig.savefig("./composite.png")

    return composer.x, y, composer.point_sigmas

def plot_gridsearch():
    import numpy as np

    save_dir = Path(__file__).parent / "exp_composition"
    data = np.load(save_dir / "gridsearch.npz")
    loss_grid, corr_grid, perf_grid, lambdas = data["loss_grid"], data["corr_grid"], data["perf_grid"], data["lambdas"]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    grids = [(loss_grid, "Final Performance Loss"), (corr_grid, "Perplexity-Sigma Correlation"), (perf_grid, "Final 0.95 quantile MAE")]
    labels = [f"{l:.1e}" for l in lambdas]

    for ax, (grid, title) in zip(axes, grids):
        cax = ax.matshow(grid, cmap="plasma")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title(title, pad=20)
        ax.set_xticks(range(len(lambdas)))
        ax.set_yticks(range(len(lambdas)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Specialization Lambda")
        ax.set_ylabel("Load Distribution Lambda")

        threshold = (np.max(grid) + np.min(grid)) / 2.0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = "black" if grid[i, j] > threshold else "white"
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "gridsearch_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    # composite_function(plot=True)    
    #train_experts_on_composite(jax.random.key(8778))
    # train_le_on_composite(
    #     jax.random.key(123),
    #     n_models=10,
    # )
    # run_gridsearch(jax.random.key(123))
    plot_gridsearch()