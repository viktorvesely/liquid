from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import optax

from learner_mnist_le import LeMlp
from liquid_solver import LEsolver, LEInfo

solver = LEsolver(
    load_distribution_lambda=0.00
)

def strict_fourier_complexity(k: jax.Array, x: jax.Array, sigma: float=1.0, n_features: int=100):

    k1, k2 = jax.random.split(k)
    omega = jax.random.normal(k1, shape=(n_features, 1)) * sigma
    phi = jax.random.uniform(k2, minval=0, maxval=2 * jnp.pi, shape=(n_features, 1))
    
    # Compute Raw Fourier Sum
    raw_sum = jnp.sum(jnp.cos(omega @ x.reshape(1, -1) + phi), axis=0)
    
    # Scale by 1/sqrt(N) to keep the input to tanh in a sensitive range
    # then squash to exactly (-1, 1)
    return jnp.tanh(raw_sum / jnp.sqrt(n_features))


def forward(x: jax.Array, model: LeMlp, params: dict) -> tuple[jax.Array, LEInfo]:
        
    ys, delegation = model.apply({"params": params}, x)
    leinfo = solver.solve_power(delegation)
    y = solver.mix_power_mean(ys, leinfo.power)

    return y, leinfo

def auxillary_losses(
        train_return: LEInfo
    ) -> dict[str, jax.Array]:
        
        return {
            "load_distribution_loss": solver.load_distribution_loss(train_return),
            "specialization_losss": solver.specialization_loss(train_return)
        }


adam = optax.adamw(5e-5)

@partial(jax.jit, static_argnames=("n_experts", "n_train_steps"))
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def train_le(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    n_experts: int,
    n_train_steps: int
):

    model = LeMlp(
        n_models=n_experts,
        body=(64,),
        delegation=(64, n_experts),
        out=(64, 1)
    )
    params = model.init(key, x[[0], :])["params"]
    opt_state = adam.init(params)

    def train_step(carry, _):
        params, opt_state = carry

        def loss_fn(params):
            yhat, leinfo = forward(x, model, params)
            aux_loss = auxillary_losses(leinfo)
            aux_values = jax.tree.reduce(lambda accum, aux_loss: accum + aux_loss, aux_loss, initializer=0.0)
            print(yhat.shape, y.shape)
            perf_loss = jnp.mean((yhat - y) ** 2)
            perf_metric = jnp.mean(jnp.abs(yhat - y))
            return perf_loss + aux_values, perf_metric
        
        (loss, perf_metric), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = adam.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), perf_metric

    _, perf_metric = jax.lax.scan(train_step, (params, opt_state), length=n_train_steps)
    return perf_metric


def powspace(start, stop, num, power=2):
    steps = jnp.linspace(0, 1, num)
    return start + (stop - start) * (steps ** power)

def show_regions():
    save_dir = Path(__file__).parent / "exp_composition"
    save_dir.mkdir(exist_ok=True, parents=True)

    key = jax.random.key(123)
    k_data, k_loop = jax.random.split(key)
    x = jnp.linspace(-3, 3, 300)[:, jnp.newaxis]
    experts = jnp.arange(1, 18, 3, dtype=jnp.int32) 

    for sigma in powspace(1, 10, num=5):
        y_sigma = strict_fourier_complexity(k_data, x, sigma=sigma)[:, jnp.newaxis]
        keys = jax.random.split(k_loop, 3)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        
        for n_exp in experts.tolist():
            perf_history = train_le(keys, x, y_sigma, n_exp, 10_000)
            mean_traj = jnp.mean(perf_history, axis=0)
            ax.plot(mean_traj, label=f"Experts: {n_exp}")
        
        ax.set_xlabel("Steps")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(save_dir / f"sigma_{sigma:.3f}.png")
        plt.close(fig)

if __name__ == "__main__":
    show_regions()