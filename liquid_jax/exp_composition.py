from functools import partial

import jax
import jax.numpy as jnp
import optax

from learner_mnist_le import LeMlp
from liquid_solver import LEsolver, LEInfo

solver = LEsolver()

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

def train_le_with_increasing_experts(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    s: int = 1,
    e: int = 10,
    n_ensembles: int = 5,
    n_train_steps: int = 10_000
):
    
    adam = optax.adamw(5e-4)
    keys = jax.random.split(key, n_ensembles)
    
    @partial(jax.jit, static_argnames="n_experts")
    @partial(jax.vmap, in_axes=(0, None))
    def train_le(
        key: jax.Array,
        n_experts: int
    ):

        model = LeMlp(
            n_models=n_experts,
            body=(5,),
            delegation=(5, n_experts),
            out=(5, 1)
        )
        params = model.init(key, x[[0], :])["params"]
        opt_state = adam.init(params)

        def train_step(carry, _):
            params, opt_state = carry

            def loss_fn(params):
                yhat, leinfo = forward(x, model, params)
                aux_loss = auxillary_losses(leinfo)
                aux_values = jax.tree.reduce(lambda accum, aux_loss: accum + aux_loss, aux_loss, initializer=0.0)
                return jnp.mean((yhat - y) ** 2) + aux_values
            
            loss, grad = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = adam.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            return (params, opt_state), loss

        _, loss = jax.lax.scan(train_step, (params, opt_state), length=n_train_steps)
        return loss[-1]


    losses = []
    for n_experts in range(s, e + 1):
        
        l = train_le(keys, n_experts)
        l = l.mean().item()
        losses.append(l)

    return losses

def powspace(start, stop, num, power=2):
    steps = jnp.linspace(0, 1, num)
    return start + (stop - start) * (steps ** power)

def show_regions():


    key = jax.random.key(123)
    
    k_data, k_loop = jax.random.split(key)

    x = jnp.linspace(-3, 3, 100)[:, jnp.newaxis]
    
    s = 1
    e = 10
    experts = jnp.arange(s, e)

    for sigma in powspace(1, 10, num=5):
        y_sigma = strict_fourier_complexity(k_data, x, sigma=sigma)
        losses = train_le_with_increasing_experts(k_loop, x, y_sigma, s=s, e=e, n_ensembles=5)
        print(sigma)
        print(losses)
        losses = jnp.array(losses)
        best = jnp.argmin(losses)
        print(experts[best].item(), losses[best].item())

if __name__ == "__main__":
    show_regions()