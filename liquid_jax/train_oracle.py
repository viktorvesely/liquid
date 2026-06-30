import math

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from structs import TrainParams, InOutData
from math_utils import optimal_convex_weights


def train_oracle(
    key: jax.Array,
    delegators: nn.Module,
    predictions: jax.Array,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    epochs_p: float = 0.1,
    n_seeds: int = 8,
):
    gpu = jax.devices("gpu")[0]
    k_init, k_loader = jax.random.split(key)

    optimal_weights = optimal_convex_weights(
        y=jnp.concatenate(
            (
                inout_train_predictions.y,
                inout_valid_predictions.y,
            ),
            axis=0,
        ),
        predictions=predictions,
        train_params=train_params,
    )

    n_train = inout_train_predictions.y.shape[0]

    optimal_w_train = optimal_weights[:n_train]
    optimal_w_valid = optimal_weights[n_train:]

    inout_train_delegations = InOutData(
        x=jax.device_put(inout_train_predictions.x, device=gpu),
        y=jax.device_put(optimal_w_train, device=gpu),
    )

    inout_valid_delegations = InOutData(
        x=jax.device_put(inout_valid_predictions.x, device=gpu),
        y=jax.device_put(optimal_w_valid, device=gpu),
    )

    optimizer = optax.adamw(learning_rate=1e-3)

    def init(init_key: jax.Array):
        params = delegators.init(
            init_key,
            inout_train_delegations.x[[0], ...],
        )["params"]
        opt_state = optimizer.init(params)
        return params, opt_state

    init_keys = jax.random.split(k_init, n_seeds)
    delegator_params, opt_states = jax.vmap(init)(init_keys)

    best_delegator_params = delegator_params
    best_valid_losses = jnp.full((n_seeds,), jnp.inf)

    batch_size = train_params.batch_size
    n_train_examples = inout_train_delegations.x.shape[0]

    assert n_train_examples % batch_size == 0, (
        "GPU batch needs to be divisible by batch_size"
    )

    n_batches = n_train_examples // batch_size

    inout_train_delegations = jax.tree.map(
        lambda x: x.reshape(
            (n_batches, batch_size) + x.shape[1:]
        ),
        inout_train_delegations,
    )

    def loss_fn(
        one_seed_params: dict,
        inout_batch: InOutData,
    ):
        delegation_logits = delegators.apply(
            {"params": one_seed_params},
            inout_batch.x,
        )

        losses = jax.vmap(
            optax.safe_softmax_cross_entropy,
            in_axes=(1, None),
            out_axes=1,
        )(
            delegation_logits,
            inout_batch.y,
        )

        return jnp.mean(losses)

    loss_and_grad_fn = jax.value_and_grad(loss_fn)

    def update_one_seed(
        one_seed_params: dict,
        one_seed_opt_state: optax.OptState,
        inout_batch: InOutData,
    ):
        _, grads = loss_and_grad_fn(
            one_seed_params,
            inout_batch,
        )

        updates, one_seed_opt_state = optimizer.update(
            grads,
            one_seed_opt_state,
            one_seed_params,
        )

        one_seed_params = optax.apply_updates(
            one_seed_params,
            updates,
        )

        return one_seed_params, one_seed_opt_state

    def batch_step(carry, inout_batch):
        params, states = carry

        params, states = jax.vmap(
            update_one_seed,
            in_axes=(0, 0, None),
        )(
            params,
            states,
            inout_batch,
        )

        return (params, states), None

    def select_params(
        improved: jax.Array,
        new_params,
        old_params,
    ):
        return jax.tree.map(
            lambda new, old: jnp.where(
                improved.reshape(
                    (improved.shape[0],)
                    + (1,) * (new.ndim - 1)
                ),
                new,
                old,
            ),
            new_params,
            old_params,
        )

    def epoch_step(carry, _):
        (
            params,
            states,
            best_params,
            best_losses,
            loader_key,
        ) = carry

        loader_key, permutation_key = jax.random.split(loader_key)

        batch_permutation = jax.random.permutation(
            permutation_key,
            n_batches,
        )

        shuffled_train_delegations = jax.tree.map(
            lambda x: x[batch_permutation],
            inout_train_delegations,
        )

        (params, states), _ = jax.lax.scan(
            batch_step,
            (params, states),
            shuffled_train_delegations,
        )

        valid_losses = jax.vmap(
            loss_fn,
            in_axes=(0, None),
        )(
            params,
            inout_valid_delegations,
        )

        improved = valid_losses < best_losses

        best_params = select_params(
            improved,
            params,
            best_params,
        )

        best_losses = jnp.minimum(
            best_losses,
            valid_losses,
        )

        return (
            params,
            states,
            best_params,
            best_losses,
            loader_key,
        ), None

    epochs = int(
        math.round(train_params.epochs * epochs_p)
    )

    (
        _,
        _,
        best_delegator_params,
        best_valid_losses,
        _,
    ), _ = jax.lax.scan(
        epoch_step,
        (
            delegator_params,
            opt_states,
            best_delegator_params,
            best_valid_losses,
            k_loader,
        ),
        xs=None,
        length=epochs,
    )

    best_seed = jnp.argmin(best_valid_losses)

    return jax.tree.map(
        lambda x: x[best_seed],
        best_delegator_params,
    )