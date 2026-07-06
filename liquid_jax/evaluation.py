from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
import tqdm

from structs import TrainParams, InOutData, Ensemble, Predictors, Delegators
from math_utils import optimal_convex_weights, loss_predictor_delegator_decomposition, aggregate_delegators
from utils import train_loader

@partial(
    jax.jit,
    static_argnames=("optimizer", "delegators", "batch_size"),
)
def train_oracle_batch(
    inout_data: InOutData,
    delegator_params,
    opt_states,
    optimizer: optax.GradientTransformationExtraArgs,
    delegators: nn.Module,
    batch_size: int,
):
    assert inout_data.x.shape[0] % batch_size == 0, (
        "GPU batch needs to be divisible by batch_size"
    )

    n_batches = inout_data.x.shape[0] // batch_size

    inout_data = jax.tree.map(
        lambda x: x.reshape(
            (n_batches, batch_size) + x.shape[1:]
        ),
        inout_data,
    )

    def oracle_loss_fn(
        one_seed_params,
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

    loss_and_grad_fn = jax.value_and_grad(oracle_loss_fn)

    def update_one_seed(
        one_seed_params,
        one_seed_opt_state,
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

    def train_step(carry, inout_batch: InOutData):
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

    (delegator_params, opt_states), _ = jax.lax.scan(
        train_step,
        (delegator_params, opt_states),
        inout_data,
    )

    return delegator_params, opt_states


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

    # Assumes predictions has no seed axis:
    # [n_train + n_valid, n_predictors, out_dim].
    print("Finding unrestricted optimal weights")
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

    inout_train_delegations = InOutData(
        x=inout_train_predictions.x,
        y=optimal_weights[:n_train],
    )

    inout_valid_delegations = InOutData(
        x=inout_valid_predictions.x,
        y=optimal_weights[n_train:],
    )

    # Keep the full training set on CPU. Chunks are moved to GPU below.
    inout_train_delegations = jax.tree.map(
        np.asarray,
        inout_train_delegations,
    )

    # Validation data is reused every epoch, so keep it on GPU.
    inout_valid_delegations = jax.tree.map(
        lambda x: jax.device_put(x, device=gpu),
        inout_valid_delegations,
    )

    optimizer = optax.adamw(learning_rate=train_params.lr)

    def init_one_seed(init_key: jax.Array):
        params = delegators.init(
            init_key,
            inout_train_delegations.x[[0], ...],
        )["params"]

        opt_state = optimizer.init(params)

        return params, opt_state

    init_keys = jax.random.split(k_init, n_seeds)

    delegator_params, opt_states = jax.vmap(
        init_one_seed,
    )(init_keys)

    best_delegator_params = delegator_params
    best_valid_losses = jnp.full(
        (n_seeds,),
        jnp.inf,
    )

    def oracle_loss_fn(
        one_seed_params,
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

    @jax.jit
    def validate_and_update_best(
        params,
        current_best_params,
        current_best_losses,
        valid_data: InOutData,
    ):
        valid_losses = jax.vmap(
            oracle_loss_fn,
            in_axes=(0, None),
        )(
            params,
            valid_data,
        )

        improved = valid_losses < current_best_losses

        current_best_params = jax.tree.map(
            lambda new, old: jnp.where(
                improved.reshape(
                    (improved.shape[0],)
                    + (1,) * (new.ndim - 1)
                ),
                new,
                old,
            ),
            params,
            current_best_params,
        )

        current_best_losses = jnp.minimum(
            current_best_losses,
            valid_losses,
        )

        return (
            current_best_params,
            current_best_losses,
            valid_losses,
        )

    batch_size = train_params.batch_size

    # Number of ordinary minibatches loaded onto GPU together.
    gpu_batch_size = (
        batch_size
        * train_params.preload_batches_to_gpu
    )

    n_train_examples = inout_train_delegations.x.shape[0]


    n_gpu_batches = math.ceil(
        n_train_examples / gpu_batch_size
    )

    epochs = int(
        math.round(train_params.epochs * epochs_p)
    )

    for _ in tqdm.tqdm(
        range(epochs),
        desc="Training oracle",
    ):
        for inout_batch, k_loader in train_loader(
            k_loader,
            inout_train_delegations,
            batch_size=gpu_batch_size,
            desired_batches=n_gpu_batches,
        ):
    
            inout_batch = jax.tree.map(
                lambda x: jax.device_put(x, device=gpu),
                inout_batch,
            )

            delegator_params, opt_states = train_oracle_batch(
                inout_data=inout_batch,
                delegator_params=delegator_params,
                opt_states=opt_states,
                optimizer=optimizer,
                delegators=delegators,
                batch_size=batch_size,
            )

        (
            best_delegator_params,
            best_valid_losses,
            valid_losses,
        ) = validate_and_update_best(
            delegator_params,
            best_delegator_params,
            best_valid_losses,
            inout_valid_delegations,
        )

    best_seed = jnp.argmin(best_valid_losses)

    return jax.tree.map(
        lambda x: x[best_seed],
        best_delegator_params,
    )

def get_evaluation_metrics(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors: Predictors,
    predictors_params: dict,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    use_seed: int = 0
):
    gpu = jax.devices("gpu")[0]

    assert (len(inout_train_predictions.y.shape) - 1) == len(inout_valid_predictions.y.shape), "Inout train predictions needs to be prepared for batching" 
    key, k_loader, k_train_oracle = jax.random.split(key, 3)
    
    # Select seeds
    selected_delegator_params = jax.tree.map(lambda x: x[use_seed, ...], delegators_params)
    selected_predictor_params = jax.tree.map(lambda x: x[use_seed, ...], predictors_params)

    # Get predictions and current delegations for all data
    n_train_examples = inout_train_predictions.x.shape[0]
    gpu_batch_size = train_params.batch_size * train_params.preload_batches_to_gpu
    n_gpu_batches = math.ceil(n_train_examples / gpu_batch_size)

    print("Aggregating final outputs")
    predictions = []

    for inout_batch, k_loader in train_loader( # TODO fix shuffling and also maybe for the oracle
            k_loader,
            inout_train_predictions,
            batch_size=gpu_batch_size,
            desired_batches=n_gpu_batches,
        ):
            inout_batch: InOutData = jax.tree.map(
                lambda x: jax.device_put(x, device=gpu),
                inout_batch,
            )

            batch_predictions = predictors.apply({"params": selected_predictor_params}, inout_batch.x)

            predictions.append(batch_predictions)

    # Validation
    validation_predictions = predictors.apply({"params": selected_predictor_params}, inout_valid_predictions.x)
    validation_delegations = delegators.apply({"params": selected_delegator_params}, inout_valid_predictions.x)

    predictions.append(validation_predictions)
    predictions = jnp.concatenate(predictions, axis=0)

    # Train oracle
    oracle_delegator_params = train_oracle(k_train_oracle, delegators, predictions, inout_train_predictions, inout_valid_predictions, train_params)
    validation_oracle_delegations = delegators.apply({"params": oracle_delegator_params}, inout_valid_predictions.x)

    valid_agg_delegations = aggregate_delegators(train_params, validation_delegations)
    valid_oracle_agg_delegations = aggregate_delegators(train_params, validation_oracle_delegations)
    
    (predictor_loss, delegator_regret), (loss, loss_under_oracle) = loss_predictor_delegator_decomposition(
        predictions=validation_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params
    )

    print(
        f"predictor_loss={jnp.mean(predictor_loss):.3f}",
        f"delegator_regret={jnp.mean(delegator_regret):.3f}",
        f"loss={jnp.mean(loss):.3f}",
        f"loss_under_oracle={jnp.mean(loss_under_oracle):.3f}",
    )
    
