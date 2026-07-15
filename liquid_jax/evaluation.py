from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct
import tqdm

from structs import TrainParams, InOutData, Ensemble, Predictors, Delegators
from math_utils import optimal_convex_weights, eval_predictor_delegator_decomposition, aggregate_delegators, verify_weights_improvement, eval_loss
from utils import train_loader


@struct.dataclass
class InOutDataOracle:
    x: jax.Array
    y: jax.Array
    predictions: jax.Array


def oracle_loss_fn(
        one_seed_params,
        inout_batch: InOutDataOracle,
        delegators: Delegators,
        train_params: TrainParams
    ):
        delegations_logits = delegators.apply(
            {"params": one_seed_params},
            inout_batch.x,
        )

        agg_delegations = aggregate_delegators(train_params, delegations_logits)
        
        losses = eval_loss(
            weights=agg_delegations,
            predictions=inout_batch.predictions,
            y=inout_batch.y,
            train_params=train_params
        )

        return jnp.mean(losses)

@partial(
    jax.jit,
    static_argnames=("optimizer", "delegators", "batch_size", "train_params"),
)
def train_oracle_batch(
    inout_data: InOutData,
    delegator_params,
    opt_states,
    optimizer: optax.GradientTransformationExtraArgs,
    delegators: nn.Module,
    batch_size: int,
    train_params: TrainParams
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


    loss_and_grad_fn = jax.value_and_grad(oracle_loss_fn)

    def update_one_seed(
        one_seed_params,
        one_seed_opt_state,
        inout_batch: InOutData,
    ):
        _, grads = loss_and_grad_fn(
            one_seed_params,
            inout_batch,
            delegators,
            train_params,
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
    train_predictions: jax.Array,
    valid_predictions: jax.Array,
    agg_delegations: jax.Array,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    epochs_p: float = 1,
    n_seeds: int = 8,
):
    gpu = jax.devices("gpu")[0]

    k_init, k_loader = jax.random.split(key)


    # print("Finding unrestricted optimal weights")
    # all_y = jnp.concatenate(
    #     (
    #         inout_train_predictions.y,
    #         inout_valid_predictions.y,
    #     ),
    #     axis=0,
    # )
    # optimal_weights = optimal_convex_weights(
    #     y=all_y,
    #     predictions=predictions,
    #     weights0=agg_delegations,
    #     train_params=train_params,
    # )

    # verify_weights_improvement(
    #     y=all_y,
    #     predictions=predictions,
    #     oracle_weights=optimal_weights,
    #     weights=agg_delegations,
    #     train_params=train_params,
    #     verbal=True
    # )
    # n_train = inout_train_predictions.y.shape[0]

    inout_train_delegations = InOutDataOracle(
        x=inout_train_predictions.x,
        y=inout_train_predictions.y,
        predictions=train_predictions
    )

    inout_valid_delegations = InOutDataOracle(
        x=inout_valid_predictions.x,
        y=inout_valid_predictions.y,
        predictions=valid_predictions
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
    best_valid_losses = jnp.full((n_seeds,), jnp.inf,)
    best_delegator_epoch = jnp.full((n_seeds,), -1)

    @jax.jit
    def validate_and_update_best(
        params,
        current_best_params,
        current_best_losses,
        current_best_epoch,
        valid_data: InOutDataOracle,
        epoch: jax.Array
    ):
        valid_losses = jax.vmap(
            oracle_loss_fn,
            in_axes=(0, None, None, None),
        )(
            params,
            valid_data,
            delegators,
            train_params
        )

        epochs = jnp.broadcast_to(epoch, current_best_epoch.shape)

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

        current_best_epoch = jax.tree.map(
            lambda new, old: jnp.where(improved, new, old),
            epochs,
            current_best_epoch
        )

        current_best_losses = jax.tree.map(
            lambda new, old: jnp.where(improved, new, old),
            valid_losses,
            current_best_losses
        )

        return (
            current_best_params,
            current_best_losses,
            current_best_epoch,
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
        round(train_params.epochs * epochs_p)
    )

    for i_epoch in tqdm.tqdm(
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
                train_params=train_params
            )

        (
            best_delegator_params,
            best_valid_losses,
            best_delegator_epoch,
            valid_losses,
        ) = validate_and_update_best(
            delegator_params,
            best_delegator_params,
            best_valid_losses,
            best_delegator_epoch,
            inout_valid_delegations,
            jnp.array(i_epoch)
        )

    best_seed = jnp.argmin(best_valid_losses)
    

    return jax.tree.map(
        lambda x: x[best_seed],
        best_delegator_params,
    ), best_delegator_epoch[best_seed]

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

    assert (inout_train_predictions.y.ndim == inout_valid_predictions.y.ndim) and inout_train_predictions.y.ndim < 2, f"{inout_train_predictions.y.shape}, {inout_valid_predictions.y.shape}" 
    key, k_loader, k_train_oracle = jax.random.split(key, 3)
    
    # Select seeds
    selected_delegator_params = jax.tree.map(lambda x: x[use_seed, ...], delegators_params)
    selected_predictor_params = jax.tree.map(lambda x: x[use_seed, ...], predictors_params)

    # Get predictions and current delegations for all data
    gpu_batch_size = train_params.batch_size * train_params.preload_batches_to_gpu

    print("Aggregating final outputs")
    train_predictions = []
    agg_delegations = []

    for inout_batch, k_loader in train_loader(
            k_loader,
            inout_train_predictions,
            batch_size=gpu_batch_size,
            serve_as_is=True
        ):
            inout_batch: InOutData = jax.tree.map(
                lambda x: jax.device_put(x, device=gpu),
                inout_batch,
            )

            batch_predictions = predictors.apply({"params": selected_predictor_params}, inout_batch.x)
            train_predictions.append(batch_predictions)
            
            batch_delegations = delegators.apply({"params": selected_delegator_params}, inout_batch.x)
            batch_agg_delegations = aggregate_delegators(train_params, batch_delegations)
            agg_delegations.append(batch_agg_delegations)
    
    train_predictions = jnp.concatenate(train_predictions, axis=0)

    # Validation
    validation_predictions = predictors.apply({"params": selected_predictor_params}, inout_valid_predictions.x)
    validation_delegations = delegators.apply({"params": selected_delegator_params}, inout_valid_predictions.x)
    valid_agg_delegations = aggregate_delegators(train_params, validation_delegations)


    agg_delegations.append(valid_agg_delegations)
    agg_delegations = jnp.concatenate(agg_delegations, axis=0)

    # Train oracle
    oracle_delegator_params, from_epoch = train_oracle(
        key=k_train_oracle,
        delegators=delegators,
        train_predictions=train_predictions,
        valid_predictions=validation_predictions,
        agg_delegations=agg_delegations,
        inout_train_predictions=inout_train_predictions,
        inout_valid_predictions=inout_valid_predictions,
        train_params=train_params
    )
    validation_oracle_delegations = delegators.apply({"params": oracle_delegator_params}, inout_valid_predictions.x)
    valid_oracle_agg_delegations = aggregate_delegators(train_params, validation_oracle_delegations)
    
    (predictor_loss, delegator_regret), (loss, loss_under_oracle) = eval_predictor_delegator_decomposition(
        predictions=validation_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params,
        use="loss"
    )

    print(
        f"best_oracle_from_epoch={jnp.int32(from_epoch)}",
        f"predictor_loss={jnp.mean(predictor_loss):.3f}",
        f"delegator_regret={jnp.mean(delegator_regret):.3f}",
        f"loss={jnp.mean(loss):.3f}",
        f"loss_under_oracle={jnp.mean(loss_under_oracle):.3f}",
    )


    (predictor_loss, delegator_regret), (loss, loss_under_oracle) = eval_predictor_delegator_decomposition(
        predictions=validation_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params,
        use="metric"
    )

    print(
        f"best_oracle_from_epoch={jnp.int32(from_epoch)}",
        f"predictor_metric={jnp.mean(predictor_loss):.3f}",
        f"delegator_regret={jnp.mean(delegator_regret):.3f}",
        f"metric={jnp.mean(loss):.3f}",
        f"metric_under_oracle={jnp.mean(loss_under_oracle):.3f}",
    )
    
