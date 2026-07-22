from dataclasses import dataclass
from functools import partial
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct
import tqdm

from structs import TrainParams, InOutData, Predictors, Delegators
from math_utils import optimal_convex_weights, eval_predictor_delegator_decomposition, aggregate_delegators, verify_weights_improvement, eval_loss, predictor_error_ambiguity_decomposition, delegator_error_ambiguity_decomposition
from utils import train_loader
from atomic_networks import three_layer_mlp
from architectures import Ensemble, get_modules

USE_THREE_LAYER_DELEGATOR = True
THREE_LAYER_DELEGATOR_BASE = 16

@struct.dataclass
class InOutDataOracle:
    x: jax.Array
    y: jax.Array
    predictions: jax.Array

@dataclass(frozen=True)
class JittedFunctions:
    train_oracle_batch: Callable
    validate_and_update_best: Callable
    apply_delegators_agg: Callable
    apply_oracle_delegators_agg: Callable
    apply_predictors: Callable
    oracle_optimizer: optax.TransformUpdateExtraArgsFn
    oracle_delegators: nn.Module

def jit_functions(
    predictors: nn.Module,
    delegators: nn.Module,
    train_params: TrainParams,
) -> JittedFunctions:


    oracle_optimizer = optax.adamw(learning_rate=train_params.lr * 4)
    _apply_delegators_agg = jax.jit(partial(
        apply_delegators_agg,
        delegators=delegators,
        train_params=train_params
    ))

    if USE_THREE_LAYER_DELEGATOR:

        assert train_params.architecture.cnn == 0, "Implement some powerful oracle for CNNs"

        architecture = three_layer_mlp.determine_size(
            predictor_base=1,
            delegator_base=THREE_LAYER_DELEGATOR_BASE,
            out_dim=train_params.task.out_dim(),
            n_predictors=train_params.n_predictors
        )
        ensemble = Ensemble(
            n_predictors=train_params.n_predictors,
            n_delegators=train_params.n_delegators,
            predictor=architecture.predictor,
            delegator=architecture.delegator,
            n_cnn_layers=architecture.cnn   
        ) 
        _, oracle_delegators = get_modules(ensemble)
        
        _apply_oracle_delegators_agg = jax.jit(partial(
            apply_delegators_agg,
            delegators=oracle_delegators,
            train_params=train_params
        ))

    else:
        oracle_delegators = delegators
        _apply_oracle_delegators_agg = _apply_delegators_agg

    return JittedFunctions(
        train_oracle_batch=jax.jit(partial(
            train_oracle_batch,
            optimizer=oracle_optimizer,
            delegators=oracle_delegators,
            train_params=train_params
        )),
        validate_and_update_best=jax.jit(partial(
            validate_and_update_best,
            delegators=oracle_delegators,
            train_params=train_params
        )),
        apply_delegators_agg=_apply_delegators_agg,
        apply_oracle_delegators_agg=_apply_oracle_delegators_agg,
        apply_predictors=jax.jit(partial(
            apply_predictors,
            predictors=predictors
        )),
        oracle_optimizer=oracle_optimizer,
        oracle_delegators=oracle_delegators
    )

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

def train_oracle_batch(
    inout_data: InOutDataOracle,
    delegator_params: dict,
    opt_states: dict,
    optimizer: optax.GradientTransformationExtraArgs,
    delegators: nn.Module,
    train_params: TrainParams
):
    # print(f" Compiling {train_oracle_batch.__name__}")
    
    batch_size = train_params.batch_size
    assert inout_data.x.shape[0] % train_params.batch_size == 0, (
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


def validate_and_update_best(
    params,
    current_best_params,
    current_best_losses,
    current_best_epoch,
    valid_data: InOutDataOracle,
    epoch: jax.Array,
    delegators: nn.Module, 
    train_params: TrainParams
):
    # print(f" Compiling {validate_and_update_best.__name__}")
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

def train_oracle(
    key: jax.Array,
    train_predictions: jax.Array,
    valid_predictions: jax.Array,
    agg_delegations: jax.Array,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    jit_funcs: JittedFunctions,
    epochs_p: float = 0.25,
    n_seeds: int = 5,
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

    optimizer = jit_funcs.oracle_optimizer
    delegators = jit_funcs.oracle_delegators

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

    epochs = max(1, round(train_params.epochs * epochs_p))

    for i_epoch in tqdm.tqdm(
        range(epochs),
        desc="Training oracle",
        position=1,
        disable=True
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

            delegator_params, opt_states = jit_funcs.train_oracle_batch(
                inout_data=inout_batch,
                delegator_params=delegator_params,
                opt_states=opt_states,
            )

        (
            best_delegator_params,
            best_valid_losses,
            best_delegator_epoch,
            valid_losses,
        ) = jit_funcs.validate_and_update_best(
            params=delegator_params,
            current_best_params=best_delegator_params,
            current_best_losses=best_valid_losses,
            current_best_epoch=best_delegator_epoch,
            valid_data=inout_valid_delegations,
            epoch=jnp.array(i_epoch)
        )

    best_seed = jnp.argmin(best_valid_losses)

    return jax.tree.map(
        lambda x: x[best_seed],
        best_delegator_params,
    ), best_delegator_epoch[best_seed]


def apply_delegators_agg(
    delegator_params: dict,
    x: jax.Array,
    delegators: nn.Module,
    train_params: TrainParams
):
    # print(f" Compiling {apply_delegators_agg.__name__}")
    delegations = delegators.apply({"params": delegator_params}, x)
    agg_delegations = aggregate_delegators(train_params, delegations)
    return agg_delegations, delegations

def apply_predictors(
    predictor_params: dict,
    x: jax.Array,
    predictors: nn.Module,
):
    # print(f" Compiling {apply_predictors.__name__}")
    predictions = predictors.apply({"params": predictor_params}, x)
    return predictions

def get_evaluation_metrics(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors: Predictors,
    predictors_params: dict,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
):
     
    n_seeds = jax.tree.leaves(delegators_params)[0].shape[0] 
    jit_funcs = jit_functions(
        predictors=predictors,
        delegators=delegators,
        train_params=train_params
    )

    all_metrics = []

    for use_seed in tqdm.tqdm(range(n_seeds), position=0, desc="Seeds", disable=True):
        metrics = one_get_evaluation_metrics(
            key=key,
            delegators=delegators,
            delegators_params=delegators_params,
            predictors=predictors,
            predictors_params=predictors_params,
            inout_train_predictions=inout_train_predictions,
            inout_valid_predictions=inout_valid_predictions,
            train_params=train_params,
            jit_funcs=jit_funcs,
            use_seed=use_seed
        )
        all_metrics.append(metrics)

    all_metrics = jax.tree.map(lambda *values: jnp.stack(values), *all_metrics)
    
    # for k, v in all_metrics.items():
    #     print(k, v.shape)

    return all_metrics

def one_get_evaluation_metrics(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors: Predictors,
    predictors_params: dict,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    jit_funcs: JittedFunctions,
    use_seed: int
):
    gpu = jax.devices("gpu")[0]

    assert (inout_train_predictions.y.ndim == inout_valid_predictions.y.ndim) and inout_train_predictions.y.ndim <= 2, f"{inout_train_predictions.y.shape}, {inout_valid_predictions.y.shape}" 
    key, k_loader, k_train_oracle = jax.random.split(key, 3)
    
    # Select seeds
    selected_delegator_params = jax.tree.map(lambda x: x[use_seed, ...], delegators_params)
    selected_predictor_params = jax.tree.map(lambda x: x[use_seed, ...], predictors_params)

    # Get predictions and current delegations for all data
    gpu_batch_size = train_params.batch_size * train_params.preload_batches_to_gpu

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

            batch_predictions = jit_funcs.apply_predictors(selected_predictor_params, inout_batch.x)
            train_predictions.append(batch_predictions)
            
            batch_agg_delegations, _ = jit_funcs.apply_delegators_agg(selected_delegator_params, inout_batch.x)
            agg_delegations.append(batch_agg_delegations)
    

    # Validation
    valid_predictions = jit_funcs.apply_predictors(selected_predictor_params, inout_valid_predictions.x)
    valid_agg_delegations, valid_delegations = jit_funcs.apply_delegators_agg(selected_delegator_params, inout_valid_predictions.x)
    agg_delegations.append(valid_agg_delegations)

    train_predictions = jnp.concatenate(train_predictions, axis=0)
    agg_delegations = jnp.concatenate(agg_delegations, axis=0)

    # Train oracle
    oracle_delegator_params, from_epoch = train_oracle(
        key=k_train_oracle,
        train_predictions=train_predictions,
        valid_predictions=valid_predictions,
        agg_delegations=agg_delegations,
        inout_train_predictions=inout_train_predictions,
        inout_valid_predictions=inout_valid_predictions,
        train_params=train_params,
        jit_funcs=jit_funcs
    )

    valid_oracle_agg_delegations, valid_oracle_delegations = jit_funcs.apply_oracle_delegators_agg(oracle_delegator_params, inout_valid_predictions.x)
    
    (predictor_loss, delegator_regret_loss), (loss, loss_under_oracle) = eval_predictor_delegator_decomposition(
        predictions=valid_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params,
        use="loss"
    )
    
    metrics = dict(
        predictor_loss=predictor_loss,
        delegator_regret_loss=delegator_regret_loss,
        loss=loss,
        loss_under_oracle=loss_under_oracle,
        from_epoch=from_epoch
    )

    (predictor_metric, delegator_regret_metric), (metric, metric_under_oracle) = eval_predictor_delegator_decomposition(
        predictions=valid_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params,
        use="metric"
    )

    metrics |= dict(
        predictor_metric=predictor_metric,
        delegator_regret_metric=delegator_regret_metric,
        metric=metric,
        metric_under_oracle=metric_under_oracle
    )


    print(f"From epoch = {from_epoch}")
    print(f"Metric = {metric:.3f} Metric oracle {metric_under_oracle:.3f}")

    # Error ambiguity decompositions

    # For predictors use the truly optimal weights 
    oracle_better = loss_under_oracle <= loss
    valid_superior_agg_delegations = valid_oracle_agg_delegations if oracle_better else valid_agg_delegations
    predictors_perfomance_per_model, predictors_ambiguity_per_model = predictor_error_ambiguity_decomposition(
        predictions=valid_predictions,
        y=inout_valid_predictions.y,
        task_type=train_params.task.task_type(),
        agg_delegation=valid_superior_agg_delegations
    )

    metrics |= dict(
        predictors_perfomance_per_model=predictors_perfomance_per_model,
        predictors_ambiguity_per_model=predictors_ambiguity_per_model,
        predictors_weight_per_model=valid_superior_agg_delegations
    )

    # For delegators use the oracle weights
    delegators_perfomance_per_model, delegators_ambiguity_per_model = delegator_error_ambiguity_decomposition(
        delegations=valid_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        train_params=train_params
    )

    metrics |= dict(
        delegators_perfomance_per_model=delegators_perfomance_per_model,
        delegators_ambiguity_per_model=delegators_ambiguity_per_model
    )


    metrics = {k: jnp.array(v) for k, v in metrics.items()}

    return metrics
    
