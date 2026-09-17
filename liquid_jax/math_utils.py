from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from structs import ForwardArgs, ForwardReturn, TrainParams, Ensemble


@partial(jax.jit, static_argnames=("train_params", "steps"))
def optimal_convex_weights(
    y: jax.Array,
    predictions: jax.Array,
    train_params: TrainParams,
    weights0: jax.Array,
    steps: int = 500
):

    optimizer = optax.adam(learning_rate=1e-3)
    def init(w):
        return optimizer.init(w)

    weights = weights0
    opt_state = init(weights)
    state = (weights, opt_state)

    def loss_fn(weights: jax.Array):
        loss = eval_loss(
            weights,
            predictions,
            y,
            train_params
        )
        return jnp.mean(loss)

    def step(state):

        weight, opt_state = state

        grads = jax.grad(loss_fn)(weight)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            weight
        )
        weight = optax.apply_updates(weight, updates)
        weight = jax.vmap(
            optax.projections.projection_simplex
        )(weight)

        return weight, opt_state

    def optimize(state, _):
        state = step(state)
        return state, None

    state, _ = jax.lax.scan(
        optimize,
        state,
        length=steps
    )
    weights, _ = state

    return weights

def verify_weights_improvement(
    y: jax.Array,
    predictions: jax.Array,
    oracle_weights: jax.Array,
    weights: jax.Array,
    train_params: TrainParams,
    verbal: bool = False,
    atol: float = 1e-4
):

    normal_loss = eval_loss(
        weights,
        predictions,
        y,
        train_params
    )
    normal_loss = jnp.mean(normal_loss)

    oracle_loss = eval_loss(
        oracle_weights,
        predictions,
        y,
        train_params
    )
    oracle_loss = jnp.mean(oracle_loss)
    
    sign_difference = oracle_loss - normal_loss
    almost_equal = sign_difference <= atol
    improved = oracle_loss < normal_loss
    
    assert improved or almost_equal, f"Loss with oracle weights is larger than without by {sign_difference}"

    if not verbal:
        return
    
    if improved:
        print("Oracle weights are better")
    elif almost_equal:
        print(f"Oracle weights are equal with difference = {sign_difference}")


def eval_predictor_delegator_decomposition(
    predictions: jax.Array,
    agg_delegations: jax.Array,
    agg_oracle_delegations: jax.Array,
    y: jax.Array,
    train_params: TrainParams,
    use: Literal["loss", "metric"] = "loss",
):
    
    eval_fn = eval_loss if use == "loss" else eval_metric
    
    metric = eval_fn(agg_delegations, predictions, y, train_params)
    metric_under_oracle = eval_fn(agg_oracle_delegations, predictions, y, train_params)

    if use == "loss":
        metric = jnp.mean(metric)
        metric_under_oracle = jnp.mean(metric_under_oracle)

    delegators_regret = metric - metric_under_oracle
    predictor_error = metric_under_oracle

    return (predictor_error, delegators_regret), (metric, metric_under_oracle)

     
def delegator_error_ambiguity_decomposition(
    delegations: jax.Array,  # (BS, delegators, predictors), logits
    predictions: jax.Array,  # (BS, predictors, out)
    y: jax.Array,
    train_params: TrainParams,
    stop_delegations_ambiguity_gradient: bool = True
):
    predictions = jax.lax.stop_gradient(predictions)
    
    task_type = train_params.task.task_type()
    weights = jax.nn.softmax(delegations, axis=-1)

    predictor_performance, predictor_ambiguity = jax.vmap(
        predictor_error_ambiguity_decomposition,
        in_axes=(None, None, None, 1),
        out_axes=1,
    )(
        predictions,
        y,
        task_type,
        weights,
    )

    weighted_performance = jnp.sum(
        weights * predictor_performance, axis=-1
    )
    weighted_ambiguity = jnp.sum(
        weights * predictor_ambiguity, axis=-1
    )

    if train_params.ambiguity_gradient_predictors == "none":
        weighted_ambiguity = jax.lax.stop_gradient(weighted_ambiguity)

    performance = weighted_performance - weighted_ambiguity

    mix = (
        mix_weighted_mean
        if task_type == "regression"
        else mix_weighted_logits
    )

    individual_predictions = jax.vmap(
        mix, in_axes=(None, 1), out_axes=1
    )(predictions, weights)

    agg_weights = aggregate_delegators(train_params, delegations)
    final_prediction = mix(predictions, agg_weights)

    if task_type == "regression":
        ambiguity = jax.vmap(
            var_ambiguity, in_axes=(None, 1), out_axes=1
        )(final_prediction, individual_predictions)

    elif task_type == "classification":
        ambiguity = jax.vmap(
            kl_ambiguity,
            in_axes=(None, 1),
            out_axes=1,
        )(
            final_prediction,
            individual_predictions,
        )

    if stop_delegations_ambiguity_gradient:
        ambiguity = jax.lax.stop_gradient(ambiguity)

    return performance, ambiguity  # Both (BS, delegators)


def predictor_error_ambiguity_decomposition(
    predictions: jax.Array,
    y: jax.Array,
    task_type: Literal["classification", "regression"],
    agg_delegation: jax.Array,
    return_agg_prediction: bool = False,
    stop_predictions_ambiguity_gradient: bool = False
):

    predictions_optional_grad = jax.lax.stop_gradient(predictions) if stop_predictions_ambiguity_gradient else predictions
        
    if task_type == "classification":
        agg_prediction = mix_weighted_logits(predictions_optional_grad, agg_delegation) 
    elif task_type == "regression":
        agg_prediction = mix_weighted_mean(predictions_optional_grad, agg_delegation)

    if task_type == "classification":
        perfomance_loss_per_model = ce_loss(predictions, y)
        ambiguity_per_model = jax.vmap(kl_ambiguity, in_axes=(None, 1), out_axes=1)(
            agg_prediction, 
            predictions_optional_grad
        )
    elif task_type == "regression":
        perfomance_loss_per_model = mse_loss(predictions, y)
        ambiguity_per_model = jax.vmap(var_ambiguity, in_axes=(None, 1), out_axes=1)(
            agg_prediction, 
            predictions_optional_grad
        )

    if return_agg_prediction:
        return (perfomance_loss_per_model, ambiguity_per_model), agg_prediction
    else:
        return perfomance_loss_per_model, ambiguity_per_model
    



def eval_loss(
    weights: jax.Array, # (BS, n_predictors)
    predictions: jax.Array, # (BS, n_predictors, out)
    y: jax.Array, # (BS, out)
    train_params: TrainParams
):
    
    task_type = train_params.task.task_type()
    
    if task_type == "classification":
        agg_prediction = mix_weighted_logits(predictions, weights)
        loss = optax.softmax_cross_entropy_with_integer_labels(agg_prediction, y)
    elif task_type == "regression":
        agg_prediction = mix_weighted_mean(predictions, weights)
        assert agg_prediction.shape == y.shape
        loss = jnp.mean((agg_prediction - y) ** 2, axis=-1)

    return loss

def eval_metric(
    weights: jax.Array, # (BS, n_predictors)
    predictions: jax.Array, # (BS, n_predictors, out)
    y: jax.Array, # (BS, out)
    train_params: TrainParams
):
    
    task_type = train_params.task.task_type()
    
    if task_type == "classification":
        agg_prediction = mix_weighted_logits(predictions, weights)
        metric = classification_accuracy(agg_prediction, y)
    elif task_type == "regression":
        agg_prediction = mix_weighted_mean(predictions, weights)
        assert agg_prediction.shape == y.shape
        metric = regression_r2(agg_prediction, y)

    return metric


@partial(jax.jit, static_argnames=("ensemble_model", "train_params"))
def loss_old(
    ensemble_params: dict,
    key: jax.Array,
    train_params: TrainParams,
    ensemble_model: Ensemble,
    x: jax.Array,
    y: jax.Array,
):
    
    task_type = train_params.task.task_type()
    ambiguity_gradient_predictors = train_params.ambiguity_gradient_predictors
    ambiguity_gradient_delegators = train_params.ambiguity_gradient_delegators
    n_predictors = train_params.n_predictors
    batch_size = x.shape[0]
    agg_delegation: jax.Array = None # (BS, n_predictors), will be probabilities
    agg_prediction: jax.Array = None # (BS, out)
    performance_loss_per_model: jax.Array = None # (BS, n_predictors)
    ambiguity_per_model: jax.Array = None # (BS, n_predictors)
 
    assert ambiguity_gradient_predictors in {"both", "delegators", "none"}

    forward_args = ForwardArgs(x)
    forward_return = ensemble_model.apply({"params": ensemble_params}, forward_args)
    predictions = forward_return.predictions # (BS, n_predictors, out)
    predictions_no_gradient = jax.lax.stop_gradient(predictions) if ambiguity_gradient_predictors != "both" else predictions 
    delegations_logits = forward_return.delegations # (BS, n_delegators, n_predictors)

    # Aggregate delegators
    agg_delegation_for_performance = agg_delegation = aggregate_delegators(train_params, delegations_logits)

    if not ambiguity_gradient_delegators:
        agg_delegation_for_performance = jax.lax.stop_gradient(agg_delegation_for_performance)
    
    # Aggregate predictors for centroid ambiguity calculations
    # Ambiguity calculation does not influence predictors
    if task_type == "classification":
        agg_prediction = mix_weighted_logits(predictions_no_gradient, agg_delegation_for_performance) 
    elif task_type == "regression":
        agg_prediction = mix_weighted_mean(predictions_no_gradient, agg_delegation_for_performance)


    metrics = {}

    # Calc losses
    if task_type == "classification":
        performance_loss_per_model = ce_loss(predictions, y)
        ambiguity_per_model = jax.vmap(kl_ambiguity, in_axes=(None, 1), out_axes=1)(
            agg_prediction, 
            predictions_no_gradient
        )
        metrics["accuracy_metric"] = classification_accuracy(agg_prediction, y)
    elif task_type == "regression":
        performance_loss_per_model = mse_loss(predictions, y)
        ambiguity_per_model = jax.vmap(var_ambiguity, in_axes=(None, 1), out_axes=1)(
            agg_prediction, 
            predictions_no_gradient
        )
        metrics["r2_metric"] = regression_r2(agg_prediction, y)
    
    # Load balancing loss
    batch_agg_delegation = agg_delegation.mean(axis=0)
    batch_agg_delegation = batch_agg_delegation / batch_agg_delegation.sum() # Shouldn't be needed
    model_usage_uniformity = gini_impurity(batch_agg_delegation) # 0 - fully pure; 1 - fully uniform 
    load_balancing_loss = train_params.load_balancing_lambda * (1 - model_usage_uniformity)

    assert performance_loss_per_model.shape == (batch_size, n_predictors), ambiguity_per_model.shape == (batch_size, n_predictors)
    weighted_perfomance = jnp.sum(agg_delegation_for_performance * performance_loss_per_model, axis=-1)
    weighted_ambiguity = jnp.sum(agg_delegation_for_performance * ambiguity_per_model, axis=-1)

    loss_per_sample =  (
        weighted_perfomance - 
        (weighted_ambiguity if ambiguity_gradient_predictors != "none" else jax.lax.stop_gradient(weighted_ambiguity)) 
    )
    performance_loss = jnp.mean(loss_per_sample)


    # Ambiguity delegator gradient
    # Ambiguity delegator gradient
    if ambiguity_gradient_delegators:
        individual_delegator_loss = 0.0
    else:
        delegations_probs = jax.nn.softmax(
            delegations_logits,
            axis=-1,
        )  # (BS, n_delegators, n_predictors)

        individual_delegator_losses = jax.vmap(
            delegator_predictor_ensemble_loss,
            in_axes=(None, None, 1, None, None),
            out_axes=1,
        )(
            predictions,
            y,
            delegations_probs,
            task_type,
            ambiguity_gradient_predictors,
        )  # (BS, n_delegators)

        individual_delegator_loss = jnp.mean(
            individual_delegator_losses
        )

    
    return performance_loss + load_balancing_loss + individual_delegator_loss, {"performance_loss": performance_loss, "load_balancing_loss": load_balancing_loss} | metrics




@partial(jax.jit, static_argnames=("ensemble_model", "train_params"))
def loss(
    ensemble_params: dict,
    key: jax.Array,
    train_params: TrainParams,
    ensemble_model: Ensemble,
    x: jax.Array,
    y: jax.Array,
):
    
    task_type = train_params.task.task_type()
    ambiguity_gradient_predictors = train_params.ambiguity_gradient_predictors
    ambiguity_gradient_delegators = train_params.ambiguity_gradient_delegators
    n_predictors = train_params.n_predictors
    batch_size = x.shape[0]
    agg_delegation: jax.Array = None # (BS, n_predictors), will be probabilities
    agg_prediction: jax.Array = None # (BS, out)
    performance_loss_per_model: jax.Array = None # (BS, n_predictors)
    ambiguity_per_model: jax.Array = None # (BS, n_predictors)
    metrics = {}
 
    assert ambiguity_gradient_predictors in {"both", "delegators", "none"}

    forward_args = ForwardArgs(x)
    forward_return = ensemble_model.apply({"params": ensemble_params}, forward_args)
    predictions = forward_return.predictions # (BS, n_predictors, out)
    predictions_no_gradient = jax.lax.stop_gradient(predictions)
    delegations_logits = forward_return.delegations # (BS, n_delegators, n_predictors)

    # Aggregate delegators
    agg_delegation = aggregate_delegators(train_params, delegations_logits)

    if not ambiguity_gradient_delegators:
        agg_delegation_for_predictor_ambiguity = jax.lax.stop_gradient(agg_delegation)
        agg_delegation_for_predictor_performance = jax.lax.stop_gradient(agg_delegation)
    elif ambiguity_gradient_predictors == "none":
        agg_delegation_for_predictor_ambiguity = jax.lax.stop_gradient(agg_delegation)
        agg_delegation_for_predictor_performance = agg_delegation
    else:
        agg_delegation_for_predictor_ambiguity = agg_delegation
        agg_delegation_for_predictor_performance = agg_delegation


    (performance_loss_per_model, ambiguity_per_model), agg_prediction = predictor_error_ambiguity_decomposition(
        predictions,
        y,
        task_type=task_type,
        agg_delegation=jax.lax.stop_gradient(agg_delegation),
        return_agg_prediction=True,
        stop_predictions_ambiguity_gradient=(ambiguity_gradient_predictors != "both")
    )

    assert performance_loss_per_model.shape == (batch_size, n_predictors), ambiguity_per_model.shape == (batch_size, n_predictors)

    weighted_perfomance = jnp.sum(agg_delegation_for_predictor_performance * performance_loss_per_model, axis=-1)
    weighted_ambiguity = jnp.sum(agg_delegation_for_predictor_ambiguity * ambiguity_per_model, axis=-1)

    loss_per_sample = weighted_perfomance - weighted_ambiguity
    performance_loss = jnp.mean(loss_per_sample)
    
    # Load balancing loss
    batch_agg_delegation = agg_delegation.mean(axis=0)
    batch_agg_delegation = batch_agg_delegation / batch_agg_delegation.sum() # Shouldn't be needed
    model_usage_uniformity = gini_impurity(batch_agg_delegation) # 0 - fully pure; 1 - fully uniform 
    load_balancing_loss = train_params.load_balancing_lambda * (1 - model_usage_uniformity)

    # Ambiguity delegator gradient
    if not ambiguity_gradient_delegators:
        performance_loss_per_delegator, ambiguity_per_delegator = delegator_error_ambiguity_decomposition(
            delegations=delegations_logits,
            predictions=predictions_no_gradient,
            y=y,
            train_params=train_params
        )

        via_delegator_performance = jnp.mean(performance_loss_per_delegator, axis=-1) 
        via_delegator_ambiguity = jnp.mean(ambiguity_per_delegator, axis=-1) 
        via_delegator_loss_per_sample = via_delegator_performance - via_delegator_ambiguity
        via_delegator_performance_loss = jnp.mean(via_delegator_loss_per_sample)

        # I am taking the mean and kind assuming that they have they same units which they should have
        # I also think they should be the same number, to only point to this is to route gradients
        performance_loss = (
            performance_loss
            + via_delegator_performance_loss 
            - jax.lax.stop_gradient(via_delegator_performance_loss)
        )
    

    # Calc metrics
    if task_type == "classification":
        metrics["accuracy_metric"] = classification_accuracy(agg_prediction, y)
    elif task_type == "regression":
        metrics["r2_metric"] = regression_r2(agg_prediction, y)

    
    return performance_loss + load_balancing_loss, {"performance_loss": performance_loss, "load_balancing_loss": load_balancing_loss} | metrics


def delegator_predictor_ensemble_loss(
    predictions: jax.Array,   # (BS, n_predictors, out)
    y: jax.Array,
    weights: jax.Array,       # (BS, n_predictors)
    task_type: Literal["classification", "regression"],
    ambiguity_gradient_predictors: Literal["both", "delegators", "none"],
):
    predictions = jax.lax.stop_gradient(predictions)

    performance_per_model, ambiguity_per_model = (
        predictor_error_ambiguity_decomposition(
            predictions,
            y,
            task_type,
            weights,
        )
    )

    weighted_performance = jnp.sum(
        weights * performance_per_model,
        axis=-1,
    )

    weighted_ambiguity = jnp.sum(
        weights * ambiguity_per_model,
        axis=-1,
    )

    return (
        weighted_performance
        -
        (
            weighted_ambiguity
            if ambiguity_gradient_predictors != "none"
            else jax.lax.stop_gradient(weighted_ambiguity)
        )
    )

def classification_accuracy(
    agg_prediction: jax.Array,
    y: jax.Array,
) -> jax.Array:
    return jnp.mean(jnp.argmax(agg_prediction, axis=-1) == y)


def regression_r2(
    agg_prediction: jax.Array,
    y: jax.Array,
) -> jax.Array:
    ss_res = jnp.sum((y - agg_prediction) ** 2)
    ss_tot = jnp.sum((y - jnp.mean(y, axis=0)) ** 2)
    return 1 - ss_res / ss_tot

def aggregate_delegators(
        train_params: TrainParams,
        delegations_logits: jax.Array
    ):
    
    delegators_mixing = train_params.delegators_mixing
    agg_delegation: jax.Array = None # (BS, n_predictors), will be probabilities
    delegations_logprobs = jax.nn.log_softmax(delegations_logits, axis=-1)
    delegations_probs = jax.nn.softmax(delegations_logits, axis=-1)
    
    if delegators_mixing == "product":
        # Mix logprobs
        agg_delegation = jnp.mean(delegations_logprobs, axis=-2)
        agg_delegation = jax.nn.softmax(agg_delegation, axis=-1) 
    elif delegators_mixing == "sum":
        # Mix probs
        agg_delegation = jnp.mean(delegations_probs, axis=-2)
        agg_delegation = agg_delegation / jnp.sum(agg_delegation, axis=-1, keepdims=True)
    
    return agg_delegation

def mix_weighted_mean(
        y: jax.Array, # (BS, n_predictors, out)
        weights: jax.Array # (BS, n_predictors)
    ) -> jax.Array:
        assert y.ndim == (weights.ndim + 1)
        return jnp.sum(y * jnp.expand_dims(weights, axis=-1), axis=1)
      

def mix_weighted_logits(
        logits: jax.Array, # (BS, n_predictors, out)
        weights: jax.Array # (BS, n_predictors)
    ) -> jax.Array:

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    weights = jnp.expand_dims(weights, axis=-1)
    mixed_weighted_logits = jnp.sum(log_probs * weights, axis=-2)
    return mixed_weighted_logits

def ce_loss(
    predictions_logits: jax.Array, # (BS, n_predictors, out)
    labels: jax.Array # (BS,)
):
    one_loss = lambda logits: optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jax.vmap(one_loss, in_axes=1, out_axes=1)(predictions_logits)

def ce_loss_no_integers(
    predictions_logits: jax.Array, # (BS, n_predictors, out)
    target_probs: jax.Array # (BS, out)
):
    one_loss = lambda logits: optax.safe_softmax_cross_entropy(logits, target_probs)
    return jax.vmap(one_loss, in_axes=1, out_axes=1)(predictions_logits)

def mse_loss(
    predictions: jax.Array,
    y: jax.Array
):
    assert (predictions.shape[0] == y.shape[0]) and (predictions.shape[-1] == y.shape[-1]) and (predictions.ndim == 3) and (y.ndim == 2) 
    return jnp.mean((predictions - y[:, jnp.newaxis, :]) ** 2, axis=-1)


def var_ambiguity(
    centroid: jax.Array,
    other: jax.Array
):
    return jnp.mean((centroid - other) ** 2, axis=-1)

def kl_ambiguity(
    centroid_logits: jax.Array,
    other_logits: jax.Array,
):
    centroid_log_probs = jax.nn.log_softmax(centroid_logits, axis=-1)
    other_log_probs = jax.nn.log_softmax(other_logits, axis=-1)

    return jnp.sum(
        jnp.exp(centroid_log_probs)
        * (centroid_log_probs - other_log_probs),
        axis=-1,
    )


def probability_mixing_ambiguity(
    true: jax.Array,
    centroid: jax.Array,
    other: jax.Array,
    epsilon: float = 1e-6,
):
    mixed_safe = jnp.clip(centroid, min=epsilon)
    other_safe = jnp.clip(other, min=epsilon)

    return jnp.sum(
        true
        * (jnp.log(mixed_safe) - jnp.log(other_safe)),
        axis=-1,
    )


def gini_impurity(
    dist: jax.Array,
    axis: int | None = None,
):  # 0 - fully pure; 1 - fully uniform 
    if axis is None:
        assert dist.ndim == 1
        axis = 0

    n = dist.shape[axis]
    impurity = 1 - jnp.sum(dist**2, axis=axis)

    return impurity / (1 - 1 / n)
