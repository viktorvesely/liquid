from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from structs import ForwardArgs, ForwardReturn, TrainParams, Model
@partial(jax.jit, static_argnames=("train_params", "steps"))
def optimal_convex_weights(
    y: jax.Array,
    predictions: jax.Array,
    train_params: TrainParams,
    steps: int = 5_000
):

    batch_size = y.shape[0]
    n_predictors = predictions.shape[1]
    optimizer = optax.adam(learning_rate=1e-3)

    def init(w):
        return optimizer.init(w)

    weights = jnp.full(
        (batch_size, n_predictors),
        1 / n_predictors
    )
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

def loss_decomposition

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

@partial(jax.jit, static_argnames=("ensemble_model", "train_params"))
def loss(
    ensemble_params: dict,
    key: jax.Array,
    train_params: TrainParams,
    ensemble_model: Model,
    x: jax.Array,
    y: jax.Array,
):
    
    task_type = train_params.task.task_type()
    delegators_mixing = train_params.delegators_mixing
    n_delegators = train_params.n_delegators
    n_predictors = train_params.n_predictors
    batch_size = x.shape[0]
    agg_delegation: jax.Array = None # (BS, n_predictors), will be probabilities
    agg_prediction: jax.Array = None # (BS, out)
    perfomance_loss_per_model: jax.Array = None # (BS, n_predictors)
    ambiguity_per_model: jax.Array = None # (BS, n_predictors)
 
    forward_args = ForwardArgs(key, x)
    forward_return = ensemble_model.apply(ensemble_params, forward_args)
    predictions = forward_return.predictions # (BS, n_predictors, out)
    predictions_no_gradient = jax.lax.stop_gradient(predictions)
    delegations_logits = forward_return.delegations # (BS, n_delegators, n_predictors)
    delegations_logprobs = jax.nn.log_softmax(delegations_logits, axis=-1)
    delegations_probs = jax.nn.softmax(delegations_logits, axis=-1)

    # Aggregate delegators
    if n_delegators == 0:
        # Unifrom weights
        agg_delegation = jnp.full((batch_size, n_predictors), 1 / n_predictors)
    elif delegators_mixing == "product":
        # Mix logprobs
        agg_delegation = jnp.mean(delegations_logprobs, axis=-2)
        agg_delegation = jnp.exp(agg_delegation)
        agg_delegation = agg_delegation / jnp.sum(agg_delegation, axis=-1, keepdims=True)  
    elif delegators_mixing == "sum":
        # Mix probs
        agg_delegation = jnp.mean(delegations_probs, axis=-2)
        agg_delegation = agg_delegation / jnp.sum(agg_delegation, axis=-1, keepdims=True)  
    
    # Aggregate predictors for centroid ambiguity calculations
    # Ambiguity calculation does not influence predictors
    if task_type == "classification":
        agg_prediction = mix_weighted_logits(predictions_no_gradient, agg_delegation) 
    elif task_type == "regression":
        agg_prediction = mix_weighted_mean(predictions_no_gradient, agg_delegation)

    # Calc losses
    if task_type == "classification":
        perfomance_loss_per_model = ce_loss(predictions, y)
        ambiguity_per_model = jax.vmap(kl_ambiguity, in_axes=(None, 1), out_axes=1)(
            jax.nn.softmax(agg_prediction), 
            jax.nn.softmax(predictions_no_gradient)
        )
    elif task_type == "regression":
        perfomance_loss_per_model = mse_loss(predictions, y)
        ambiguity_per_model = jax.vmap(var_ambiguity, in_axes=(None, 1), out_axes=1)(
            agg_prediction, 
            predictions_no_gradient
        )
    
    assert perfomance_loss_per_model.shape == (batch_size, n_predictors), perfomance_loss_per_model.shape

    loss_per_sample = jnp.sum(agg_delegation * perfomance_loss_per_model, axis=-1) - jnp.sum(agg_delegation * ambiguity_per_model, axis=-1)
    
    return jnp.mean(loss_per_sample)

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
    centroid: jax.Array,
    other: jax.Array,
    epsilon: float = 1e-6,
):
    centroid_safe = jnp.clip(centroid, min=epsilon)
    other_safe = jnp.clip(other, min=epsilon)

    kl_per_class = centroid * (
        jnp.log(centroid_safe) - jnp.log(other_safe)
    )

    return jnp.sum(kl_per_class, axis=-1)
    
def non_uniformity(
    dist: jax.Array
):
    assert dist.ndim == 1, "Implement some form of axing"

    n = dist.shape[-1]
    # Make it like uniform (more stable than maximizing entropy)
    non_uniformity = n* jnp.sum(dist ** 2) - 1
    
    # Range before: [0, n_models - 1], now [0, 1]
    non_uniformity = non_uniformity / (n - 1)

    return non_uniformity 


