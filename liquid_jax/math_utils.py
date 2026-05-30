import jax
import jax.numpy as jnp
import optax

def mix_weighted_mean(
        y: jax.Array,
        weights: jax.Array
    ) -> jax.Array:
        
        # y (batch, n_models, n_out)
        # power (batch, n_models)
        assert y.ndim == (weights.ndim + 1)
        return jnp.sum(y * jnp.expand_dims(weights, axis=-1), axis=1)


def mix_weighted_logits_to_probs(
    logits: jax.Array,
    weights: jax.Array
) -> jax.Array:
    
    assert logits.ndim == (weights.ndim + 1)
    probs = jax.nn.softmax(logits, axis=-1)
    return mix_weighted_mean(probs, weights)
      

def mix_weighted_logits(logits: jax.Array, weights: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    weights = jnp.expand_dims(weights, axis=-1)
    mixed_log_probs = jnp.sum(log_probs * weights, axis=-2)
    return jax.nn.log_softmax(mixed_log_probs, axis=-1)

def ce_loss(
    logits: jax.Array,
    labels: jax.Array
):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)

def ce_loss_logprobs_labels(
    logprobs: jax.Array, # log_2 = log_e (p)
    labels: jax.Array
):
    n_classes = logprobs.shape[-1]
    I = jnp.eye(n_classes)
    onehot = I[labels]
    loss = -jnp.sum(onehot * logprobs, axis=-1)
    return loss / jnp.log(2)

def mse_loss(
    yhat: jax.Array,
    y: jax.Array
):
    assert yhat.shape == y.shape
    return (yhat - y) ** 2

def jsd(
    logits: jax.Array,  # (BS, models, classes)
    weights: jax.Array  # (BS, models)
) -> jax.Array:
    
    # Expand weights to broadcast across classes: (BS, models, 1)
    w_expanded = jnp.expand_dims(weights, axis=-1)
    
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # 1. Calculate the weighted mean probability distribution across models
    mean_probs = jnp.sum(probs * w_expanded, axis=1, keepdims=True)
    
    # 2. Log of mean probs (clamped to prevent log(0) - numerical stability)
    log_mean_probs = jnp.log(jnp.clip(mean_probs, a_min=1e-8))
    
    # 3. KL divergence between each model's distribution and the mean distribution -> (BS, models)
    kl = jnp.sum(probs * (log_probs - log_mean_probs), axis=-1)
    
    # 4. JSD is the weighted sum of the KL divergences across the ensemble models
    jsd_per_sample = jnp.sum(weights * kl, axis=1) # (BS,)

    return jsd_per_sample

def bregman_divergence(
    weights: jax.Array, # (BS, models)
    logits: jax.Array, # (BS, models, classes)
    labels: jax.Array  # (BS,)
):
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Gather probabilities for the target class across all models
    target_probs = jnp.take_along_axis(probs, labels[:, jnp.newaxis, jnp.newaxis], axis=-1).squeeze(-1)
    eps = 1e-7
    target_probs = jnp.clip(target_probs, eps, 1.0)
    # (BS, models)

    mean_target_probs = jnp.sum(weights * target_probs, axis=1, keepdims=True)
    mean_target_probs = jnp.clip(mean_target_probs, eps, 1.0)
    # (BS, 1)

    # 3. Calculate Bregman Divergence with g(t) = -log(t)
    # d_{-log}(p, p_mean) = -log(p) + log(p_mean) + (p - p_mean) / p_mean
    bregman_div = -jnp.log(target_probs) + jnp.log(mean_target_probs) + (target_probs - mean_target_probs) / mean_target_probs
    bregman_div = bregman_div * weights
    # (BS, models)
    return jnp.sum(bregman_div, axis=-1)

def mse_loss(
    yhat: jax.Array,
    y: jax.Array
):
    assert yhat.shape == y.shape
    return (yhat - y) ** 2

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


def mix_mean(
        y: jax.Array,
    ) -> jax.Array:
        # y (batch, n_models, n_out)
        n_models = y.shape[1]
        weights = jnp.ones((1, n_models)) / n_models
        return mix_weighted_mean(y, weights)

def mix_logits(
    logits: jax.Array,
) -> jax.Array:
    
    raise ValueError("Implement mixing probabilities like we said in the paper")

    # logits (batch, n_models, n_out)
    n_models = logits.shape[1]
    weights = jnp.ones((1, n_models)) / n_models
    
    return mix_weighted_logits(logits, weights)

