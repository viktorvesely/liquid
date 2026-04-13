import jax
import jax.numpy as jnp

def mix_weighted_mean(
        y: jax.Array,
        weights: jax.Array
    ) -> jax.Array:
        
        # y (batch, n_models, n_out)
        # power (batch, n_models)
        assert y.ndim == (weights.ndim + 1)
        assert y.ndim == 3
        return jnp.sum(y * jnp.expand_dims(weights, axis=-1), axis=1)

def mix_weighted_logits(
    logits: jax.Array,
    weights: jax.Array
) -> jax.Array:

    # logits (batch, n_models, n_out)
    # power (batch, n_models)
    assert logits.ndim == (weights.ndim + 1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Expand power to (batch, n_models, 1) for broadcasting over n_out
    weights = jnp.expand_dims(weights, axis=-1)
    
    # Compute the weighted sum (Weighted Geometric Mean in probability space)
    mixed_log_probs = jnp.sum(log_probs * weights, axis=1)
    
    return mixed_log_probs



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
    # logits (batch, n_models, n_out)
    n_models = logits.shape[1]
    weights = jnp.ones((1, n_models)) / n_models
    
    return mix_weighted_logits(logits, weights)