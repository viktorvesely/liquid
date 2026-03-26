import jax
import jax.numpy as jnp

def symexp(x: jax.Array):
    return jnp.sign(x) * (jnp.power(10, jnp.abs(x)) - 1)

def symlog(x: jax.Array):
    return jnp.sign(x) *  jnp.log10(jnp.abs(x) + 1)


def bin_stops(
        x: jax.Array,
        n_bins: int,
        two_way_continous: bool = True,
        epsilon: float = 1e-6
    ):
    n = n_bins + (0 if two_way_continous else 1)
    stops = jnp.linspace(x.min(), x.max() + epsilon, endpoint=True, num=n)
    return stops

def bin_encode(
        x: jax.Array,
        stops: jax.Array,
        two_way_continous: bool = True,
        epsilon: float = 1e-6
    ):

    x = jnp.clip(x, stops[0], stops[-1] - epsilon)    

    n = stops.shape[0]
    n_bins = n - (0 if two_way_continous else 1)

    if two_way_continous:
        lower_i = jnp.searchsorted(stops, x, "right") - 1
        lower_i = jnp.clip(lower_i, 0, n - 2)
        l = stops[lower_i]
        h = stops[lower_i + 1]
        t = (x - l) / (h - l) 

        n_samples = x.shape[0]
        encoded = jnp.zeros((n_samples, n_bins))
        sample_inds = jnp.arange(n_samples, dtype=jnp.int32)

        encoded = encoded.at[sample_inds, lower_i].set((1 - t))
        encoded = encoded.at[sample_inds, lower_i + 1].set((t))
    else:
        bucket_i = jnp.searchsorted(stops, x, "right") - 1
        eye = jnp.eye(n_bins)
        
        encoded = eye[bucket_i]

    return encoded

def bin_decode(
        encoded: jax.Array,
        stops: jax.Array,
        two_way_continous: bool = True
    ):

    if two_way_continous:
        decoded = jnp.squeeze(encoded @ stops[:, jnp.newaxis], -1)
    else:
        # Center is the least bias estimator given stops
        bucket_i = jnp.argmax(encoded, axis=-1)
        l = stops[bucket_i]
        h = stops[bucket_i + 1]
        decoded =  l + (h - l) / 2
    
    return decoded
    