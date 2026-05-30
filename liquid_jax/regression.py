from typing import Literal

from flax import struct
import jax
import jax.numpy as jnp


def normalize_feature(x: jax.Array) -> jax.Array:
    mean = jnp.mean(x)
    std = jnp.std(x)
    return (x - mean) / std

def int_to_onehot(x: jax.Array, n_classes: int) -> jax.Array:
    return jax.nn.one_hot(x, n_classes)

def dict_to_x(
        data: dict[str, jax.Array],
        normalize_features: tuple[str, ...] = (),
        onehot_features: tuple[str, ...] = (),
        noop_features: tuple[str, ...] = (),
    ) -> jax.Array:
    
    X = []

    for nf in normalize_features:
        f = normalize_feature(data[nf])
        assert f.ndim == 1
        X.append(f[:, jnp.newaxis])

    for of in onehot_features:
        f = data[of].astype(jnp.int32)
        assert f.ndim == 1
        low, high = f.min(), f.max()
        assert low == 0, f"feature: {of}, min = {low}"
        assert high != 0
        X.append(int_to_onehot(f, high + 1))
        
    for nof in noop_features:
        f = data[nof]
        assert f.ndim == 1
        X.append(f[:, jnp.newaxis])

    return jnp.concatenate(X, axis=-1)
