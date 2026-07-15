import math

import jax
import jax.numpy as jnp
from structs import InOutData


def train_loader(
    key: jax.Array,
    inout: InOutData,
    batch_size: int,
    desired_batches: int | None = None,
    serve_as_is: bool = False
):
    
    actual_size = inout.x.shape[0]

    if serve_as_is:
        desired_batches = int(math.ceil(actual_size / batch_size))
        desired_size = actual_size
    else:
        assert desired_batches is not None
        desired_size = batch_size * desired_batches
    

    difference = desired_size - actual_size
    assert difference < actual_size

    k1, k2, k_next = jax.random.split(key, 3)

    actual_inds = jnp.arange(actual_size)
    proper_inds = jax.random.permutation(k1, actual_size)
    added_inds = jax.random.permutation(k2, actual_size)[:difference] 
    
    all_inds = jnp.concatenate((proper_inds, added_inds))
    
    for i_batch in range(desired_batches):
        start = i_batch * batch_size
        end = start + batch_size

        if serve_as_is:
            batch_inds = actual_inds[start:end]
        else:
            batch_inds = all_inds[start:end]

        batch = jax.tree.map(lambda x: x[batch_inds, ...], inout)

        k_use = jax.random.fold_in(k_next, i_batch)
        yield batch, k_use