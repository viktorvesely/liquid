import jax
import jax.numpy as jnp
from structs import InOutData


def train_loader(
    key: jax.Array,
    inout: InOutData,
    batch_size: int,
    desired_batches: int
):
    
    desired_size = batch_size * desired_batches
    actual_size = inout.x.shape[0]
    difference = desired_size - actual_size
    assert difference < actual_size

    k1, k2, k_next = jax.random.split(key, 3)

    proper_inds = jax.random.permutation(k1, actual_size)
    added_inds = jax.random.permutation(k2, difference) 
    
    all_inds = jnp.concatenate((proper_inds, added_inds))
    
    for i_batch in range(desired_batches):
        start = i_batch * batch_size
        end = start + batch_size

        batch_inds = all_inds[start:end]
        batch = jax.tree.map(lambda x: x[batch_inds, ...], inout)

        yield batch, k_next