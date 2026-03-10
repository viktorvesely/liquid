

from functools import partial
import math
from typing import Callable
import warnings

import jax
import optax
import tqdm
from flax import linen as nn
from flax import struct
import jax.numpy as jnp

from mnist import Mnist
from learner_base import Learner

from structs import TrainParams

params = TrainParams(
    batch_size=512,
    preload_batches_to_gpu=5,
    valid_batches=2,
    epochs=2,
    lr=1e-3,
    optimizer="adam",
    performance_loss="ce",
    task=Mnist,
    learner=None
)


@struct.dataclass
class InOutData:
    
    x: jax.Array
    y: jax.Array
    where: jax.Array

@partial(jax.jit, static_argnames=("optimizer", "net", "train_params"))
def loss_fn(
    network_params: dict,
    key: jax.Array,
    inout: InOutData,
    model: nn.Module,
    train_params: TrainParams
):
    
    k_forward, k_loss = jax.random.split(key)

    yhat, train_return = train_params.learner.forward(
        key=k_forward,
        x=inout.x,
        model=model,
        params=network_params
    )

    if train_params.performance_loss == "ce":
        loss_batch = optax.softmax_cross_entropy_with_integer_labels(
            yhat, inout.y, where=inout.where
        )
    else:
        raise ValueError(f"loss={train_params.performance_loss} is not implemented")
    
    
    performance_loss = jnp.sum(loss_batch) / jnp.sum(inout.where)
    auxillary_losses = train_params.learner.auxillary_losses(k_loss, train_return, inout.where)
    losses = {train_params.performance_loss: performance_loss} | auxillary_losses

    aux_values = jax.tree.reduce(lambda accum, aux_loss: accum + aux_loss, auxillary_losses)
    loss = aux_values + performance_loss
    
    return loss, losses


@partial(jax.jit, static_argnames=("optimizer", "net", "train_params"))
def train_batch(
    key: jax.Array,
    inout_data: InOutData,
    network_params: dict,
    opt_state: dict,
    optimizer: optax.GradientTransformationExtraArgs,
    model: nn.Module,
    train_params: TrainParams
):
    
    # To batches
    bs = train_params.batch_size
    assert (inout_data.x.shape[0] % bs) == 0, "Gpu batch needs to be devisible by the batch_size"
    n_batches = inout_data.x.shape[0] // bs
    inout_data = jax.tree.map(lambda x: x.reshape((n_batches, bs) + x.shape[1:]), inout_data)

    
    def train_step(carry, inout_batch: InOutData):
        
        key, network_params, opt_state = carry
        key, k_use = jax.random.split(key)


        (loss, losses), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            network_params=network_params,
            key=k_use,
            inout=inout_batch,
            model=model,
            train_params=train_params
        )

        updates, opt_state = optimizer.update(grad, opt_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return (key, network_params, opt_state), (loss, losses)
    

    (key, network_params, opt_state), (loss, losses) =jax.lax.scan(train_step, (key, network_params, opt_state), inout_data)
    
    return (network_params, opt_state), (loss, losses)



def train(key: jax.Array, train_params: TrainParams):

    
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]
    

    key, k_init, k_loop = jax.random.split(key, 3) 

    # Data
    fullData = train_params.task.load_cpu(split="train")
    x, y = train_params.task.get_xy(fullData)
    inout_data = InOutData(
        x=x, y=y, where=jnp.ones((x.shape[0],), dtype=jnp.bool, device=cpu)
    )

    # Padd data to align wit gpu batch size
    gpu_batch = train_params.batch_size * train_params.preload_batches_to_gpu
    n_data = x.shape[0]
    n_batches = math.ceil(n_data / gpu_batch)
    n_pad = (gpu_batch - (n_data % gpu_batch)) % gpu_batch
    if n_pad > 0:
        inout_data =  jax.tree.map(lambda x: jnp.concatenate((
            x,
            jnp.zeros((n_pad,) + x.shape[1:], dtype=x.dtype, device=x.device)
        ), axis=0), inout_data)

    p_fake = n_pad / gpu_batch
    T_fake = 0.4
    if p_fake > T_fake:
        warnings.warn(f"{int(p_fake * 100)}% ({n_pad} samples) of the last batch are padded samples, consider cutting of the data")

    # Train, Valid split
    n_valid = train_params.batch_size * train_params.valid_batches
    inout_valid = jax.tree.map(lambda x: jax.device_put(x[:n_valid, ...], device=gpu), inout_data)
    inout_train = jax.tree.map(lambda x: x[n_valid:, ...], inout_data)

    # Model
    model = train_params.learner.get_model()
    model_params = model.init(k_init, x[[0], ...])["params"]

    # Optimizer
    optimizer = {
        "sgd": optax.sgd(learning_rate=train_params.lr),
        "adam": optax.adamw(learning_rate=train_params.lr)
    }[train_params.optimizer]
    opt_state = optimizer.init(model_params)
    
    for i_epoch in tqdm.tqdm(range(train_params.epochs)):
        for i_batch in range(n_batches):

            start = i_batch * gpu_batch
            end = start + gpu_batch
            inout_batch = jax.tree.map(lambda x: jax.device_put(x[start, end, ...], device=gpu), inout_train)
        
        

            


    

