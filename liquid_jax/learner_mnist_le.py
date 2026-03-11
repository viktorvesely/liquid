from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner
from liquid_solver import LEsolver, LEInfo


solver = LEsolver()

import jax 
import jax.numpy as jnp
from flax import linen as nn

def get_layers(neurons: tuple[int, ...]):

    if not neurons:
        return None
    
    layers = []
    for n in neurons:
        layers.append(
            nn.Dense(n)
        )
    return tuple(layers)

def forward(h, last_linear, layers):
    if not layers: 
        return h
    
    for layer in layers[:-1]:
        h = nn.relu(layer(h))
    
    final_layer = layers[-1]
    h = final_layer(h)
    
    return h if last_linear else nn.relu(h)   

class DeModelMlp(nn.Module):
    
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None
    
    def setup(self):
        self.body_layers = get_layers(self.body)
        self.out_layers = get_layers(self.out)
        self.delegation_layers = get_layers(self.delegation)

    def __call__(self, x: jax.Array):
        h = x
        h_body = forward(h, last_linear=False, layers=self.body_layers)
        y = forward(h_body, last_linear=True, layers=self.out_layers)
        d = forward(h_body, last_linear=True, layers=self.delegation_layers)

        return y, d

class LeMlp(nn.Module):
    n_models: int
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None

    def setup(self):
        
        assert self.delegation[-1] == self.n_models, "Last dim of delegation needs to equal to n_models"

        VmappedDeModelMlp = nn.vmap(
            DeModelMlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.ensemble = VmappedDeModelMlp(
            out=self.out, 
            delegation=self.delegation, 
            body=self.body
        )

    def __call__(self, x: jax.Array):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        y, d = self.ensemble(x)
        d = nn.softmax(d, axis=-1)
        return y, d
            

class LeMnistLearner(Learner[LEInfo]):

    @staticmethod
    def get_model() -> nn.Module:
        
        n_models = 10

        return LeMlp(
            n_models=n_models,
            body=(128, 64),
            out=(32, 10),
            delegation=(32, n_models)
        )

    @staticmethod
    def forward(key: jax.Array, x: jax.Array, model: LeMlp, params: dict) -> tuple[jax.Array, LEInfo]:
        
        ys, delegation = model.apply({"params": params}, x)
        leinfo = solver.solve_power(delegation)

        y = solver.mix_power_logits(ys, leinfo.power)

        return y, leinfo
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        train_return: LEInfo
    ) -> dict[str, jax.Array]:
        
        return {
            "load_distribution_loss": solver.load_distribution_loss(train_return),
            "specialization_losss": solver.specialization_loss(train_return)
        }
