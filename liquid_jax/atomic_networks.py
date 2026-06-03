from __future__ import annotations

from flax import struct
import flax.linen as nn
import jax

@struct.dataclass
class Architecture:
    predictor: tuple[int, ...]
    delegator: tuple[int, ...]
    cnn: int = 0

    def determine_size(
            self,
            predictor_base: int,
            delegator_base: int,
            out_dim: int,
            n_predictors: int
        ) -> Architecture:
        arch = Architecture(
            predictor=tuple((predictor_base * l) for l in self.predictor) + (out_dim,),
            delegator=tuple((delegator_base * l) for l in self.delegator) + (n_predictors,),
            cnn=self.cnn
        )
        print(arch)
        return arch
    
two_layer_mlp = Architecture(
    predictor=(2, 1),
    delegator=(2, 1)
)

three_layer_mlp = Architecture(
    predictor=(3, 2, 1),
    delegator=(3, 2, 1)
)

small_cnn = Architecture(
    predictor=(1, 2, 4),
    delegator=(1, 2, 4),
    cnn=3
)

big_cnn = Architecture(
    predictor=(1, 2, 4, 8),
    delegator=(1, 2, 4, 8),
    cnn=4
)  

def get_layers(neurons: tuple[int, ...], bias_std: float = 1e-4):

    if not neurons:
        return None
    
    layers = []
    for n in neurons:
        layers.append(
            nn.Dense(n)
        )
    return tuple(layers)



def get_cnn_layers(
    channels: tuple[int, ...],
    kernel_size: int,
    stride: int
    ):
    
    if not channels:
        return None
    
    layers = []
    for c in channels:
        layers.append(
            nn.Conv(c, kernel_size, stride)
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

class Mlp(nn.Module):
    
    body: tuple[int, ...]
    last_linear: bool = True
    
    def setup(self):
        self.body_layers = get_layers(self.body)

    def __call__(self, x: jax.Array):
        out = forward(x, last_linear=self.last_linear, layers=self.body_layers)
        return out
    

class Cnn(nn.Module):
    
    body: tuple[int, ...]
    
    def setup(self):
        self.body_layers = get_cnn_layers(self.body)

    def __call__(self, x: jax.Array):
        out = forward(x, last_linear=True, layers=self.body_layers)
        return out
    
class CnnMlp(nn.Module):

    cnn: tuple[int, ...]
    mlp: tuple[int, ...]
    kernel_size: int
    stride: int
    

    def setup(self):
        self.cnn_layers = get_cnn_layers(self.cnn, kernel_size=self.kernel_size, stride=self.stride)
        self.mlp_layers = get_layers(self.mlp)

    def __call__(self, x: jax.Array):
        h_body = forward(x, last_linear=False, layers=self.cnn_layers)
        batch_shape = h_body.shape[:-3]
        h_body = h_body.reshape(batch_shape + (-1,))
        out = forward(h_body, last_linear=True, layers=self.mlp_layers)
        return out