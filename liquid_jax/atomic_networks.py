import flax.linen as nn
import jax

def get_layers(neurons: tuple[int, ...]):

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
    
    def setup(self):
        self.body_layers = get_layers(self.body)

    def __call__(self, x: jax.Array):
        out = forward(x, last_linear=True, layers=self.body_layers)
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