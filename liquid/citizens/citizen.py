
from typing import Self
import torch
import torch.nn as nn

class Citizen(nn.Module):

    def get_constructor(self) -> dict:
        raise NotImplementedError("get_constructor")

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        raise NotImplementedError("apply_constructor")

    @classmethod
    def name(cls) -> str:
        return cls.__name__



def get_sequential(layers: list[int], last_linear: bool = False, dropout: float = 0.0) -> nn.Sequential:
    # last_linear here means last layer

    if len(layers) < 2:
        return nn.Identity()

    arch = []
    n_pairs = len(layers) - 1
    is_dropout = dropout > 0

    for i, (src, tar) in enumerate(zip(layers[:-1], layers[1:], strict=True)):
        arch.append(nn.Linear(src, tar))
        is_last = (i == n_pairs - 1)

        if is_last and last_linear:
            continue

        arch.append(nn.LeakyReLU())

        if is_dropout:
            arch.append(nn.Dropout())

    return nn.Sequential(*arch)

class CitizenFC(Citizen):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        layers: int,
        width: int,
        last_linear: bool,
        dropout: float
    ):

        super().__init__()

        self.n_input= n_input
        self.n_output= n_output
        self.layers = layers
        self.width = width
        self.last_linear = last_linear
        self.dropout = dropout

        network = [n_input] + [width] * layers + [n_output]
        self.network = get_sequential(network, last_linear=last_linear, dropout=dropout)

    def forward(self, x: torch.Tensor):
        return self.network(x)


    def get_constructor(self) -> dict:
        return {
            "n_input":  self.n_input,
            "n_output":  self.n_output,
            "layers":  self.layers,
            "width":  self.width,
            "last_linear": self.last_linear,
            "dropout": self.dropout
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance

class RouterFC(Citizen):

    def __init__(
        self,
        n_input: int,
        n_citizens: int,
        layers: int,
        width: int,
        dropout: float
    ):

        super().__init__()

        self.n_input= n_input
        self.n_citizens= n_citizens
        self.layers = layers
        self.width = width
        self.dropout = dropout

        network = [n_input] + [width] * layers + [n_citizens]
        self.network = get_sequential(network, last_linear=True, dropout=dropout)

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def get_constructor(self) -> dict:
        return {
            "n_input":  self.n_input,
            "n_citizens":  self.n_citizens,
            "layers":  self.layers,
            "width":  self.width,
            "dropout": self.dropout
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance


