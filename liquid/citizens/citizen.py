
from typing import Self
import torch
import torch.nn as nn

class Citizen(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_constructor(self) -> dict:
        raise NotImplementedError("get_constructor")

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        raise NotImplementedError("apply_constructor")

    @classmethod
    def name(cls) -> str:
        return cls.__name__


def get_sequential(layers: list[int], last_linear_true: bool = True) -> nn.Sequential:

    arch = []
    for src, tar in zip(layers[:-1], layers[1:], strict=True):
        arch.append(nn.Linear(src, tar))
        arch.append(nn.LeakyReLU())

    if last_linear_true:
        arch.pop()

    return nn.Sequential(*arch)

class CitizenFC(Citizen):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        layers: int,
        width: int,
        last_linear: bool
    ):

        super().__init__()

        self.n_input= n_input
        self.n_output= n_output
        self.layers = layers
        self.width = width
        self.last_linear = last_linear

        network = [n_input] + [width] * layers + [n_output]
        self.network = get_sequential(network, last_linear_true=last_linear)

    def forward(self, x: torch.Tensor):
        return self.network(x)


    def get_constructor(self) -> dict:
        return {
            "n_input":  self.n_input,
            "n_output":  self.n_output,
            "layers":  self.layers,
            "width":  self.width,
            "last_linear": self.last_linear
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
    ):

        super().__init__()

        self.n_input= n_input
        self.n_citizens= n_citizens
        self.layers = layers
        self.width = width

        network = [n_input] + [width] * layers + [n_citizens]
        self.network = get_sequential(network, last_linear_true=True)

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def get_constructor(self) -> dict:
        return {
            "n_input":  self.n_input,
            "n_citizens":  self.n_citizens,
            "layers":  self.layers,
            "width":  self.width
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance


