from __future__ import annotations

from typing import Self
import torch
import torch.nn as nn

from .citizen import Citizen



def get_sequential(layers: list[int], last_linear_true: bool = True) -> nn.Sequential:

    arch = []
    for src, tar in zip(layers[:-1], layers[1:], strict=True):
        arch.append(nn.Linear(src, tar))
        arch.append(nn.LeakyReLU())

    if last_linear_true:
        arch.pop()

    return nn.Sequential(*arch)



class DelegatingFC(Citizen):

    def __init__(
        self,
        n_input: int,
        n_citizens: int,
        n_output: int,
        layers_body: int,
        layers_y: int,
        layers_d: int,
        width: int
    ):

        super().__init__()

        self.n_input= n_input
        self.n_citizens= n_citizens
        self.n_output= n_output
        self.layers_body= layers_body
        self.layers_y= layers_y
        self.layers_d= layers_d
        self.width= width

        body = [n_input] + [width] * layers_body
        self.body = get_sequential(body, last_linear_true=False)

        y = [width] + [width] * layers_y + [n_output]
        self.y_head = get_sequential(y)

        d = [width] + [width] * layers_d + [n_citizens]
        self.d_head = get_sequential(d)
        self.d_head.append(nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor):

        b = self.body(x)
        c = self.y_head(b)
        d = self.d_head(b)

        return c, d


    def get_constructor(self) -> dict:

        return {
            "n_input":  self.n_input,
            "n_citizens":  self.n_citizens,
            "n_output":  self.n_output,
            "layers_body":  self.layers_body,
            "layers_y":  self.layers_y,
            "layers_d":  self.layers_d,
            "width":  self.width
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance