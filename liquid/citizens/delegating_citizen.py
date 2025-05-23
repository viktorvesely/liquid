from __future__ import annotations

from typing import Self
import torch
import torch.nn as nn

from .citizen import Citizen



def get_sequential(layers: list[int], last_linear: bool = False) -> nn.Sequential:

    if len(layers) < 2:
        return nn.Identity()

    arch = []
    for src, tar in zip(layers[:-1], layers[1:], strict=True):
        arch.append(nn.Linear(src, tar))
        arch.append(nn.GELU())

    if last_linear:
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
        width_body: int,
        width_y: int,
        width_d: int,
        last_linear: bool = False
    ):

        super().__init__()

        self.n_input = n_input
        self.n_citizens = n_citizens
        self.n_output = n_output
        self.layers_body = layers_body
        self.layers_y = layers_y
        self.layers_d = layers_d
        self.width_body = width_body
        self.width_y = width_y
        self.width_d = width_d
        self.last_linear = last_linear

        body = [n_input] + [width_body] * layers_body
        self.body = get_sequential(body)

        y = [body[-1]] + [width_y] * layers_y + [n_output]
        self.y_head = get_sequential(y, last_linear=last_linear)

        d = [body[-1]] + [width_d] * layers_d + [n_citizens]
        self.d_head = get_sequential(d, last_linear=True)
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
            "last_linear": self.last_linear,
            "width_body" : self.width_body,
            "width_y" : self.width_y,
            "width_d" : self.width_d,
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance