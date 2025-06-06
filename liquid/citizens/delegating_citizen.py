from __future__ import annotations

from typing import Self
import torch
import torch.nn as nn

from .citizen import Citizen, get_sequential


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
        dropout_body: float,
        dropout_y: float,
        dropout_d: float,
        last_linear: bool = False,
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
        self.dropout_body = dropout_body
        self.dropout_y = dropout_y
        self.dropout_d = dropout_d

        body = [n_input] + [width_body] * layers_body
        self.body = get_sequential(body, dropout=dropout_body)

        y = [body[-1]] + [width_y] * layers_y + [n_output]
        self.y_head = get_sequential(y, last_linear=last_linear, dropout=dropout_y)

        d = [body[-1]] + [width_d] * layers_d + [n_citizens]
        self.d_head = get_sequential(d, last_linear=True, dropout=dropout_d)
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
            "dropout_body": self.dropout_body,
            "dropout_y": self.dropout_y,
            "dropout_d": self.dropout_d,
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance