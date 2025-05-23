from typing import Self
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .le_layer import LiquidEnsembleLayer
from ..citizens.delegating_citizen import DelegatingFC

class LongRegression(nn.Module):

    def __init__(
            self,
            n_input: int,
            n_citizens: int,
            layers_body: int,
            width_body: int,
            layers_y: int,
            width_y: int,
            layers_d: int,
            width_d: int,
            n_output: int,
            le_kwargs: dict | None = None,
            head_kwargs: dict | None = None
        ):
        super().__init__()

        if le_kwargs is None:
            le_kwargs = dict()

        if head_kwargs is None:
            head_kwargs = dict()

        self.n_input = n_input
        self.n_citizens = n_citizens
        self.layers_body = layers_body
        self.layers_y = layers_y
        self.layers_d = layers_d
        self.n_output = n_output
        self.width_body = width_body
        self.width_y = width_y
        self.width_d = width_d

        self.le_layer = LiquidEnsembleLayer(
            [
                DelegatingFC(
                    n_input=n_input,
                    n_citizens=n_citizens,
                    layers_body=layers_body,
                    layers_y=layers_y,
                    layers_d=layers_d,
                    n_output=n_output,
                    width_body = width_body,
                    width_y = width_y,
                    width_d = width_d,
                    last_linear=True
                )
            for _ in range(n_citizens) ],
            **le_kwargs
        )


    def forward(self, x: torch.Tensor):
        return self.le_layer(x)

    def get_constructor(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_citizens": self.n_citizens,
            "layers_body": self.layers_body,
            "layers_y": self.layers_y,
            "layers_d": self.layers_d,
            "n_output": self.n_output,
            "le_layer": self.le_layer.get_constructor(),
            "width_body" : self.width_body,
            "width_y" : self.width_y,
            "width_d" : self.width_d,
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        le_layer = constructor.pop("le_layer")

        instance = cls(**constructor)

        instance.le_layer = LiquidEnsembleLayer.apply_constructor(le_layer)

        return instance

    def get_le_layers(self) -> list[LiquidEnsembleLayer]:
        return [self.le_layer]


