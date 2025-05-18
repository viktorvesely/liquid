from __future__ import annotations

from typing import Self
import numpy as np
import torch
import torch.nn as nn

from .citizen import Citizen



def get_sequential(layers: list[int]) -> nn.Sequential:

    arch = []
    for src, tar in zip(layers[:-1], layers[1:], strict=True):
        arch.append(nn.Conv2d(src, tar, kernel_size=3, padding=1))
        arch.append(nn.BatchNorm2d(tar))
        arch.append(nn.LeakyReLU())
        arch.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*arch)


def monotonically_increasing_cnn(in_channels: int, out_channels: int, depth: int, addition: int = 1):
    layers = np.linspace(in_channels, out_channels, num=(depth + addition), endpoint=True).round().astype(int)
    return get_sequential(layers)


class DelegatingVisionCitizen(Citizen):

    def __init__(
        self,
        in_channels: int,
        n_citizens: int,
        out_channels: int,
        layers_body: int,
        layers_y: int,
        layers_d: int
    ):

        super().__init__()

        self.in_channels = in_channels
        self.n_citizens= n_citizens
        self.n_output = out_channels
        self.layers_body = layers_body
        self.layers_y = layers_y
        self.layers_d = layers_d

        max_depth = layers_body + max(layers_y, layers_d)
        layers = np.linspace(in_channels, out_channels, num=(max_depth + 1), endpoint=True).round().astype(int)
        body_out_channels = layers[layers_body]

        self.body = monotonically_increasing_cnn(
            in_channels,
            body_out_channels,
            depth=layers_body
        )

        self.y_head = monotonically_increasing_cnn(
            body_out_channels,
            out_channels,
            depth=layers_y
        )

        self.d_head = monotonically_increasing_cnn(
            body_out_channels,
            out_channels,
            depth=layers_y
        )


    def forward(self, x: torch.Tensor):

        b = self.body(x)
        c = self.y_head(b)
        d = self.d_head(b)

        return c, d


    def get_constructor(self) -> dict:

        return {
            "in_channels":  self.in_channels,
            "n_citizens":  self.n_citizens,
            "out_channels":  self.out_channels,
            "layers_body":  self.layers_body,
            "layers_y":  self.layers_y,
            "layers_d":  self.layers_d
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance


class VisionCitizen(Citizen):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int
    ):

        super().__init__()

        self.in_channels = in_channels
        self.n_output = out_channels
        self.layers = layers

        self.layers = monotonically_increasing_cnn(
            in_channels,
            out_channels,
            depth=layers
        )



    def forward(self, x: torch.Tensor):
        y = self.layers(x)
        return y

    def get_constructor(self) -> dict:

        return {
            "in_channels":  self.in_channels,
            "out_channels":  self.out_channels,
            "layers":  self.layers
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance


class FinalGlobalHead(nn.Module):

    def __init__(
        self,
        n_output: int,
    ):

        super().__init__()

        self.n_output = n_output

        self.out = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.Linear(1, n_output)
        )


    def forward(self, x: torch.Tensor):
        y = self.layers(x)
        return y

    def get_constructor(self) -> dict:
        return {
            "n_output":  self.n_output
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance