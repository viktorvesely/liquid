from __future__ import annotations

from typing import Self
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .citizen import Citizen


class CNN2FC(Citizen):

    def __init__(
        self,
        in_channels: int,
        n_square_output: int = 2
    ):

        super().__init__()

        self.in_channels = in_channels
        self.n_square_output = n_square_output

        self.out = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=n_square_output),
            nn.Flatten()
        )


    @property
    def out_size(self) -> float:
        return (self.n_square_output ** 2) * self.in_channels

    def forward(self, x: torch.Tensor):
        y = self.out(x)
        return y

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "n_square_output": self.n_square_output
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance



class FinalGlobalHead(Citizen):

    def __init__(
        self,
        in_channels: int,
        n_output: int,
        size: int = 2
    ):

        super().__init__()

        self.in_channels = in_channels
        self.n_output = n_output
        self.size = size

        self.out = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=size),
            nn.Flatten(),
            nn.Linear(in_channels * (size ** 2), n_output)
        )

    def forward(self, x: torch.Tensor):
        y = self.out(x)
        return y

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "n_output":  self.n_output,
            "size":  self.size
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance


def get_sequential(layers: list[int], max_pool_every: int = 1) -> nn.Sequential:

    arch = []
    for i, (src, tar) in enumerate(zip(layers[:-1], layers[1:], strict=True)):

        arch.append(nn.Conv2d(src, tar, kernel_size=3, padding=1))
        arch.append(nn.BatchNorm2d(tar))
        arch.append(nn.LeakyReLU())

        if ((i + 1) % max_pool_every) == 0:
            arch.append(nn.MaxPool2d(kernel_size=2, stride=2))


    return nn.Sequential(*arch)


def monotonically_increasing_cnn(
        in_channels: int,
        out_channels: int,
        depth: int,
        addition: int = 1,
        max_pool_every: int = 1
    ):
    layers = np.linspace(in_channels, out_channels, num=(depth + addition), endpoint=True).round().astype(int)
    return get_sequential(layers, max_pool_every=max_pool_every)


class DelegatingVisionCitizen(Citizen):

    def __init__(
        self,
        in_channels: int,
        n_citizens: int,
        out_channels: int,
        layers_body: int,
        layers_y: int,
        layers_d: int,
        max_pool_every: int = 2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.n_citizens= n_citizens
        self.out_channels = out_channels
        self.layers_body = layers_body
        self.layers_y = layers_y
        self.layers_d = layers_d
        self.max_pool_every = max_pool_every

        max_depth = layers_body + max(layers_y, layers_d)
        layers = np.linspace(in_channels, out_channels, num=(max_depth + 1), endpoint=True).round().astype(int)
        body_out_channels = layers[layers_body]

        self.body = monotonically_increasing_cnn(
            in_channels,
            body_out_channels,
            depth=layers_body,
            max_pool_every=max_pool_every
        )

        self.y_head = monotonically_increasing_cnn(
            body_out_channels,
            out_channels,
            depth=layers_y,
            max_pool_every=max_pool_every
        )

        self.d_head = monotonically_increasing_cnn(
            body_out_channels,
            out_channels,
            depth=layers_d,
            max_pool_every=max_pool_every
        )

        self.global_d = FinalGlobalHead(out_channels, n_citizens)


    def forward(self, x: torch.Tensor):

        b = self.body(x)
        y = self.y_head(b)

        d_h = self.d_head(b)
        d = F.softmax(self.global_d(d_h), dim=1)

        return y, d


    def get_constructor(self) -> dict:

        return {
            "in_channels":  self.in_channels,
            "n_citizens":  self.n_citizens,
            "out_channels":  self.out_channels,
            "layers_body":  self.layers_body,
            "layers_y":  self.layers_y,
            "layers_d":  self.layers_d,
            "global_d": self.global_d.get_constructor(),
            "max_pool_every": self.max_pool_every
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        global_d_constructor = constructor.pop("global_d")
        instance = cls(**constructor)
        instance.global_d = FinalGlobalHead(**global_d_constructor)

        return instance

class VisionCitizen(Citizen):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        max_pool_every: int = 2
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool_every = max_pool_every
        self.layers = layers

        self.layers = monotonically_increasing_cnn(
            in_channels,
            out_channels,
            depth=layers,
            max_pool_every=max_pool_every
        )



    def forward(self, x: torch.Tensor):
        y = self.layers(x)
        return y

    def get_constructor(self) -> dict:

        return {
            "in_channels":  self.in_channels,
            "out_channels":  self.out_channels,
            "layers":  self.layers,
            "max_pool_every": self.max_pool_every
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance

class VisionRouter(Citizen):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_citizens: int,
            layers: int,
            max_pool_every: int = 2
        ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.max_pool_every = max_pool_every
        self.n_citizens = n_citizens

        self.layers = monotonically_increasing_cnn(
            in_channels,
            out_channels,
            depth=layers,
            max_pool_every=max_pool_every
        )

        self.out = FinalGlobalHead(out_channels, n_citizens)


    def forward(self, x: torch.Tensor):
        h = self.layers(x)
        return self.out(h)

    def get_constructor(self) -> dict:

        return {
            "in_channels":  self.in_channels,
            "out_channels":  self.out_channels,
            "layers":  self.layers,
            "max_pool_every": self.max_pool_every,
            "n_citizens": self.n_citizens
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        instance = cls(**constructor)
        return instance