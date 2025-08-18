from typing import Self
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .le_layer import LiquidEnsembleLayer
from ..citizens.vision_citizen import DelegatingVisionCitizen, CNN2FC
from ..citizens.delegating_citizen import DelegatingFC

class LeLongCifar(nn.Module):

    def __init__(
            self,
            in_channels: int,
            last_channels: int,
            n_citizens: int,
            n_output: int,
            le_kwargs: dict | None = None,
            le_cnn_kwargs: dict | None = None,
            le_fc_kwargs: dict | None = None
        ):
        super().__init__()

        if le_kwargs is None:
            le_kwargs = dict()

        if le_cnn_kwargs is None:
            le_cnn_kwargs = dict()

        if le_fc_kwargs is None:
            le_fc_kwargs = dict()

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens

        self.n_output = n_output

        self.le_cnn_layers = LiquidEnsembleLayer(
            [
                DelegatingVisionCitizen(
                    in_channels=in_channels,
                    n_citizens=n_citizens,
                    out_channels=last_channels,
                    **le_cnn_kwargs
                )
            for _ in range(n_citizens) ],
            **le_kwargs
        )


        self.flattenner = CNN2FC(
            in_channels=last_channels,
            n_square_output=2
        )

        n_cnn_output = self.flattenner.out_size

        self.le_fc_layers = LiquidEnsembleLayer(
            [
                DelegatingFC(
                    n_input=n_cnn_output,
                    n_citizens=n_citizens,
                    n_output=n_output,
                    last_linear=True,
                    **le_fc_kwargs
                )
            for _ in range(n_citizens) ],
            **le_kwargs
        )


    def forward(self, x: torch.Tensor):
        h = self.le_cnn_layers(x)
        h = self.flattenner(h)
        return self.le_fc_layers(h)

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "n_output": self.n_output,
            "le_cnn_layers": self.le_cnn_layers.get_constructor(),
            "le_fc_layers": self.le_fc_layers.get_constructor(),
            "flattenner": self.flattenner.get_constructor()
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        le_cnn_layers = constructor.pop("le_cnn_layers")
        le_fc_layers = constructor.pop("le_fc_layers")
        flattenner = constructor.pop("flattenner")

        instance = cls(**constructor)

        instance.le_cnn_layers = LiquidEnsembleLayer.apply_constructor(le_cnn_layers)
        instance.le_fc_layers = LiquidEnsembleLayer.apply_constructor(le_fc_layers)
        instance.flattenner = CNN2FC.apply_constructor(flattenner)

        return instance

    def get_le_layers(self) -> list[LiquidEnsembleLayer]:
        return [self.le_cnn_layers, self.le_fc_layers]

def le_cnn_block(
    in_channels: int,
    out_channels: int,
    n_citizens: int,
    block_kwargs: dict,
    le_kwargs: dict,
    max_pool: bool
):
    return LiquidEnsembleLayer(
            [
                DelegatingVisionCitizen(
                    in_channels=in_channels,
                    n_citizens=n_citizens,
                    out_channels=out_channels,
                    layers_body=1,
                    layers_d=1,
                    layers_y=1,
                    max_pool_every=2 if max_pool else 1000,
                    **block_kwargs
                )
            for _ in range(n_citizens) ],
            **le_kwargs
    )

def le_fc_block(
    n_input: int,
    n_output: int,
    n_citizens: int,
    block_kwargs: dict,
    le_kwargs: dict,
    linear_output: bool
):
    return LiquidEnsembleLayer(
            [
                DelegatingFC(
                    n_input=n_input,
                    n_citizens=n_citizens,
                    n_output=n_output,
                    layers_body=1,
                    layers_d=1,
                    layers_y=1,
                    last_linear=linear_output,
                    **block_kwargs
                )
            for _ in range(n_citizens) ],
            **le_kwargs
    )


class LeBlockCifar(nn.Module):

    def __init__(
            self,
            in_channels: int,
            last_channels: int,
            n_output: int,
            n_citizens: int,
            n_cnn_le_blocks: int,
            n_fc_le_blocks: int,
            max_pool_every: int,
            le_kwargs: dict | None = None,
            le_cnn_kwargs: dict | None = None,
            le_fc_kwargs: dict | None = None
        ):
        super().__init__()

        if le_kwargs is None:
            le_kwargs = dict()

        if le_cnn_kwargs is None:
            le_cnn_kwargs = dict()

        if le_fc_kwargs is None:
            le_fc_kwargs = dict()

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens
        self.n_cnn_le_blocks = n_cnn_le_blocks
        self.n_fc_le_blocks = n_fc_le_blocks
        self.n_output = n_output
        self.max_pool_every = max_pool_every

        channels = np.linspace(
            in_channels, last_channels, num=(n_cnn_le_blocks + 1), endpoint=True
        ).round().astype(int)

        self.le_cnn_blocks = nn.Sequential(
            *[
                le_cnn_block(
                    in_channels=prev,
                    out_channels=fol,
                    n_citizens=n_citizens,
                    le_kwargs=le_kwargs,
                    block_kwargs=le_cnn_kwargs,
                    max_pool=((i + 1) % max_pool_every) == 0
                )
            for i, (prev, fol) in enumerate(zip(channels[:-1], channels[1:], strict=True))]
        )

        self.flattenner = CNN2FC(
            channels[-1]
        )

        fc_input = self.flattenner.out_size

        layers = np.linspace(fc_input, n_output, num=(n_fc_le_blocks + 1), endpoint=True).round().astype(int)

        self.le_fc_blocks = nn.Sequential(
            *[
                le_fc_block(
                    n_input=prev,
                    n_output=fol,
                    n_citizens=n_citizens,
                    le_kwargs=le_kwargs,
                    block_kwargs=le_fc_kwargs,
                    linear_output=(i == (layers.size - 2)) # last layer
                )
            for i, (prev, fol) in enumerate(zip(layers[:-1], layers[1:], strict=True))]
        )

    def forward(self, x: torch.Tensor):
        h = self.le_cnn_blocks(x)
        h = self.flattenner(h)
        return self.le_fc_blocks(h)

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "n_output": self.n_output,
            "le_cnn_blocks": [block.get_constructor() for block in self.le_cnn_blocks],
            "le_fc_blocks": [block.get_constructor() for block in self.le_fc_blocks],
            "flattenner": self.flattenner.get_constructor()
        }


    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        flattener = constructor.pop("flattenner")
        le_cnn_blocks = constructor.pop("le_cnn_blocks")
        le_fc_blocks = constructor.pop("le_fc_blocks")

        instance = cls(**constructor)

        instance.le_cnn_blocks = nn.Sequential(*[
            DelegatingVisionCitizen.apply_constructor(block) for block in le_cnn_blocks
        ])
        instance.flattenner = CNN2FC.apply_constructor(flattener)
        instance.le_fc_blocks = nn.Sequential(*[
            DelegatingFC.apply_constructor(block) for block in le_fc_blocks
        ])

        return instance

    def get_le_layers(self) -> list[LiquidEnsembleLayer]:
        return list(self.le_cnn_blocks) + list(self.le_fc_blocks)
