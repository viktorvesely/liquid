# moecifar.py

from typing import Self
import numpy as np
import torch
import torch.nn as nn

from .moe_layer import MoELayer
from ..citizens.vision_citizen import VisionCitizen, VisionRouter, CNN2FC
from ..citizens.citizen import RouterFC, CitizenFC


class MoeLongCifar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        last_channels: int,
        n_citizens: int,
        n_output: int,
        moe_kwargs: dict | None = None,
        moe_cnn_kwargs: dict | None = None,
        moe_fc_kwargs: dict | None = None,
        router_cnn_kwargs: dict | None = None,
        router_fc_kwargs: dict | None = None,
    ):
        super().__init__()

        if moe_kwargs is None:
            moe_kwargs = {}
        if moe_cnn_kwargs is None:
            moe_cnn_kwargs = {}
        if moe_fc_kwargs is None:
            moe_fc_kwargs = {}
        if router_cnn_kwargs is None:
            router_cnn_kwargs = {}
        if router_fc_kwargs is None:
            router_fc_kwargs = {}

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens
        self.n_output = n_output

        self.moe_cnn_layers = MoELayer(
            experts=[
                VisionCitizen(
                    in_channels=in_channels,
                    out_channels=last_channels,
                    **moe_cnn_kwargs,
                )
                for _ in range(n_citizens)
            ],
            router=VisionRouter(
                in_channels=in_channels,
                out_channels=last_channels,
                n_citizens=n_citizens,
                **router_cnn_kwargs,
            ),
            **moe_kwargs,
        )

        self.flattenner = CNN2FC(in_channels=last_channels, n_square_output=2)
        n_cnn_output = self.flattenner.out_size

        self.moe_fc_layers = MoELayer(
            experts=[
                CitizenFC(
                    n_input=n_cnn_output,
                    n_output=n_output,
                    **moe_fc_kwargs,
                )
                for _ in range(n_citizens)
            ],
            router=RouterFC(
                n_input=n_cnn_output,
                n_citizens=n_citizens,
                **router_fc_kwargs,
            ),
            **moe_kwargs,
        )

    def forward(self, x: torch.Tensor):
        h = self.moe_cnn_layers(x)
        h = self.flattenner(h)
        return self.moe_fc_layers(h)

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "n_output": self.n_output,
            "moe_cnn_layers": self.moe_cnn_layers.get_constructor(),
            "moe_fc_layers": self.moe_fc_layers.get_constructor(),
            "flattenner": self.flattenner.get_constructor(),
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        moe_cnn_layers = constructor.pop("moe_cnn_layers")
        moe_fc_layers = constructor.pop("moe_fc_layers")
        flattenner = constructor.pop("flattenner")

        instance = cls(**constructor)
        instance.moe_cnn_layers = MoELayer.apply_constructor(moe_cnn_layers)
        instance.moe_fc_layers = MoELayer.apply_constructor(moe_fc_layers)
        instance.flattenner = CNN2FC.apply_constructor(flattenner)
        return instance

    def get_moe_layers(self) -> list[MoELayer]:
        return [self.moe_cnn_layers, self.moe_fc_layers]


def moe_cnn_block(
    in_channels: int,
    out_channels: int,
    n_citizens: int,
    block_kwargs: dict,
    router_kwargs: dict,
    moe_kwargs: dict,
):
    return MoELayer(
        experts=[
            VisionCitizen(
                in_channels=in_channels,
                out_channels=out_channels,
                layers=1,
                **block_kwargs,
            )
            for _ in range(n_citizens)
        ],
        router=VisionRouter(
            in_channels=in_channels,
            out_channels=out_channels,
            n_citizens=n_citizens,
            layers=1,
            **router_kwargs,
        ),
        **moe_kwargs,
    )


def moe_fc_block(
    n_input: int,
    n_output: int,
    n_citizens: int,
    block_kwargs: dict,
    router_kwargs: dict,
    moe_kwargs: dict,
):
    return MoELayer(
        experts=[
            CitizenFC(
                n_input=n_input,
                n_output=n_output,
                layers=1,
                **block_kwargs,
            )
            for _ in range(n_citizens)
        ],
        router=RouterFC(
            n_input=n_input,
            n_citizens=n_citizens,
            layers=1,
            **router_kwargs,
        ),
        **moe_kwargs,
    )


class MoeBlockCifar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        last_channels: int,
        n_output: int,
        n_citizens: int,
        n_cnn_moe_blocks: int,
        n_fc_moe_blocks: int,
        moe_kwargs: dict | None = None,
        moe_cnn_kwargs: dict | None = None,
        moe_fc_kwargs: dict | None = None,
        router_cnn_kwargs: dict | None = None,
        router_fc_kwargs: dict | None = None,
    ):
        super().__init__()

        if moe_kwargs is None:
            moe_kwargs = {}
        if moe_cnn_kwargs is None:
            moe_cnn_kwargs = {}
        if moe_fc_kwargs is None:
            moe_fc_kwargs = {}
        if router_cnn_kwargs is None:
            router_cnn_kwargs = {}
        if router_fc_kwargs is None:
            router_fc_kwargs = {}

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens
        self.n_cnn_moe_blocks = n_cnn_moe_blocks
        self.n_fc_moe_blocks = n_fc_moe_blocks
        self.n_output = n_output

        channels = np.linspace(
            in_channels, last_channels, num=(n_cnn_moe_blocks + 1), endpoint=True
        ).round().astype(int)

        self.moe_cnn_blocks = nn.Sequential(
            *[
                moe_cnn_block(
                    in_channels=prev,
                    out_channels=fol,
                    n_citizens=n_citizens,
                    block_kwargs=moe_cnn_kwargs,
                    router_kwargs=router_cnn_kwargs,
                    moe_kwargs=moe_kwargs,
                )
                for prev, fol in zip(channels[:-1], channels[1:], strict=True)
            ]
        )

        self.flattenner = CNN2FC(channels[-1])
        fc_input = self.flattenner.out_size

        layers = np.linspace(
            fc_input, n_output, num=(n_fc_moe_blocks + 1), endpoint=True
        ).round().astype(int)

        self.moe_fc_blocks = nn.Sequential(
            *[
                moe_fc_block(
                    n_input=prev,
                    n_output=fol,
                    n_citizens=n_citizens,
                    block_kwargs=moe_fc_kwargs,
                    router_kwargs=router_fc_kwargs,
                    moe_kwargs=moe_kwargs,
                )
                for prev, fol in zip(layers[:-1], layers[1:], strict=True)
            ]
        )

    def forward(self, x: torch.Tensor):
        h = self.moe_cnn_blocks(x)
        h = self.flattenner(h)
        return self.moe_fc_blocks(h)

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "n_cnn_moe_blocks": self.n_cnn_moe_blocks,
            "n_fc_moe_blocks": self.n_fc_moe_blocks,
            "n_output": self.n_output,
            "moe_cnn_blocks": [blk.get_constructor() for blk in self.moe_cnn_blocks],
            "moe_fc_blocks": [blk.get_constructor() for blk in self.moe_fc_blocks],
            "flattenner": self.flattenner.get_constructor(),
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        flattenner = constructor.pop("flattenner")
        moe_cnn_blocks = constructor.pop("moe_cnn_blocks")
        moe_fc_blocks = constructor.pop("moe_fc_blocks")

        instance = cls(**constructor)
        instance.moe_cnn_blocks = nn.Sequential(
            *[MoELayer.apply_constructor(b) for b in moe_cnn_blocks]
        )
        instance.flattenner = CNN2FC.apply_constructor(flattenner)
        instance.moe_fc_blocks = nn.Sequential(
            *[MoELayer.apply_constructor(b) for b in moe_fc_blocks]
        )
        return instance

    def get_moe_layers(self) -> list[MoELayer]:
        return list(self.moe_cnn_blocks) + list(self.moe_fc_blocks)
