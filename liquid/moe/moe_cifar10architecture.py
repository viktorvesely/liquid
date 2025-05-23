from typing import Self
import numpy as np
import torch
import torch.nn as nn

from .moe_layer import MoELayer
from ..citizens.vision_citizen import VisionCitizen, FinalGlobalHead, VisionRouter


def moe_block(
        layers: int,
        layers_router: int,
        n_citizens: int,
        in_channels: int,
        out_channels: int,
        moe_kwargs: dict
):
    moe = MoELayer(
        experts = [
            VisionCitizen(in_channels, out_channels, layers)
            for _ in range(n_citizens)
        ],
        router=VisionRouter(
            in_channels=in_channels,
            out_channels=out_channels,
            n_citizens=n_citizens,
            layers=layers_router
        ),
        **moe_kwargs
    )

    return moe

class MoeCifar10(nn.Module):

    def __init__(
            self,
            in_channels: int,
            last_channels: int,
            n_citizens: int,
            n_moe_blocks: int,
            layers_in_moe_blocks: int,
            layers_router: int,
            n_output: int,
            moe_kwargs: dict | None = None,
            head_kwargs: dict | None = None
        ):
        super().__init__()

        if moe_kwargs is None:
            moe_kwargs = dict()

        if head_kwargs is None:
            head_kwargs = dict()

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens
        self.layers_in_moe_blocks = layers_in_moe_blocks
        self.n_moe_blocks = n_moe_blocks
        self.layers_router = layers_router
        self.n_output = n_output

        channels = np.linspace(in_channels, last_channels, num=(n_moe_blocks + 1), endpoint=True).round().astype(int)

        self.expert_blocks = nn.Sequential(
            *[
                moe_block(
                   layers=layers_in_moe_blocks,
                   layers_router=layers_router,
                   n_citizens=n_citizens,
                   in_channels=prev,
                   out_channels=fol,
                   moe_kwargs=moe_kwargs
                )
            for prev, fol in zip(channels[:-1], channels[1:], strict=True)]
        )

        self.output_head = FinalGlobalHead(
            in_channels=last_channels,
            n_output=n_output,
            **head_kwargs
        )

    def get_moe_layers(self) -> list[MoELayer]:
        return self.expert_blocks

    def forward(self, x: torch.Tensor):
        h = self.expert_blocks(x)
        return self.output_head(h)

    def get_constructor(self) -> dict:

        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "layers_in_moe_blocks": self.layers_in_moe_blocks,
            "n_moe_blocks": self.n_moe_blocks,
            "layers_router": self.layers_router,
            "n_output": self.n_output,
            "output_head": self.output_head.get_constructor(),
            "expert_blocks": [moe.get_constructor() for moe in self.expert_blocks]
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        output_head = constructor.pop("output_head")
        expert_blocks = constructor.pop("expert_blocks")

        instance = cls(**constructor)

        instance.output_head = FinalGlobalHead.apply_constructor(output_head)
        instance.expert_blocks = [MoELayer.apply_constructor(moe) for moe in expert_blocks]

        return instance



