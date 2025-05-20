from typing import Self
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .le_layer import LiquidEnsembleLayer
from ..citizens.vision_citizen import DelegatingVisionCitizen, FinalGlobalHead

class LongCifar10(nn.Module):

    def __init__(
            self,
            in_channels: int,
            last_channels: int,
            n_citizens: int,
            layers_body: int,
            layers_y: int,
            layers_d: int,
            n_output: int,
            le_kwargs: dict | None = None,
            head_kwargs: dict | None = None
        ):
        super().__init__()

        if le_kwargs is None:
            le_kwargs = dict()

        if head_kwargs is None:
            head_kwargs = dict()

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens
        self.layers_body = layers_body
        self.layers_y = layers_y
        self.layers_d = layers_d
        self.n_output = n_output

        self.le_layer = LiquidEnsembleLayer(
            [
                DelegatingVisionCitizen(
                    in_channels=in_channels,
                    n_citizens=n_citizens,
                    out_channels=last_channels,
                    layers_body=layers_body,
                    layers_y=layers_y,
                    layers_d=layers_d,
                )
            for _ in range(n_citizens) ],
            **le_kwargs
        )

        self.output_head = FinalGlobalHead(
            in_channels=last_channels,
            n_output=n_output,
            **head_kwargs
        )


    def forward(self, x: torch.Tensor):
        h = self.le_layer(x)
        return self.output_head(h)

    def auxiliary_loss(
            self,
            model: nn.Module,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            yhat_batch: torch.Tensor
    ) -> torch.Tensor:
        return self.le_layer.distribution_specialization_loss()

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "layers_body": self.layers_body,
            "layers_y": self.layers_y,
            "layers_d": self.layers_d,
            "n_output": self.n_output,
            "le_layer": self.le_layer.get_constructor(),
            "output_head": self.output_head.get_constructor()
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        output_head = constructor.pop("output_head")
        le_layer = constructor.pop("le_layer")

        instance = cls(**constructor)

        instance.global_d = FinalGlobalHead.apply_constructor(output_head)
        instance.le_layer = LiquidEnsembleLayer.apply_constructor(le_layer)

        return instance

    def power_entropy(self):
        return self.le_layer.power_entropy()

    def speaker_entropy(self):
        return self.le_layer.speaker_entropy()

def le_block(
    in_channels: int,
    out_channels: int,
    n_citizens: int,
    layers_body: int,
    layers_y: int,
    layers_d: int,
    le_kwargs: dict | None = None
):
    return LiquidEnsembleLayer(
            [
                DelegatingVisionCitizen(
                    in_channels=in_channels,
                    n_citizens=n_citizens,
                    out_channels=out_channels,
                    layers_body=layers_body,
                    layers_y=layers_y,
                    layers_d=layers_d,
                )
            for _ in range(n_citizens) ],
            **le_kwargs
    )


class BlockCifar10(nn.Module):

    def __init__(
            self,
            in_channels: int,
            last_channels: int,
            n_citizens: int,
            n_le_blocks: int,
            layers_body: int,
            layers_y: int,
            layers_d: int,
            n_output: int,
            le_kwargs: dict | None = None,
            head_kwargs: dict | None = None
        ):
        super().__init__()

        if le_kwargs is None:
            le_kwargs = dict()

        if head_kwargs is None:
            head_kwargs = dict()

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_citizens = n_citizens
        self.layers_body = layers_body
        self.layers_y = layers_y
        self.layers_d = layers_d
        self.n_output = n_output

        channels = np.linspace(in_channels, last_channels, num=(n_le_blocks + 1), endpoint=True).round().astype(int)

        self.le_blocks = nn.Sequential(
            *[
                le_block(
                    in_channels=prev,
                    out_channels=fol,
                    n_citizens=n_citizens,
                    layers_body=layers_body,
                    layers_y=layers_y,
                    layers_d=layers_d,
                    le_kwargs=le_kwargs
                )
            for prev, fol in zip(channels[:-1], channels[1:], strict=True)]
        )

        self.output_head = FinalGlobalHead(
            in_channels=last_channels,
            n_output=n_output,
            **head_kwargs
        )


    def forward(self, x: torch.Tensor):
        h = self.le_blocks(x)
        return self.output_head(h)

    def auxiliary_loss(
            self,
            model: nn.Module,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            yhat_batch: torch.Tensor
    ) -> torch.Tensor:
        return self.le_layer.distribution_specialization_loss()

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_citizens": self.n_citizens,
            "layers_body": self.layers_body,
            "layers_y": self.layers_y,
            "layers_d": self.layers_d,
            "n_output": self.n_output,
            "le_layer": self.le_layer.get_constructor(),
            "output_head": self.output_head.get_constructor()
        }


    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        output_head = constructor.pop("output_head")
        le_layer = constructor.pop("le_layer")

        instance = cls(**constructor)

        instance.global_d = FinalGlobalHead.apply_constructor(output_head)
        instance.le_layer = LiquidEnsembleLayer.apply_constructor(le_layer)

        return instance

    def power_entropy(self):
        return torch.mean(torch.stack(tuple(le.power_entropy() for le in self.le_blocks)))

    def speaker_entropy(self):
        return torch.mean(torch.stack(tuple(le.speaker_entropy() for le in self.le_blocks)))

