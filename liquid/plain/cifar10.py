# simplecifar.py

from typing import Self
import torch
import torch.nn as nn

from ..citizens.vision_citizen import VisionCitizen, CNN2FC
from ..citizens.citizen import CitizenFC


class SimpleCifar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        last_channels: int,
        n_output: int,
        n_cnn_layers: int,
        max_pool_every: int,
        n_fc_layers: int,
        cnn_kwargs: dict | None = None,
        fc_kwargs: dict | None = None,
    ):
        super().__init__()
        if cnn_kwargs is None:
            cnn_kwargs = {}
        if fc_kwargs is None:
            fc_kwargs = {}

        self.in_channels = in_channels
        self.last_channels = last_channels
        self.n_output = n_output
        self.n_cnn_layers = n_cnn_layers
        self.max_pool_every = max_pool_every
        self.n_fc_layers = n_fc_layers

        self.cnn = VisionCitizen(
            in_channels=in_channels,
            out_channels=last_channels,
            layers=n_cnn_layers,
            max_pool_every=max_pool_every,
            **cnn_kwargs,
        )

        self.flattenner = CNN2FC(in_channels=last_channels, n_square_output=2)
        n_cnn_output = self.flattenner.out_size

        self.fc = CitizenFC(
            n_input=n_cnn_output,
            n_output=n_output,
            layers=n_fc_layers,
            **fc_kwargs,
        )

    def forward(self, x: torch.Tensor):
        h = self.cnn(x)
        h = self.flattenner(h)
        return self.fc(h)

    def get_constructor(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "last_channels": self.last_channels,
            "n_output": self.n_output,
            "n_cnn_layers": self.n_cnn_layers,
            "max_pool_every": self.max_pool_every,
            "n_fc_layers": self.n_fc_layers,
            "cnn": self.cnn.get_constructor(),
            "fc": self.fc.get_constructor(),
            "flattenner": self.flattenner.get_constructor(),
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        cnn = constructor.pop("cnn")
        fc = constructor.pop("fc")
        flattenner = constructor.pop("flattenner")

        instance = cls(**constructor)
        instance.cnn = VisionCitizen.apply_constructor(cnn)
        instance.fc = CitizenFC.apply_constructor(fc)
        instance.flattenner = CNN2FC.apply_constructor(flattenner)
        return instance
