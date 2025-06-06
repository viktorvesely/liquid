from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_layer import MoELayer
from ..citizens.citizen import CitizenFC, RouterFC

class LongRegression(nn.Module):

    def __init__(
            self,
            n_input: int,
            n_citizens: int,
            layers_body: int,
            width_body: int,
            layers_router: int,
            width_router: int,
            body_dropout: float,
            router_dropout: float,
            n_output: int,
            moe_kwargs: dict | None = None,
            head_kwargs: dict | None = None
        ):
        super().__init__()

        if moe_kwargs is None:
            moe_kwargs = dict()

        if head_kwargs is None:
            head_kwargs = dict()

        self.n_input = n_input
        self.n_citizens = n_citizens
        self.layers_body = layers_body
        self.layers_router = layers_router
        self.n_output = n_output
        self.width_body = width_body
        self.width_router = width_router
        self.body_dropout = body_dropout
        self.router_dropout = router_dropout

        self.moe_layer = MoELayer(
            [
                CitizenFC(
                    n_input=n_input,
                    layers=layers_body,
                    n_output=n_output,
                    width = width_body,
                    last_linear=True,
                    dropout=body_dropout
                )
            for _ in range(n_citizens) ],
            router=RouterFC(
                n_input=n_input,
                n_citizens=n_citizens,
                layers=layers_router,
                width=width_router,
                dropout=router_dropout
            ),
            **moe_kwargs
        )

    def get_moe_layers(self) -> list[MoELayer]:
        return [self.moe_layer]

    def forward(self, x: torch.Tensor):
        return self.moe_layer(x)

    def get_constructor(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_citizens": self.n_citizens,
            "layers_body": self.layers_body,
            "layers_router": self.layers_router,
            "n_output": self.n_output,
            "moe_layer": self.moe_layer.get_constructor(),
            "width_body" : self.width_body,
            "width_router" : self.width_router,
            "body_dropout": self.body_dropout,
            "router_dropout": self.router_dropout,
        }


    def calculate_confidence(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = self(x)
        return {
            "confidence_gate": self.moe_layer.confidence_gate_entropy(),
            "confidence_std": self.moe_layer.confidence_std()
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        moe_layer = constructor.pop("moe_layer")

        instance = cls(**constructor)

        instance.moe_layer = MoELayer.apply_constructor(moe_layer)

        return instance

    def power_entropy(self):
        return self.moe_layer.power_entropy()

    def speaker_entropy(self):
        return self.moe_layer.speaker_entropy()

