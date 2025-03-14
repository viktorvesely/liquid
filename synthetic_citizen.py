from __future__ import annotations

import copy
import torch
import torch.nn as nn

class Citizen(nn.Module):

    def __init__(self, layers: tuple[int, ...] | None = None):
        super().__init__()

        if layers is None:
            layers = (30, 60, 120, 240, 480)

        self.layers = layers

        all_layers = [2] + list(layers)
        body = []
        for src, tar in zip(all_layers[:-1], all_layers[1:], strict=True):
            body.append(nn.Linear(src, tar))
            body.append(nn.LeakyReLU())

        body.append(nn.Flatten())
        self.body = nn.Sequential(*body)

        self.n_body_f = layers[-1]

        self.class_head = nn.Sequential(
            nn.Linear(self.n_body_f, self.mid(self.n_body_f, 3)),
            nn.LeakyReLU(),
            nn.Linear(self.mid(self.n_body_f, 3), 3),
            nn.Softmax(dim=1)
        )


    @staticmethod
    def mid(source, target):
        return int(round(source + (target - source) / 2))


    def forward(self, x: torch.Tensor):

        b = self.body(x)
        c = self.class_head(b)

        return c

    def spawn_clone(self) -> Citizen:
        new_citizen = Citizen(self.layers).to("cuda")
        new_citizen.body = copy.deepcopy(self.body)
        new_citizen.class_head = copy.deepcopy(self.class_head)
        return new_citizen




class DelegatingCitizen(Citizen):

    def __init__(self, n_citizens: int, channels: tuple[int, ...] | None = None):
        super().__init__(channels)

        n_body_f = self.n_body_f

        self.delegate_head = nn.Sequential(
            nn.Linear(n_body_f, self.mid(n_body_f, n_citizens)),
            nn.LeakyReLU(),
            nn.Linear(self.mid(n_body_f, n_citizens), n_citizens),
            nn.Softmax(dim=1)
        )


    def forward(self, x: torch.Tensor):

        b = self.body(x)
        c = self.class_head(b)
        d = self.delegate_head(b)

        return c, d

    def cut_delegation_head(self) -> Citizen:
        new_citizen = Citizen(self.layers).to("cuda")
        new_citizen.body = copy.deepcopy(self.body)
        new_citizen.class_head = copy.deepcopy(self.class_head)
        return new_citizen
