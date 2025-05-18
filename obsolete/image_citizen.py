from __future__ import annotations

import copy
import torch
import torch.nn as nn

class Citizen(nn.Module):


    def __init__(self, channels: tuple[int, ...] | None = None):
        super().__init__()

        if channels is None:
            channels = (10, 20, 30, 40)

        self.channels = channels

        all_channels = [1] + list(channels)
        body = []
        for src, tar in zip(all_channels[:-1], all_channels[1:], strict=True):
            body.append(nn.Conv2d(src, tar, kernel_size=3, padding=1, stride=2))
            body.append(nn.LeakyReLU())

        body.append(nn.Flatten())
        self.body = nn.Sequential(*body)

        self.n_body_f = channels[-1] * 4
        n_body_f = self.n_body_f

        self.class_head = nn.Sequential(
            nn.Linear(n_body_f, self.mid(n_body_f, 10)),
            nn.LeakyReLU(),
            nn.Linear(self.mid(n_body_f, 10), 10),
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
        new_citizen = Citizen(self.channels).to("cuda")
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
        new_citizen = Citizen(self.channels).to("cuda")
        new_citizen.body = copy.deepcopy(self.body)
        new_citizen.class_head = copy.deepcopy(self.class_head)
        return new_citizen
