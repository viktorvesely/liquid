from __future__ import annotations

import copy
import torch
import torch.nn as nn
import numpy as np

def total_params(b, c, d, width):
    one = width ** 2
    return b* one, c * one, d * one


def build_layers(layers: int, body: float, classification: float, delegation: float, width: int = 50) -> tuple[int, list[int], list[int], list[int]]:

    weights = np.array([body, classification, delegation])
    weights = weights / weights.sum()

    l = np.round(layers * weights).astype(int)

    def to_layers(l: int) -> list:
        return [width for _ in range(l)]

    lb, lc, ld = l

    lb = max(lb, 1)
    lc = max(lc, 1)
    ld = max(ld, 1)

    b = to_layers(lb)
    c = to_layers(lc)
    d = to_layers(ld)

    return layers - l.sum(), b, c, d,

def get_sequential(layers: list[int]) -> list[nn.Module]:
    arch = []
    for src, tar in zip(layers[:-1], layers[1:], strict=True):
        arch.append(nn.Linear(src, tar))
        arch.append(nn.LeakyReLU())

    return arch

class Citizen(nn.Module):

    def __init__(self, body_layers: list[int], class_layers: list[int]):
        super().__init__()

        body_layers = [2] + body_layers
        self.body = nn.Sequential(*get_sequential(body_layers))

        class_layers = class_layers + [3]
        self.class_head = get_sequential(class_layers)
        self.class_head.pop()
        self.class_head.append(nn.Softmax(dim=1))
        self.class_head = nn.Sequential(*self.class_head)


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

    def __init__(self, n_citizens: int, layers: int = 6, ratios: tuple[int, int, int] = (2, 1, 1), layer_width: int = 30):
        _, body_l, class_l, del_l = build_layers(layers, *ratios, width=layer_width)

        # print(layers, len(body_l), len(class_l), len(del_l))

        super().__init__(body_l, class_l)

        del_l = del_l + [n_citizens]
        self.delegate_head = get_sequential(del_l)
        self.delegate_head.pop()
        self.delegate_head.append(nn.Softmax(dim=1))
        self.delegate_head = nn.Sequential(*self.delegate_head)


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
