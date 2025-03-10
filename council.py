from typing import Self
import torch
import torch.nn as nn

from copy import deepcopy

class Council(nn.Module):

    n_digits: int = 10

    def __init__(self, n_citizens: int, citizens: nn.ModuleList):
        super().__init__()

        self.n_citizens = n_citizens
        self.citizens = citizens
        self.predict_criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward")

    def predict_loss(self, y: torch.Tensor, classifications: torch.Tensor):
        return [self.predict_criterion(classifications[i_voter, :, :], y) for i_voter in range(classifications.shape[0])]

    def loss(self, y: torch.Tensor, classifications: torch.Tensor):
        raise NotImplementedError("loss")

    def vote(self, classifications: torch.Tensor):
        # classifications (n, n_batch, out)
        # power (n_batch, n)
        # distributions (n_batch, out)
        raise NotImplementedError("vote")

    def name(self):
        raise NotImplementedError("name")

