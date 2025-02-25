from __future__ import annotations

import torch
import torch.nn as nn

from council import Council
from citizen import Citizen

class MajorityCouncil(Council):


    def __init__(self, n_citizens: int, citizens: nn.ModuleList | None = None):

        if citizens is None:
            citizens = [Citizen(n_citizens).to("cuda") for _ in range(n_citizens)]
            citizens = nn.ModuleList(citizens)

        super().__init__(n_citizens, citizens)

    def forward(self, x: torch.Tensor):

        classifications = []

        for citizen in self.citizens:
            c = citizen(x)
            classifications.append(c)

        classifications = torch.cat([torch.unsqueeze(c, 0) for c in classifications], dim=0)

        # (n, batch, out)
        return classifications


    def loss(self, y: torch.Tensor, classifications: torch.Tensor):
        predict_losses = self.predict_loss(y, classifications)

        final_loss = 0
        for i_citizen, citizen_predict_loss in enumerate(predict_losses):
            # citizen_predict_loss (n_batch,)
            # power (n_batch, n)
            final_loss += citizen_predict_loss

        return final_loss.mean()


    def vote(self, classifications: torch.Tensor):
        # classifications (n, n_batch, out)

        labels_hat_per_voter = torch.argmax(classifications, dim=2)
        labels_hat, _ = torch.mode(labels_hat_per_voter, dim=0)

        return labels_hat

    def make_dictator(self) -> MajorityCouncil:
        dictator = self.citizens[0].spawn_clone()
        dictator = nn.ModuleList([dictator])
        return MajorityCouncil(self.n_citizens, dictator)

