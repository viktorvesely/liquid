import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from citizen import Citizen

class Council(nn.Module):

    n_digits: int = 10

    def __init__(self, n_citizens: int):
        super().__init__()

        self.n_citizens = n_citizens
        self.citizens = [Citizen(n_citizens).to("cuda") for _ in range(n_citizens)]
        self.citizens = nn.ModuleList(self.citizens)

        self.predict_criterion = nn.CrossEntropyLoss(reduction="none")


    def forward(self, x: torch.Tensor):

        classifications = []
        delegations = []

        for citizen in self.citizens:
            c, d = citizen(x)
            classifications.append(c)
            delegations.append(d)

        power, D = self.solve_delegation_iterative(delegations)

        return classifications, power, D

    def predict_loss(self, y: torch.Tensor, classifications: list[torch.Tensor]):
        return [self.predict_criterion(c, y) for c in classifications]


    def loss(self, y: torch.Tensor, classifications: list[torch.Tensor], power: torch.Tensor):
        predict_losses = self.predict_loss(y, classifications)

        final_loss = 0
        for i_citizen, citizen_predict_loss in enumerate(predict_losses):
            # citizen_predict_loss (n_batch,)
            # power (n_batch, n)
            final_loss += citizen_predict_loss * power[:, i_citizen]

        return final_loss.mean()


    def vote_distribution(self, classifications: list[torch.Tensor], power: torch.Tensor):
        # classifications (n, n_batch, digits)
        # power (n_batch, n)
        # distributions (n_batch, digits)

        classifications = torch.cat([torch.unsqueeze(c, 0) for c in classifications], dim=0)

        bs = power.shape[0]

        # Just in case normalize power
        sums = power.sum(1, keepdim=True)
        if (~torch.isclose(sums, torch.tensor(self.n_citizens, device=power.device, dtype=power.dtype), atol=1e-2)).any():
            print("Warning, power sum is not close to n_citizens")

        power = power / sums

        distributions = torch.zeros((bs, self.n_digits), dtype=power.dtype, device=power.device)

        for i_batch in range(bs):
            # (n, digits)
            c = classifications[:, i_batch, :]

            # (n, 1)
            p = torch.unsqueeze(power[i_batch, :], -1)

            distributions[i_batch, :] = torch.sum(c * p, 0)

        return distributions


    @classmethod
    def self_delegation(cls, D: torch.Tensor):
        diag_indices = torch.arange(D.shape[-1])
        return D[:, diag_indices, diag_indices].mean()

    @classmethod
    def entropy(cls, power: torch.Tensor):
        # power (n_batch, n)

        bs, n = power.shape

        dist = Categorical(logits=power)
        entropy = dist.entropy() / math.log(n)

        return torch.mean(entropy)

    @classmethod
    def step(cls, p_delegatable: torch.Tensor, p_kept: torch.Tensor, D: torch.Tensor):

        diag_indices = torch.arange(D.shape[-1])

        # Get the diagonal elements of W: shape (batch_size, n)
        diag_W = D[:, diag_indices, diag_indices]

        # Compute the amount to keep
        keep_amount = diag_W * p_delegatable
        outflow = p_delegatable - keep_amount
        new_kept = p_kept + keep_amount

        # Sum each row of W and subtract the diagonal, adding a small epsilon for stability.
        outgoing_sums = D.sum(dim=2) - diag_W + 1e-6

        # Create an effective W with zeros on the diagonal.
        mask = torch.ones_like(D)
        mask[:, diag_indices, diag_indices] = 0

        W_eff = D * mask

        # Normalize each row of W_eff by its outgoing sum and weight by the outflow.
        # W_eff: (batch, n, n), outgoing_sums.unsqueeze(2): (batch, n, 1), outflow.unsqueeze(2): (batch, n, 1)
        contribution = (W_eff / outgoing_sums.unsqueeze(2)) * outflow.unsqueeze(2)

        # Sum contributions over the rows to get the new delegatable values (for each target node).
        new_delegatable = contribution.sum(dim=1)

        return new_delegatable, new_kept

    @classmethod
    def solve_delegation_iterative(cls, Ds: list[torch.Tensor], max_iter: int = 100, tol:float =1e-4) -> tuple[torch.Tensor, torch.Tensor]:

        # TODO check dim
        Ds_cat_ready = [torch.unsqueeze(d, 1) for d in Ds]
        D = torch.cat(Ds_cat_ready, dim=1)

        p_delegatable = torch.ones((D.shape[0], D.shape[1]), device=D.device, dtype=D.dtype)
        p_kept = torch.zeros_like(p_delegatable)

        for _ in range(max_iter):
            old_sum = p_delegatable.sum() + p_kept.sum()

            new_delegatable, new_kept = cls.step(p_delegatable, p_kept, D)
            new_sum = new_delegatable.sum() + new_kept.sum()

            if not torch.isclose(old_sum, new_sum, atol=1e-6):
                print(f"Warning: total power changed from {old_sum.item()} to {new_sum.item()}")

            if (torch.allclose(new_delegatable, p_delegatable, atol=tol) and
                torch.allclose(new_kept, p_kept, atol=tol)):
                return new_delegatable + new_kept, D

            p_delegatable, p_kept = new_delegatable, new_kept

        return p_delegatable + p_kept, D


    @classmethod
    def solve_delegation(cls, Ds: list[torch.Tensor]) -> torch.Tensor:

        # Create a delegation matrix
        Ds_cat_ready = [torch.unsqueeze(d, -1) for d in Ds]
        D = torch.cat(Ds_cat_ready, dim=-1)
        n_batch, n, _ = D.shape

        # Extract the power the model wants to keep for itself
        p = torch.diagonal(D, dim1=-2, dim2=-1)
        mask = torch.ones_like(D)
        batch_indices = torch.arange(n_batch).unsqueeze(1).unsqueeze(2)
        diag_indices = torch.arange(n).unsqueeze(0).expand(n, -1)
        mask[batch_indices, diag_indices, diag_indices] = 0
        D_no_diag = D * mask

        identity = torch.zeros_like(D)
        identity[batch_indices, diag_indices, diag_indices] = 1

        p_column = p.unsqueeze(-1)
        inverse = torch.pinverse(identity - D_no_diag)
        influence = torch.bmm(inverse, p_column)
        influence = torch.squeeze(influence)

        return influence

