import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

from council import Council
from image_citizen import DelegatingCitizen
from synthetic_citizen import DelegatingCitizen as SyntheticDelegatingCitizen
from majority_council import MajorityCouncil

class LiquidCouncil(Council):


    def __init__(
            self,
            n_citizens: int,
            citizens: nn.ModuleList | None = None,
            synthetic: bool = False,
            load_distribution_lambda: float = 1.0
        ):

        CitizenClass = SyntheticDelegatingCitizen if synthetic else DelegatingCitizen

        if citizens is None:
            citizens = [CitizenClass(n_citizens).to("cuda") for _ in range(n_citizens)]
            citizens = nn.ModuleList(citizens)

        self.load_distribution_lambda = load_distribution_lambda

        super().__init__(n_citizens, citizens)

        self.last_D = None
        self.last_power = None


    def forward(self, x: torch.Tensor):

        classifications = []
        delegations = []

        for citizen in self.citizens:
            c, d = citizen(x)
            # d (n_batch, n_citizen)
            classifications.append(c)
            delegations.append(d)

        # D (n_batch, n_citizen, n_citizen)
        power, D = self.solve_delegation_iterative(delegations)

        classifications = torch.cat([torch.unsqueeze(c, 0) for c in classifications], dim=0)

        self.last_D = D
        self.last_power = power

        # (n, batch, out)
        return classifications


    def load_distribution_loss(self, power: torch.Tensor | None = None, temperature: float = 0.1):

        if power is None:
            power = self.last_power
        # power (n_batch, n)

        bs, n = power.shape

        # (n_batch, n) differentiable argmax, weighted towards highest power citizen
        soft_assign = torch.softmax(power / temperature, dim=1)

        # (n), how much was each citizen active
        histogram = soft_assign.sum(dim=0)
        dist = Categorical(probs=histogram / histogram.sum())

        # close to 1 - even load, close to 0 not even load
        normalized_entropy = dist.entropy() / math.log(n)

        power_entropy = self.power_entropy()

        return self.load_distribution_lambda * (power_entropy - normalized_entropy)

    def loss(self, y: torch.Tensor, classifications: torch.Tensor):
        predict_losses = self.predict_loss(y, classifications)

        final_loss = 0
        for i_citizen, citizen_predict_loss in enumerate(predict_losses):
            # citizen_predict_loss (n_batch,)
            # power (n_batch, n)
            final_loss += citizen_predict_loss * self.last_power[:, i_citizen]

        load_loss = self.load_distribution_loss()

        return final_loss.mean() + load_loss


    def vote(self, classifications: torch.Tensor, power: torch.Tensor | None = None):
        # classifications (n, n_batch, out)
        # power (n_batch, n)
        # distributions (n_batch, out)

        if power is None:
            power = self.last_power

        bs = power.shape[0]

        # Just in case normalize power
        sums = power.sum(1, keepdim=True)
        if (~torch.isclose(sums, torch.tensor(self.n_citizens, device=power.device, dtype=power.dtype), atol=1e-2)).any():
            print("Warning, power sum is not close to n_citizens")

        power = power / sums

        distributions = torch.zeros((bs, self.n_classes), dtype=power.dtype, device=power.device)

        for i_batch in range(bs):
            # (n, digits)
            c = classifications[:, i_batch, :]

            # (n, 1)
            p = torch.unsqueeze(power[i_batch, :], -1)

            distributions[i_batch, :] = torch.sum(c * p, 0)

        labels_hat = torch.argmax(distributions, dim=1)
        return labels_hat



    def self_delegation(self):

        D = self.last_D
        diag_indices = torch.arange(D.shape[-1])
        return D[:, diag_indices, diag_indices].mean()


    def speaker_entropy(self, power: torch.Tensor | None = None):
        """Entropy over histogram of speakers, 1 - everybody speaks (max p) uniformly, 0 - one speaker
        """

        if power is None:
            power = self.last_power

        # power (n_batch, n)
        bs, n = power.shape
        speaker = torch.argmax(power, dim=1)
        histogram = torch.bincount(speaker)
        dist = Categorical(probs=histogram / histogram.sum())
        return dist.entropy() / math.log(n)

    def power_entropy(self, power: torch.Tensor | None = None):
        """Mean Entropy axis=power, 1 - everybody is speaking every time (not good), 0 - one person speaks per decision (better)
        """

        if power is None:
            power = self.last_power
        # power (n_batch, n)

        bs, n = power.shape

        dist = Categorical(probs=power)
        entropy = dist.entropy() / math.log(n)

        return torch.mean(entropy)

    def cut_delegation_heads(self) -> MajorityCouncil:
        headless = [citizen.cut_delegation_head() for citizen in self.citizens]
        headless = nn.ModuleList(headless)
        return MajorityCouncil(self.n_citizens, headless)

    def name(self):
        return "liquid"

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

