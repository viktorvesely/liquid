import math
from pathlib import Path
from typing import Callable, Literal
import torch
import torch.nn as nn
from torch.distributions import Categorical

from council import Council
from image_citizen import DelegatingCitizen
from synthetic_citizen import DelegatingCitizen as SyntheticDelegatingCitizen
from majority_council import MajorityCouncil



Delegations = list[torch.Tensor]
Power = torch.Tensor
DelegationMatrix = torch.Tensor
SolveFunc = Callable[[Delegations], tuple[Power, DelegationMatrix]]

class LiquidCouncil(Council):

    def __init__(
            self,
            n_citizens: int,
            citizens: nn.ModuleList | None = None,
            synthetic: bool = False,
            load_distribution_lambda: float = 1.0,
            solver: Literal["sink_one", "sink_many"] = "sink_one",
            layers: int = 24,
            citizens_ratio: tuple[int, int, int] = (2, 1, 1),
            citizen_width: int = 30
        ):

        CitizenClass = SyntheticDelegatingCitizen if synthetic else DelegatingCitizen

        if citizens is None:
            citizens = [CitizenClass(n_citizens, layers // n_citizens, citizens_ratio, citizen_width).to("cuda") for _ in range(n_citizens)]
            citizens = nn.ModuleList(citizens)

        self.load_distribution_lambda = load_distribution_lambda

        super().__init__(n_citizens, citizens)

        self.last_D = None
        self.last_power = None

        self.solver: SolveFunc = None
        if solver == "sink_one":
            self.solver = self.solve_delegation_one_sink
        elif solver == "sink_many":
            self.solver = self.solve_delegation_many_sinks
        else:
            raise ValueError(f"'{solver}' is unknown solver")

    def forward(self, x: torch.Tensor):

        classifications = []
        delegations = []

        for citizen in self.citizens:
            c, d = citizen(x)
            # d (n_batch, n_citizen)
            classifications.append(c)
            delegations.append(d)

        # D (n_batch, n_citizen, n_citizen)
        power, D = self.solver(delegations)
        self.last_D = D

        classifications = torch.cat([torch.unsqueeze(c, 0) for c in classifications], dim=0)

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
    def load_from_checkpoint(cls, folder: Path):
        checkpoint = torch.load(folder / "liquid.pt", weights_only=False)
        model = LiquidCouncil(
            checkpoint["n_citizens"],
            synthetic=True,
            solver=checkpoint["solver"],
            layers=checkpoint["params"],
            citizens_ratio=checkpoint["ratio"],
            citizen_width=checkpoint["width"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


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
    def solve_delegation_iterative(cls, Ds: Delegations,  max_iter: int = 100, tol:float =1e-4) -> Power:

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

        return p_delegatable + p_kept




    @classmethod
    def solve_delegation_many_sinks(
        cls,
        Ds: Delegations,
        epsilon: float = 0.01,
        long_delegation_penalty: float = 0.90,
        threshold: float = 0.01
        ) -> Power:


        # Create a delegation matrix where the columns are the preferences
        # d (batch, n)
        # (batch, n, 1)
        Ds_cat_ready = [torch.unsqueeze(d, 1) for d in Ds]
        D = torch.cat(Ds_cat_ready, dim=1)
        n_batch, n, _ = D.shape
        n2 = 2 * n

        mask = torch.ones((n_batch, n, n), dtype=D.dtype, device=D.device)
        batch_indices = torch.arange(n_batch).unsqueeze(1).unsqueeze(2)
        diag_indices = torch.arange(n).unsqueeze(0).expand(n, -1)
        mask[batch_indices, diag_indices, diag_indices] = 0
        identity = torch.eye(n, dtype=D.dtype, device=D.device).unsqueeze(0).expand(n_batch, -1, -1)


        D_no_diag = D * mask
        self_connections = torch.diagonal(D, offset=0, dim1=1, dim2=2).clone().squeeze()
        # Add small self connection for convergence
        self_connections += epsilon

        D_ext = torch.zeros((n_batch, n2, n2), dtype=D.dtype, device=D.device)
        # Keeep the original connections same without diagonal
        D_ext[:, :n, :n] = D_no_diag

        # The new fake nodes have only self connection
        D_ext[:, n:, n:] = identity

        # Add connections from original nodes to fake nodes
        src_indices = torch.arange(n, device=D.device)
        tgt_indices = src_indices + n
        batch_indices = torch.arange(n_batch, device=D.device).unsqueeze(1).expand(-1, n)
        src_indices = src_indices.unsqueeze(0).expand(n_batch, -1)
        tgt_indices = tgt_indices.unsqueeze(0).expand(n_batch, -1)

        # Flatten all for advanced indexing
        batch_indices = batch_indices.reshape(-1)
        src_indices = src_indices.reshape(-1)
        tgt_indices = tgt_indices.reshape(-1)

        # TODO maybe invert the src and target?
        D_ext[batch_indices, src_indices, tgt_indices] = self_connections.flatten()

        # Everybody gets one vote except the fake voters
        p_column = torch.ones((n_batch, n2, 1), dtype=D.dtype, device=D.device)
        p_column[:, n:] = 0.0
        acumulated_power = torch.zeros_like(p_column)

        D_ext = torch.transpose(D_ext, 1, 2)
        # Iiterate
        for i in range(1_000):
            acumulated_power += p_column
            new_p = torch.bmm(long_delegation_penalty * D_ext, p_column)
            if torch.allclose(new_p, p_column, atol=threshold):
                p_column = new_p
                break

            p_column = new_p

        power = torch.squeeze(acumulated_power[:, n:])

        if power.ndim == 1:
             power = power.unsqueeze(0)

        power = (power / power.sum(1, keepdim=True)) * n

        return power, D

    @classmethod
    def solve_delegation_one_sink(cls, Ds: Delegations, epsilon: float = 0.01) -> Power:

        # Create a delegation matrix where the columns are the preferences
        # d (batch, n)
        # (batch, n, 1)
        Ds_cat_ready = [torch.unsqueeze(d, -1) for d in Ds]
        D = torch.cat(Ds_cat_ready, dim=-1)
        n_batch, n, _ = D.shape

        # Extend the delegation matrix to include the additional agent (agent n+1)
        # For agents 1..n, redefine preferences as ((1 - epsilon)*x_i, epsilon)
        # For the new agent, set its preference to (0, ..., 0, 1)
        D_ext = torch.zeros((n_batch, n + 1, n + 1), dtype=D.dtype, device=D.device)
        # For original agents: scale the old delegation preferences and add epsilon to the new column.
        D_ext[:, :n, :n] = (1 - epsilon) * D
        D_ext[:, n, :n] = epsilon
        D_ext[:, n, n] = 1.0

        # Extract the power the model wants to keep for itself
        # p = torch.diagonal(D, dim1=-2, dim2=-1)
        # p_column = p.unsqueeze(-1)
        # Everybody gets one vote
        p_column = torch.ones((n_batch, n + 1, 1), dtype=D.dtype, device=D.device)

        # Remove diagonal
        mask = torch.ones_like(D_ext)
        batch_indices = torch.arange(n_batch).unsqueeze(1).unsqueeze(2)
        diag_indices = torch.arange(n + 1).unsqueeze(0).expand(n + 1, -1)
        mask[batch_indices, diag_indices, diag_indices] = 0
        D_no_diag = D_ext * mask


        # Create identity batch matrix
        # identity = torch.zeros_like(D)
        # identity[batch_indices, diag_indices, diag_indices] = 1
        identity = torch.eye(n + 1, dtype=D.dtype, device=D.device).unsqueeze(0).expand(n_batch, -1, -1)

        # Solve the equation
        inverse = torch.pinverse(identity - D_no_diag)
        power_ext = torch.bmm(inverse, p_column)
        power_ext = torch.squeeze(power_ext, -1)
        power_ext = power_ext * torch.diagonal(D_ext, offset=0, dim1=1, dim2=2).squeeze()

        # Return lost power uniformly back
        power = torch.clone(power_ext[:, :-1])
        power += power_ext[:, [-1]] / n
        power -= 1 / n

        return power, torch.transpose(D, 1, 2)


if __name__ == "__main__":

    def tt(x):
        return torch.tensor(x, dtype=torch.float).unsqueeze(0)

    Ds = [
        tt([0.5, 0.5, 0, 0]),
        tt([0, 1, 0, 0]),
        tt([0, 0, 1, 0]),
        tt([0, 0, 0, 1]),
    ]

    epsilon = 0.01

    power, _ = LiquidCouncil.solve_delegation_one_sink(Ds, epsilon)

    print(power)

