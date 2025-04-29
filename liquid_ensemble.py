import math
from pathlib import Path
from typing import Callable, Literal
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Literal


Delegations = list[torch.Tensor]
Power = torch.Tensor
DelegationMatrix = torch.Tensor
SolveFunc = Callable[[Delegations], tuple[Power, DelegationMatrix]]

class LiquidEnsembleLayer(nn.Module):

    def __init__(
            self,
            citizens: nn.ModuleList | None = None,
            load_distribution_lambda: float = 0.0,
            specialization_lambda: float = 0.0,
            solver: Literal["sink_one", "sink_many"] = "sink_one"
        ):
        super().__init__()


        n_citizens = len(citizens)
        self.n_citizens = n_citizens
        self.citizens = citizens

        self.load_distribution_lambda = load_distribution_lambda
        self.specialization_lambda = specialization_lambda

        self.last_D = None
        self.last_power = None
        self.last_y: torch.Tensor = None

        self.solver: SolveFunc = None
        if solver == "sink_one":
            self.solver = self.solve_delegation_one_sink
        elif solver == "sink_many":
            self.solver = self.solve_delegation_many_sinks
        else:
            raise ValueError(f"'{solver}' is unknown solver")

    def forward(self, x: torch.Tensor):

        ys = []
        delegations = []

        for citizen in self.citizens:
            y_citizen, delegation = citizen(x)
            # d (batch, n_citizen)
            ys.append(y_citizen)
            delegations.append(delegation)

        # D (batch, n_citizen, n_citizen)
        # Power (batch, n_citizen)
        power, D = self.solver(delegations)

        power_sums = power.sum(1, keepdim=True)
        if not torch.isclose(
            power_sums,
            torch.tensor(self.n_citizens, device=power.device, dtype=power.dtype),
            atol=1e-2
        ).all():
            print("Warning, power sum is not close to n_citizens")

        # (n_citizen, batch, out)
        ys = torch.stack(ys)
        # (batch, n_citizen, out)
        ys = torch.transpose(ys, 0, 1)

        # (batch, out)
        y = torch.sum(ys * power.unsqueeze(2), dim=1)

        self.last_y = ys
        self.last_D = D
        self.last_power = power

        # (batch, out)
        return y

    def load_distribution_loss(self, power: torch.Tensor | None = None, temperature: float = 0.1) -> torch.Tensor:

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
        speaker_entropy = dist.entropy() / math.log(n)

        power_entropy = self.power_entropy()

        return self.specialization_lambda * power_entropy - speaker_entropy * self.load_distribution_lambda


    def vote(self, classifications: torch.Tensor, power: torch.Tensor | None = None):
        # classifications (n, n_batch, out)
        # power (n_batch, n)
        # distributions (n_batch, out)

        if power is None:
            power = self.last_power

        bs = power.shape[0]

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

    power, _ = LiquidEnsembleLayer.solve_delegation_one_sink(Ds, epsilon)

    print(power)

