import math
from pathlib import Path
from typing import Callable, Literal, Self
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Literal

from ..citizens.citizen import Citizen
from ..citizens.name_to_citizen import name_to_citizen

from ..globals import config

Delegations = list[torch.Tensor]
Power = torch.Tensor
DelegationMatrix = torch.Tensor
SolveFunc = Callable[[Delegations], tuple[Power, DelegationMatrix]]

class LiquidEnsembleLayer(nn.Module):

    def __init__(
            self,
            citizens: list[Citizen],
            load_distribution_lambda: float = 0.0,
            specialization_lambda: float = 0.0,
            solver: Literal["sink_one", "sink_many"] = "sink_one"
        ):
        super().__init__()

        n_citizens = len(citizens)
        self.n_citizens = n_citizens
        self.citizens = nn.ModuleList(citizens)
        self.citizens_param_count = torch.tensor([self.count_params(citizen) for citizen in self.citizens], dtype=torch.int)
        self.all_param_count = torch.sum(self.citizens_param_count)

        self.load_distribution_lambda = load_distribution_lambda
        self.specialization_lambda = specialization_lambda

        self.last_D = None
        self.last_power = None
        self.last_y: torch.Tensor = None
        self.solver_name = solver

        self.solver: SolveFunc = None
        if solver == "sink_one":
            self.solver = self.solve_delegation_one_sink
        elif solver == "sink_many":
            self.solver = self.solve_delegation_many_sinks
        else:
            raise ValueError(f"'{solver}' is unknown solver")


    def get_constructor(self) -> dict:

        constructor = {
            'citizens_classes': [c.name() for c in self.citizens],
            'solver': self.solver_name,
            'load_distribution_lambda': self.load_distribution_lambda,
            'specialization_lambda': self.specialization_lambda,
            'citizens': [c.get_constructor() for c in self.citizens]
        }

        return constructor

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        citizen_classes = constructor.pop("citizens_classes")
        constructors = constructor.pop("citizens")

        citizens = []
        for c_class, c_constructor in zip(citizen_classes, constructors, strict=True):
            CitizenClass = name_to_citizen[c_class]
            citizen = CitizenClass.apply_constructor(c_constructor)
            citizens.append(citizen)

        constructor["citizens"] = citizens

        return LiquidEnsembleLayer(**constructor)


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

        if ys.ndim == 5:
            power = power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch, n_citizen, 1, 1, 1)
        else:
            power = power.unsqueeze(-1)  # (batch, n_citizen, 1)

        # (batch, out)
        y = torch.sum(ys * power, dim=1) / self.n_citizens

        self.last_y = ys
        self.last_D = D
        self.last_power = power.squeeze().clone().detach()

        # (batch, out)
        return y


    @staticmethod
    def batch_entropy(x: torch.Tensor):
        bs, n = x.shape
        dist = Categorical(probs=x)
        entropy = dist.entropy() / math.log(n)
        return entropy

    def confidence_D_entropy(self, D: torch.Tensor | None = None) -> torch.Tensor:

        if D is None:
            D = self.last_D

        es = []
        for i in range(D.shape[1]):
            es.append(self.batch_entropy(D[:, i, :]))

        es = torch.stack(es)
        return 1 - torch.mean(es, dim=0)

    def confidence_power_entropy(self, power: torch.Tensor | None = None):

        if power is None:
            power = self.last_power

        return 1 - self.batch_entropy(power)

    def confidence_self_delegation(self, D: torch.Tensor | None = None):
        if D is None:
            D = self.last_D

        diagonal = D.diagonal(dim1=1, dim2=2)
        return torch.mean(diagonal, dim=1)


    def confidence_std(self, ys: torch.Tensor | None = None):

        # (batch, n_citizen, out...)
        if ys is None:
            ys = self.last_y

        # (batch, out...)
        stds_per_out = torch.std(ys, dim=1)
        other_dims = tuple(range(1, stds_per_out.ndim))

        return 1 / (torch.mean(stds_per_out, dim=other_dims) + 1e-6) # (batch,)


    def auxiliary_loss(self, power: torch.Tensor | None = None, temperature: float = 0.1) -> torch.Tensor:

        if power is None:
            power = self.last_power
        # power (n_batch, n)

        bs, n = power.shape

        # (n_batch, n) differentiable argmax, weighted towards highest power expert
        soft_assign = torch.softmax(power / temperature, dim=1)

        # (n), how much was each expert active
        histogram = soft_assign.sum(dim=0)
        dist = Categorical(probs=histogram / histogram.sum())

        # close to 1 - even load, close to 0 not even load
        speaker_entropy = dist.entropy() / math.log(n)

        power_entropy = self.power_entropy(power)

        return self.specialization_lambda * power_entropy - speaker_entropy * self.load_distribution_lambda



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

    @staticmethod
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def p_active_parameters(self, power: torch.Tensor | None = None, T: float = 0.01) -> torch.Tensor:
        """Return (batch,) of [0, 1]
        """

        if power is None:
            power = self.last_power
        # power (n_batch, n)

        power = power / self.n_citizens

        used_mask = (power > T).to(torch.int)

        if self.citizens_param_count.device != used_mask.device:
            self.citizens_param_count = self.citizens_param_count.to(device=used_mask.device)

        used_params = self.citizens_param_count.unsqueeze(0) * used_mask

        return used_params.sum(dim=1) / self.all_param_count


    @classmethod
    def solve_delegation_many_sinks(
        cls,
        Ds: Delegations,
        epsilon: float = 0.01,
        long_delegation_penalty: float = 0.90,
        threshold: float = 0.05
        ) -> Power:


        # Create a delegation matrix where the columns are the preferences
        # d (batch, n)
        # (batch, n, 1)
        Ds_cat_ready = [torch.unsqueeze(d, 1) for d in Ds]
        D = torch.cat(Ds_cat_ready, dim=1)
        n_batch, n, _ = D.shape
        n2 = 2 * n

        if config.make_delegation_uniform:
            uniform_power = torch.ones((n_batch, n), dtype=torch.float, device=D.device)
            return uniform_power, D


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

        if config.make_delegation_uniform:
            uniform_power = torch.ones((n_batch, n), dtype=torch.float, device=D.device)
            return uniform_power, D

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

        # print(f"\nResidual power per citizen={torch.mean(power_ext[:, [-1]] / n):.3f}\n")

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

