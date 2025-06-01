import math
from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..citizens.citizen import Citizen
from ..citizens.name_to_citizen import name_to_citizen


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with fully differentiable **ReLU routing**.

    This variant follows the *ReMoE* paper (Wang et al., 2024) but is simplified
    for whole-sample routing (no token dimension).  Sparsity is enforced by the
    **adaptive L1 regularisation** scheme from Eq. (6-8) of the paper:
    """

    def __init__(
        self,
        experts: list[Citizen],
        router: Citizen,
        load_distribution_lambda: float = 0.0,
        specialization_lambda: float = 0.0
    ) -> None:
        super().__init__()

        self.experts = nn.ModuleList(experts)
        self.load_distribution_lambda = load_distribution_lambda
        self.specialization_lambda = specialization_lambda
        self.last_gate: torch.Tensor = None
        self.last_y: torch.Tensor = None

        self.router = router



    def get_constructor(self) -> dict:

        constructor = {
            'expert_classes': [c.name() for c in self.experts],
            'load_distribution_lambda': self.load_distribution_lambda,
            'specialization_lambda': self.specialization_lambda,
            'experts': [c.get_constructor() for c in self.experts],
            'router_class': self.router.name(),
            'router': self.router.get_constructor()
        }

        return constructor

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        experts_classes = constructor.pop("expert_classes")
        constructors = constructor.pop("experts")
        r_class = constructor.pop("router_class")
        r_constructor = constructor.pop("router")

        experts = []
        for c_class, c_constructor in zip(experts_classes, constructors, strict=True):
            CitizenClass = name_to_citizen[c_class]
            expert = CitizenClass.apply_constructor(c_constructor)
            experts.append(expert)

        RouterClass = name_to_citizen[r_class]
        router = RouterClass(**r_constructor)

        constructor["router"] = router
        constructor["experts"] = experts

        instance = MoELayer(**constructor)


        return instance


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates_raw = self.router(x)
        gates = F.relu(gates_raw)
        gates = gates + 1e-5 # Due to stability

        # Normalise gates
        denom = gates.sum(dim=-1, keepdim=True)
        gates = gates / denom

        self.last_gate = gates.clone().detach()

        y_accum: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for e, expert in enumerate(self.experts):

            # (batch, out)
            y = expert(x)
            g = gates[:, e]

            if y.ndim == 4:
                g = g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                g = g.unsqueeze(-1)

            y_accum.append(y * g)
            ys.append(y)


        ys = torch.stack(ys, dim=0) # (n_citizens, batch, out)
        ys = torch.transpose(ys, 0, 1) # (batch, n_citizens, out)
        self.last_y = ys.clone().detach()

        y_out = torch.stack(y_accum, dim=0).sum(0)
        return y_out


    def speaker_entropy(self, gate: torch.Tensor | None = None):
        """Entropy over histogram of speakers, 1 - everybody speaks (max p) uniformly, 0 - one speaker
        """

        if gate is None:
            gate = self.last_gate

        # gate (n_batch, n_experts)
        bs, n_experts = gate.shape
        speaker = torch.argmax(gate, dim=1)
        histogram = torch.bincount(speaker)
        dist = Categorical(probs=histogram / histogram.sum())

        return dist.entropy() / math.log(n_experts)

    def power_entropy(self, gate: torch.Tensor | None = None):
        """Mean Entropy axis=power, 1 - everybody is speaking every time (not good), 0 - one person speaks per decision (better)
        """

        if gate is None:
            gate = self.last_gate
        # gate (n_batch, n)
        bs, n = gate.shape

        dist = Categorical(probs=gate)
        entropy = dist.entropy() / math.log(n)

        return torch.mean(entropy)


    @staticmethod
    def batch_entropy(x: torch.Tensor):
        bs, n = x.shape
        dist = Categorical(probs=x)
        entropy = dist.entropy() / math.log(n)
        return entropy

    def confidence_power_entropy(self, power: torch.Tensor | None = None):

        if power is None:
            power = self.last_power

        return self.batch_entropy(power)

    def confidence_std(self, ys: torch.Tensor | None = None):

        # (batch, n_citizen, out...)
        if ys is None:
            ys = self.last_y

        # (batch, out...)
        stds_per_out = torch.std(ys, dim=1)
        other_dims = tuple(range(1, stds_per_out.ndim))

        return torch.mean(stds_per_out, dim=other_dims) # (batch,)

    def auxiliary_loss(self, gate: torch.Tensor | None = None, temperature: float = 0.1) -> torch.Tensor:

        if gate is None:
            gate = self.last_gate
        # gate (n_batch, n)

        bs, n = gate.shape

        # (n_batch, n) differentiable argmax, weighted towards highest power citizen
        soft_assign = torch.softmax(gate / temperature, dim=1)

        # (n), how much was each citizen active
        histogram = soft_assign.sum(dim=0)
        dist = Categorical(probs=histogram / histogram.sum())

        # close to 1 - even load, close to 0 not even load
        speaker_entropy = dist.entropy() / math.log(n)

        power_entropy = self.power_entropy(gate)

        return self.specialization_lambda * power_entropy - speaker_entropy * self.load_distribution_lambda
