import math
from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..citizens.citizen import Citizen, CitizenFC
from ..citizens.name_to_citizen import name_to_citizen


class MoELayer(nn.Module):
    """Mixture‑of‑Experts layer with fully differentiable **ReLU routing**.

    This variant follows the *ReMoE* paper (Wang et al., 2024) but is simplified
    for whole‑sample routing (no token dimension).  Sparsity is enforced by the
    **adaptive L1 regularisation** scheme from Eq. (6–8) of the paper:

        λ ← λ · α^{sign(target − S)}

    where *S* is the batch‑averaged sparsity (fraction of zero gates).
    The coefficient λ is stored as a buffer so it is saved with the model but is
    *not* a learnable parameter.
    """

    def __init__(
        self,
        input_dim: int,
        experts: list[Citizen],
        k_active: int = 1,
        sparsity_lambda: float = 1e-8,
        sparsity_move: float = 1.0,
    ) -> None:
        super().__init__()

        num_experts = len(experts)
        assert 0 < k_active <= num_experts, "k_active must be in 1…E"

        self.input_dim = input_dim
        self.experts = nn.ModuleList(experts)
        self.E = num_experts
        self.k_active = k_active
        self.target_sparsity = 1.0 - k_active / num_experts
        self.sparsity_move = sparsity_move
        self.last_gate: torch.Tensor = None

        self.register_buffer("sparsity_lambda", torch.tensor(sparsity_lambda))

        self.router = CitizenFC(input_dim, num_experts, layers=4, width=50)


    def get_constructor(self) -> dict:

        constructor = {
            'expert_classes': [c.name() for c in self.experts],
            'input_dim': self.input_dim,
            'k_active': self.k_active,
            'sparsity_move': self.sparsity_move,
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

        constructor["experts"] = experts

        instance = MoELayer(**constructor)

        RouterClass = name_to_citizen[r_class]
        instance.router = RouterClass(**r_constructor)

        return instance


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates_raw = self.router(x)
        gates = F.relu(gates_raw)
        gates = gates + 1e-5
        self.last_gate = gates.clone().detach()

        with torch.no_grad():
            sparsity = (gates == 0).float().mean()
            self._update_lambda(sparsity)

        # Normalise gates (optional—keeps scale stable)
        denom = gates.sum(dim=-1, keepdim=True)
        gates = gates / denom

        y_accum: list[torch.Tensor] = []
        for e, expert in enumerate(self.experts):
            # TODO not compute bellow some threshold
            y_e = expert(x) * gates[:, e:e+1]
            y_accum.append(y_e)

        y_out = torch.stack(y_accum, 0).sum(0)
        return y_out

    def sparsity_loss(self, mean_l1_gate: torch.Tensor | None = None) -> torch.Tensor:
        mean_l1_gate = self.last_gate.mean() if  mean_l1_gate is None else mean_l1_gate
        return self.sparsity_lambda * mean_l1_gate

    def _update_lambda(self, batch_sparsity: torch.Tensor) -> None:
        """Update λ in‑place once *per call* using eq (7)."""
        target = self.target_sparsity
        direction = torch.sign(target - batch_sparsity)  # +1 if too dense, ‑1 if too sparse
        if direction != 0:
            self.sparsity_lambda.mul_(self.sparsity_move ** direction.item())

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
