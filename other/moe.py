import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class ReLURouterMoE(nn.Module):
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
        num_experts: int,
        hidden_dim: int,
        k_active: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.GELU(),
        bias: bool = True,
        lambda0: float = 1e-8,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        assert 0 < k_active <= num_experts, "k_active must be in 1…E"

        self.E = num_experts
        self.k = k_active
        self.target_sparsity = 1.0 - k_active / num_experts
        self.alpha = alpha
        self.last_l1_gate: torch.Tensor = None

        self.register_buffer("lambda_L1", torch.tensor(lambda0))

        self.router = nn.Linear(input_dim, num_experts, bias=bias)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=bias),
                activation,
                nn.Linear(hidden_dim, input_dim, bias=bias),
            ) for _ in range(num_experts)
        ])

        # Fallback expert idx (keeps gradient non‑zero if all gates 0)
        self.register_buffer("_fallback", torch.tensor(0, dtype=torch.long))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d = x.shape
        gates_raw = self.router(x)          # (B,E)
        gates = F.relu(gates_raw)           # ReLU routing

        with torch.no_grad():
            sparsity = (gates == 0).float().mean()
            self._update_lambda(sparsity)

        l1_reg = gates.mean()
        self.last_l1_gate = l1_reg

        # Normalise gates (optional—keeps scale stable)
        denom = gates.sum(dim=-1, keepdim=True)
        empty = denom.squeeze(-1) == 0
        if empty.any():
            gates[empty, self._fallback] = 1.0
            denom = gates.sum(dim=-1, keepdim=True)
        gates = gates / denom

        # Weighted expert aggregation
        y_accum: list[torch.Tensor] = []
        for e, expert in enumerate(self.experts):
            y_e = expert(x) * gates[:, e:e+1]  # (B,d)
            y_accum.append(y_e)

        y_out = torch.stack(y_accum, 0).sum(0)
        return y_out


    def sparsity_loss(self, l1_gate: torch.Tensor | None = None) -> torch.Tensor:
        l1_gate = self.last_l1_gate if  l1_gate is None else l1_gate
        return self.lambda_L1 * l1_gate


    def _update_lambda(self, batch_sparsity: torch.Tensor) -> None:
        """Update λ in‑place once *per call* using eq (7)."""
        target = self.target_sparsity
        direction = torch.sign(target - batch_sparsity)  # +1 if too dense, ‑1 if too sparse
        if direction != 0:
            self.lambda_L1.mul_(self.alpha ** direction.item())
