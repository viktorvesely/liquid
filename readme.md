# Liquid Ensemble Learning

A differentiable, liquid-democracy–inspired ensemble layer for PyTorch. Liquid Ensemble (LE) lets each expert **partially delegate** a sample to other experts via a learned *delegation head* $d(x)$. A differentiable solver then resolves all delegations into a **voting power vector** $p(x)$ that weights each expert’s prediction. This jointly optimizes **what to predict** and **who should speak**.

LE generalizes majority voting and relates to Mixture-of-Experts (MoE), but replaces hard routing/top‑k with **soft, learnable vote delegation** that is optimized end‑to‑end. Unlike hard-outing-MoE (i) it enables ensemble method to be used for the *delegation problem*, (ii) it allows for dynamic allocation of experts (harder samples recieve more experts), (iii) and it can operate under partiall information about the models, making it usefull for federative learning.   

---

## TLDR features

* **Learned delegation**: Every expert outputs both a prediction and a normalized delegation vector $d_i(x)$ over experts.
* **Differentiable resolution**: Two solvers convert the delegation graph into final expert powers $p(x)$:

  * `sink_one`: closed‑form solution with a single absorbing sink; fast & stable.
  * `sink_many`: iterative diffusion with per‑expert sinks; encourages specialization.
* **Uncertainty from structure**: Confidence measures derived from the delegation matrix $D(x)$, power entropy, self‑delegation, and inter‑expert disagreement.
* **Federated/partial observability friendly**: Experts can learn to delegate toward experts with access to complementary views.
* **Auxiliary loss** for sample **load balancing** and sample **specializations**.
* **Drop‑in layer**: Works with regression, classification, or abitrary intermediate (embedding) layer. Tested on MLPs, CNNs, but could be easily adjusted to transformers or RNNs.

---
## Quickstart

Here’s how to define simple MLP and CNN experts that are compatible with `LiquidEnsembleLayer`.  
Each expert must return `(prediction, delegation)` where `delegation` is a probability distribution over experts.

### MLP and CNN Experts

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from liquid.layers.liquid_ensemble_layer import LiquidEnsembleLayer

class MLPExperts(nn.Module):
    def __init__(self, in_dim, out_dim, n_experts, hidden=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        self.router = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_experts), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        y = self.predictor(x)
        d = self.router(x)
        return y, d


class CNNExperts(nn.Module):
    def __init__(self, in_channels, n_experts, out_dim=10):
        super().__init__()
        self.features = nn.Sequential(...)  # e.g. conv layers + pooling + flatten
        feat_dim = ...                      # match output of self.features
        self.predictor = nn.Linear(feat_dim, out_dim)
        self.router = nn.Sequential(
            nn.Linear(feat_dim, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        h = self.features(x)
        y = self.predictor(h)
        d = self.router(h)
        return y, d


class LiquidEnsembleModel(nn.Module):
    def __init__(self, experts, solver="sink_one"):
        super().__init__()
        self.le = LiquidEnsembleLayer(experts, solver=solver)

    def forward(self, x):
        return self.le(x)

    def loss(self, y_pred, y_true, aux_weight=1e-3):
        task_loss = F.mse_loss(y_pred, y_true)
        aux_loss = self.le.auxiliary_loss()
        return task_loss + aux_weight * aux_loss
```

---

## Concept

For a batch of inputs, each expert $i$ returns:

* prediction $y_i(x)$
* delegation $d_i(x)$ (a probability over experts)

Stacking $d_i(x)$ produces a batch‑wise delegation matrix $D(x)\in\mathbb{R}^{B\times N\times N}$. A solver maps $D$ to **power** $p(x)\in\mathbb{R}^{B\times N}$ with $\sum_j p_j(x) = N$. The final output is the power‑weighted mean:

$y(x) = \frac{1}{N}\sum_j p_j(x)\, y_j(x).$


**Solvers**

* `solve_delegation_one_sink(D, epsilon=0.01)` — **closed‑form** via $(I - \tilde D)^{-1}$ with a single absorbing node; redistributes residual power uniformly.
* `solve_delegation_many_sinks(D, epsilon=0.01, long_delegation_penalty=0.90, threshold=0.05)` — **iterative** diffusion solution resolves residual power correctly.

---

## Research Use

### Protein Tertiary Structure (regression)

* Evaluate RMSE and the quality of confidence $c$ by Kendall’s $\tau$ between $c$ and error.
* Study hyperparameters: solver choice, $\lambda_{\text{load}}$, $\lambda_{\text{spec}}$, diffusion penalty.

### CIFAR‑10 (classification)

* Compare **block** vs **long** designs for LE and MoE under fixed compute.
* Report AUVC (area under validation accuracy) to capture convergence speed and overfitting.

> The repo includes CSVs/figures for reproducibility and example experiment scripts.

---

## Design Patterns

* **Long LE**: one LE layer, deep experts.
* **Block LE**: multiple LE layers, shallow experts per block.

## Citation

If you use Liquid Ensemble in academic work, please cite:

```bibtex
@misc {vesely2025liquidensemble,
  title={Liquid Ensemble Learning},
  author={Viktor Vesel\'y},
  year={2025},
  note={arXiv:xxxx.xxxxx}
}
```
---

## Contributing

Issues and PRs are welcome.
