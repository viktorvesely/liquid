from typing import Literal

from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner, find_best_matching_architecture_scalar, to_param
from atomic_networks import get_layers, forward, Mlp
from task_base import Task
from liquid_solver import LEsolver, LEInfo
from math_utils import mix_weighted_logits, mix_weighted_mean
from structs import TrainParams

class MoeGate(nn.Module):
    
    topk: int
    n_models: int

    def setup(self):
        self.W_g = nn.Dense(self.n_models, use_bias=False)
        self.W_noise = nn.Dense(self.n_models, use_bias=False)

    def __call__(self, key: jax.Array, x: jax.Array):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        H = self.W_g(x)
        H = H + jax.random.normal(key, shape=H.shape) * nn.softplus(self.W_noise(x))
        topk_vals, _ = jax.lax.top_k(H, self.topk)
        kth_vals = topk_vals[..., -1:]
        H_top = jnp.where(H >= kth_vals, H, -jnp.inf)
        G = nn.softmax(H_top)
        return G

class MoeMlp(nn.Module):
    n_models: int
    body: tuple[int, ...]
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    solver: LEsolver
    output_mixing: Literal["classification", "regression"]

    def setup(self):
        
        assert self.delegation[-1] == self.n_models, "Last dim of delegation needs to equal to n_models"

        VmappedModelMlp = nn.vmap(
            Mlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.ensemble = VmappedModelMlp(
            out=self.out, 
            delegation=self.delegation, 
            body=self.body
        )

    def __call__(self, x: jax.Array):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        ys, delegation = self.ensemble(x)
        delegation = nn.softmax(delegation, axis=-1)

        leinfo = self.solver.solve_power(delegation)

        if self.output_mixing == "classification":
            y = mix_weighted_logits(ys, leinfo.power)
        elif self .output_mixing == "regression":
            y = mix_weighted_mean(ys, leinfo.power)

        return y, leinfo
    
            

class LeMnistLearner(Learner[LEInfo]):

    @staticmethod
    def get_model(
        train_params: TrainParams,
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        

        builder = lambda alpha: LeMlp(
            n_models=train_params.n_models_in_ensemble,
            body=(),
            out=(to_param(32 * alpha), to_param(16 * alpha), train_params.task.out_dim()),
            delegation=(to_param(32 * alpha), to_param(8 * alpha), train_params.n_models_in_ensemble),
            solver=LEsolver(),
            output_mixing=train_params.task.task_type()
        )

        if param_budget is None:
            model = builder(alpha=1.0)
        else:
            alpha, actual_params = find_best_matching_architecture_scalar(
                param_budget=param_budget, 
                dummy_input=dummy_input, 
                model_builder=builder
            )

            diff = abs(actual_params - param_budget)
            print(f"Difference between requested and actual #params = {diff}")
            assert abs((diff / actual_params) - 1) > 0.2, r"Difference is more than 20% of the budget" 
            model = builder(alpha=alpha)
            
        return model


    @staticmethod
    def forward(key: jax.Array, x: jax.Array, model: LeMlp, params: dict) -> tuple[jax.Array, LEInfo]:
        return model.apply({"params": params}, x)
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        model: LeMlp,
        train_return: LEInfo
    ) -> dict[str, jax.Array]:
        
        return {
            "load_distribution_loss": model.solver.load_distribution_loss(train_return),
            "specialization_loss": model.solver.specialization_loss(train_return)
        }
