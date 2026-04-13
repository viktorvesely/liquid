from typing import Literal

from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner, find_best_matching_architecture_scalar, to_param
from atomic_networks import get_layers, forward, Mlp, Cnn, CnnMlp
from math_utils import mix_logits, mix_mean
from structs import TrainParams

class DeMlp(nn.Module):
    n_models: int
    body: tuple[int, ...]
    output_mixing: Literal["classification", "regression"]
    cnn_layers: int

    def setup(self):

    
        VmappedLeModelMlp = nn.vmap(
            CnnMlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.ensemble = VmappedLeModelMlp(
            cnn=self.body[:self.cnn_layers], mlp=self.body[self.cnn_layers:],
            kernel_size=3, stride=2
        )

    def __call__(self, x: jax.Array):

        if self.cnn_layers ==  0:
            x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        
        ys = self.ensemble(x)

        if self.output_mixing == "classification":
            y = mix_logits(ys)
        elif self.output_mixing == "regression":
            y = mix_mean(ys)
        
        return y
        

class DeLearner(Learner[None]):

    @staticmethod
    def get_model(
        train_params: TrainParams,
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        

        builder = lambda alpha: DeMlp(
            n_models=train_params.n_models_in_ensemble,
            body=(to_param(9 * alpha), to_param(18 * alpha), to_param(32 * alpha), train_params.task.out_dim()),
            output_mixing=train_params.task.task_type(),
            cnn_layers=3
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
    def forward(
        key: jax.Array,
        x: jax.Array,
        model: nn.module,
        params: dict
    ) -> tuple[jax.Array, None]:
        y = model.apply({"params": params}, x)
        return y, None
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        model: nn.Module,
        params: dict,
        train_return: None
    ) -> dict[str, jax.Array]:
        return {}
