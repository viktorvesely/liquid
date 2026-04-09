from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner, get_layers, forward, find_best_matching_architecture_scalar, to_param
from task_base import Task

class Mlp(nn.Module):
    
    body: tuple[int, ...]
    
    def setup(self):
        self.body_layers = get_layers(self.body)

    def __call__(self, x: jax.Array):
        out = forward(x, last_linear=True, layers=self.body_layers)
        return out

class DeMlp(nn.Module):
    n_models: int
    body: tuple[int, ...] | None = None

    def setup(self):

        VmappedLeModelMlp = nn.vmap(
            Mlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.ensemble = VmappedLeModelMlp(
            body=self.body
        )

    def __call__(self, x: jax.Array):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        y= self.ensemble(x)
        return y

def mix_mean(
        y: jax.Array,
    ) -> jax.Array:
    # y (batch, n_models, n_out)
    assert y.ndim == 3
    return jnp.mean(y, axis=1)

def mix_logits(
    logits: jax.Array,
) -> jax.Array:
    # logits (batch, n_models, n_out)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mixed_logits = jnp.mean(log_probs, axis=1)
    return mixed_logits
        

class DeLearner(Learner[None]):

    @staticmethod
    def get_model(
        out_dim: int,
        n_models: int,
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        

        builder = lambda alpha: DeMlp(
            n_models=n_models,
            body=(to_param(32 * alpha), to_param(16 * alpha), out_dim)
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
        params: dict,
        task: Task
    ) -> tuple[jax.Array, None]:
        
        ys = model.apply({"params": params}, x)

        tt = task.task_type()
        if tt == "classification":
            y = mix_logits(ys)
        elif tt == "regression":
            y = mix_mean(ys)

        return y, None
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        train_return: None
    ) -> dict[str, jax.Array]:
        return {}
