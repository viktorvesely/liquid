from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner, find_best_matching_architecture_scalar, to_param, get_layers, forward
from task_base import Task
from liquid_solver import LEsolver, LEInfo

solver = LEsolver(
    load_distribution_lambda=2e-1,
    specialization_lambda=0,
)

class LeModelMlp(nn.Module):
    
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None
    
    def setup(self):
        self.body_layers = get_layers(self.body)
        self.out_layers = get_layers(self.out)
        self.delegation_layers = get_layers(self.delegation)

    def __call__(self, x: jax.Array):
        h = x
        h_body = forward(h, last_linear=False, layers=self.body_layers)
        y = forward(h_body, last_linear=True, layers=self.out_layers)
        d = forward(h_body, last_linear=True, layers=self.delegation_layers)

        return y, d

class LeMlp(nn.Module):
    n_models: int
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    body: tuple[int, ...] | None = None

    def setup(self):
        
        assert self.delegation[-1] == self.n_models, "Last dim of delegation needs to equal to n_models"

        VmappedLeModelMlp = nn.vmap(
            LeModelMlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.ensemble = VmappedLeModelMlp(
            out=self.out, 
            delegation=self.delegation, 
            body=self.body
        )

    def __call__(self, x: jax.Array):
        x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))
        y, d = self.ensemble(x)
        d = nn.softmax(d, axis=-1)
        return y, d
    
            

class LeMnistLearner(Learner[LEInfo]):

    @staticmethod
    def get_model(
        out_dim: int,
        n_models: int,
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        

        builder = lambda alpha: LeMlp(
            n_models=n_models,
            body=(to_param(32 * alpha),),
            out=(to_param(16 * alpha), out_dim),
            delegation=(to_param(8 * alpha), n_models)
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
    def forward(key: jax.Array, x: jax.Array, model: LeMlp, params: dict, task: Task) -> tuple[jax.Array, LEInfo]:
        
        ys, delegation = model.apply({"params": params}, x)
        leinfo = solver.solve_power(delegation)

        y = solver.mix_power_logits(ys, leinfo.power)

        return y, leinfo
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        train_return: LEInfo
    ) -> dict[str, jax.Array]:
        
        return {
            "load_distribution_loss": solver.load_distribution_loss(train_return),
            "specialization_losss": solver.specialization_loss(train_return)
        }
