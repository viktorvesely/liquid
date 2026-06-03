from typing import Literal

from flax import linen as nn
import jax 
import jax.numpy as jnp
import optuna

from learner_base import Learner, count_params_without_alloc, find_best_matching_architecture_scalar, to_param
from atomic_networks import CnnMlp, Mlp, get_cnn_layers, get_layers, forward
from liquid_solver import LEsolver
from math_utils import non_uniformity
from structs import TrainParams, TrainReturn


class Le(nn.Module):
    n_predictors: int
    n_delegators: int
    predictor: tuple[int, ...]
    delegator: tuple[int, ...]
    solver: LEsolver | None
    output_mixing: Literal["classification", "regression"]
    delegation_mixing: Literal["sum", "product"]
    n_cnn_layers: int = 0
    kernel_size: int = 3

    def setup(self):
        
        assert self.delegator[-1] == self.n_predictors, "Last dim of delegation needs to equal to n_predictors"

        pred_cnn = self.predictor[:self.n_cnn_layers]
        pred_mlp = self.predictor[self.n_cnn_layers:]
        del_cnn = self.delegator[:self.n_cnn_layers]
        del_mlp = self.delegator[self.n_cnn_layers:]

        if self.n_cnn_layers > 0:
            BaseModule = CnnMlp
            pred_kwargs = {'cnn': pred_cnn, 'mlp': pred_mlp, 'kernel_size': self.kernel_size, 'stride': 2}
            del_kwargs = {'cnn': del_cnn, 'mlp': del_mlp, 'kernel_size': self.kernel_size, 'stride': 2}
        else:
            BaseModule = Mlp
            pred_kwargs = {'body': pred_mlp}
            del_kwargs = {'body': del_mlp}

        Predictors = nn.vmap(
            BaseModule,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_predictors,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.predictors = Predictors(**pred_kwargs)

        if self.n_delegators > 0:
            Delegators = nn.vmap(
                BaseModule,
                variable_axes={'params': 0},
                split_rngs={'params': True}, # Vmap over different models
                in_axes=None, # Do not vmap over batch elements
                axis_size=self.n_delegators,
                out_axes=1 # Stack the model outputs to axis=1
            )

            self.delegators = Delegators(**del_kwargs)
        else:
            self.delegators = lambda x: jnp.ones((x.shape[0], 1, self.n_predictors)) / self.n_predictors 

    def __call__(self, x: jax.Array):

        if self.n_cnn_layers == 0:
            x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))

        ys = self.predictors(x)
        delegation_logit = self.delegators(x)
        delegation_logprobs = nn.log_softmax(delegation_logit, axis=-1)
        delegation_probs = nn.softmax(delegation_logit, axis=-1)



        if self.delegation_mixing == "sum":
            unscaled_power = jnp.mean(delegation_probs, axis=-2)
            power = unscaled_power / (jnp.sum(unscaled_power, axis=-1, keepdims=True) + 1e-8)
            output = TrainReturn(
                delegation=delegation_probs,
                power=power,
                ys=ys
            )
        elif self.delegation_mixing == "product":
            power  = nn.softmax(jnp.sum(delegation_logprobs, axis=-2))
            output = TrainReturn(
                delegation=delegation_probs,
                power=power,
                ys=ys
            )

        else:
            raise ValueError(f"{self.delegation_mixing} unknown delegation mixing")
        
        return output
    

LOAD_DISTRIBUTION_LAMBDA = 1

class LeLearner(Learner[TrainReturn]):

    @staticmethod
    def get_model(
        train_params: TrainParams,
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        


        model = Le(
            n_predictors=train_params.n_predictors,
            n_delegators=train_params.n_delegators,
            predictor=train_params.architecture.predictor,
            delegator=train_params.architecture.delegator,
            n_cnn_layers=train_params.architecture.cnn,
            solver=None,
            output_mixing=train_params.task.task_type(),
            delegation_mixing=train_params.delegators_mixing
        )


        # builder = lambda alpha: Le(
        #     n_predictors=train_params.n_predictors,
        #     n_delegators=train_params.n_delegators,
        #     predictor=(to_param(16 * alpha), to_param(8 * alpha), train_params.task.out_dim()),
        #     delegator=(to_param(8 * alpha), train_params.n_predictors),
        #     solver=None,
        #     output_mixing=train_params.task.task_type(),
        #     delegation_mixing="mean"
        # )



        # if param_budget is None:
        #     model = builder(alpha=1.0)
        #     params = count_params_without_alloc(model, dummy_input=dummy_input)
        #     print(f"{(params / 1_000):_.2f}K params")
        # else:
        #     alpha, actual_params = find_best_matching_architecture_scalar(
        #         param_budget=param_budget, 
        #         dummy_input=dummy_input, 
        #         model_builder=builder
        #     )

        #     diff = abs(actual_params - param_budget)
        #     print(f"Difference between requested and actual #params = {diff}")
        #     relative_diff = diff / param_budget
        #     assert relative_diff < 0.2, f"Budget mismatch: {relative_diff:.2%} error"
        #     model = builder(alpha=alpha)
            
        return model


    @staticmethod
    def forward(key: jax.Array, x: jax.Array, model: Le, params: dict) -> tuple[jax.Array, TrainReturn]:
        return model.apply({"params": params}, x)
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        model: nn.Module,
        params: dict,
        train_return: TrainReturn,
        train_params: TrainParams
    ) -> dict[str, jax.Array]:
        
        soft_chair = LEsolver.get_soft_chair_dist(train_return.power)
        load_distribution_loss = non_uniformity(soft_chair) * LOAD_DISTRIBUTION_LAMBDA

        return {
            "load_distribution_loss": load_distribution_loss
        }

    
    @staticmethod
    def boot_from_trial(
        train_params: TrainParams,
        dummy_input: jax.Array,
        trial: optuna.Trial,
    ) -> nn.Module:
    
        n_models = trial.suggest_int("n_models", 2, 64, log=True)
        dynamic_limit = int(256 / ((n_models - 1)**(1/2)))

        def suggest_block(name: str, mindepth: int, maxdepth: int, maxwidth: int):
            
            current_max = min(maxwidth, dynamic_limit)
            depth = trial.suggest_int(f"{name}_depth", mindepth, maxdepth)
            block = [trial.suggest_int(f"{name}_layer_{i}", 1, current_max, log=True) for i in range(depth)]
            block = tuple(block)
            return block
        
        body = suggest_block("body", mindepth=2, maxdepth=5, maxwidth=64)
        out = suggest_block("out", mindepth=0, maxdepth=4, maxwidth=128)
        delegation = suggest_block("delegation", mindepth=0, maxdepth=4, maxwidth=128)

        out = out + (train_params.task.out_dim(),)
        delegation = delegation + (n_models,)

        model = Le(
            n_models=n_models,
            body=body,
            out=out,
            delegation=delegation,
            solver=LEsolver(
                load_distribution_lambda=trial.suggest_float("load_distribution_lambda", 1e-8, 5.0, log=True),
                load_distribution_temperature=trial.suggest_float("load_distribution_temperature", 1e-8, 2.0),
                specialization_lambda=trial.suggest_float("specialization_lambda", 1e-8, 5.0, log=True),
                long_delegations_penalty=trial.suggest_float("long_delegations_penalty", 0.1, 0.99),
                solver=trial.suggest_categorical("solver", ["sink_one", "sink_many"])
            ),
            output_mixing=train_params.task.task_type()
        )

        param_count = count_params_without_alloc(model, dummy_input)
        print(param_count, n_models, body, out, delegation)
        if param_count > 5_000_000:
            raise optuna.TrialPruned()
    
        return model