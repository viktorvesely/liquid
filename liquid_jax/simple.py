from typing import Literal

from flax import linen as nn
import jax 
import jax.numpy as jnp
import optuna

from learner_base import Learner, count_params_without_alloc, find_best_matching_architecture_scalar, to_param
from atomic_networks import Mlp, get_cnn_layers, get_layers, forward
from liquid_solver import LEsolver, LEInfo
from math_utils import mix_weighted_logits, mix_weighted_mean, non_uniformity
from structs import TrainParams

class Le(nn.Module):
    n_predictors: int
    n_delegators: int
    predictor: tuple[int, ...]
    delegator: tuple[int, ...]
    solver: LEsolver | None
    delegation_mixing: Literal["mean", "liquid"]
    cnn_body: bool = False

    def setup(self):
        
        assert self.delegator[-1] == self.n_predictors, "Last dim of delegation needs to equal to n_predictors"

        Predictors = nn.vmap(
            Mlp,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_predictors,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.predictors = Predictors(
            body=self.predictor
        )


        if self.n_delegators > 0:
            Delegators = nn.vmap(
                Mlp,
                variable_axes={'params': 0},
                split_rngs={'params': True}, # Vmap over different models
                in_axes=None, # Do not vmap over batch elements
                axis_size=self.n_delegators,
                out_axes=1 # Stack the model outputs to axis=1
            )

            self.delegators = Delegators(body=self.delegator)
        else:
            self.delegators = lambda x: jnp.ones((x.shape[0], 1, self.n_predictors)) / self.n_predictors 

    def __call__(self, x: jax.Array):

        if not self.cnn_body:
            x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))

        ys = self.predictors(x)
        delegation = self.delegators(x)
        delegation = nn.softmax(delegation, axis=-1)


        if self.delegation_mixing == "liquid":
            leinfo = self.solver.solve_power(delegation)
        elif self.delegation_mixing == "mean":
            unscaled_power = jnp.mean(delegation, axis=-2)
            power = unscaled_power / jnp.sum(unscaled_power, keepdims=True)
            leinfo = LEInfo(
                delegation=delegation,
                power=power
            )
        else:
            raise ValueError(f"{self.delegation_mixing} unknown delegation mixing")
        
        leinfo = leinfo.replace(ys=ys)

        # Shape (bs, experts, body_dim)
        if self.output_mixing == "classification":
            y = mix_weighted_logits(ys, leinfo.power)
        elif self .output_mixing == "regression":
            y = mix_weighted_mean(ys, leinfo.power)

        return y, leinfo