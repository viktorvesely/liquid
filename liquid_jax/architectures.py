
from flax import linen as nn
import jax 
import jax.numpy as jnp

from atomic_networks import CnnMlp, Mlp
from structs import ForwardReturn, ForwardArgs

from flax.core import FrozenDict, freeze

class Predictors(nn.Module):
    n_predictors: int
    predictor: tuple[int, ...]
    n_cnn_layers: int = 0
    kernel_size: int = 3

    def setup(self):
        
        pred_cnn = self.predictor[:self.n_cnn_layers]
        pred_mlp = self.predictor[self.n_cnn_layers:]

        if self.n_cnn_layers > 0:
            BaseModule = CnnMlp
            pred_kwargs = {'cnn': pred_cnn, 'mlp': pred_mlp, 'kernel_size': self.kernel_size, 'stride': 2}
        else:
            BaseModule = Mlp
            pred_kwargs = {'body': pred_mlp}

        Predictors = nn.vmap(
            BaseModule,
            variable_axes={'params': 0},
            split_rngs={'params': True}, # Vmap over different models
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_predictors,
            out_axes=1 # Stack the model outputs to axis=1
        )
        
        self.predictors = Predictors(**pred_kwargs)

    def __call__(self, x: jax.Array):

        if self.n_cnn_layers == 0:
            x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))

        predictions = self.predictors(x)
        return predictions

class Delegators(nn.Module):
    n_delegators: int
    n_predictors: int
    delegator: tuple[int, ...]
    n_cnn_layers: int = 0
    kernel_size: int = 3

    def setup(self):
        
        assert self.delegator[-1] == self.n_predictors, "Last dim of delegation needs to equal to n_predictors"

        del_cnn = self.delegator[:self.n_cnn_layers]
        del_mlp = self.delegator[self.n_cnn_layers:]

        if self.n_cnn_layers > 0:
            BaseModule = CnnMlp
            del_kwargs = {'cnn': del_cnn, 'mlp': del_mlp, 'kernel_size': self.kernel_size, 'stride': 2}
        else:
            BaseModule = Mlp
            del_kwargs = {'body': del_mlp}

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
    
        delegations_logits = self.delegators(x)

        return delegations_logits



class Ensemble(nn.Module):
    n_predictors: int
    n_delegators: int
    predictor: tuple[int, ...]
    delegator: tuple[int, ...]
    n_cnn_layers: int = 0
    kernel_size: int = 3

    def setup(self):
        self.predictors, self.delegators = get_modules(self)
        

    def __call__(self, args: ForwardArgs) -> ForwardReturn:
        
        delegations_logits = self.delegators(args.x)
        predictions = self.predictors(args.x)

        return ForwardReturn(
            delegations=delegations_logits,
            predictions=predictions
        )


def get_modules(ensemble: Ensemble):
    predictors = Predictors(
        n_predictors = ensemble.n_predictors,
        predictor = ensemble.predictor,
        n_cnn_layers = ensemble.n_cnn_layers,
        kernel_size = ensemble.kernel_size
    )

    delegators = Delegators(
        n_delegators=ensemble.n_delegators,
        n_predictors=ensemble.n_predictors,
        delegator=ensemble.delegator,
        n_cnn_layers=ensemble.n_cnn_layers,
        kernel_size=ensemble.kernel_size
    )

    return predictors, delegators

def split_ensemble(
        ensemble: Ensemble,
        ensemble_params: FrozenDict
    ) -> tuple[tuple[Predictors, FrozenDict], tuple[Delegators, FrozenDict]]:
    
    delegators_params = ensemble_params["delegators"]
    predictors_params = ensemble_params["predictors"]

    predictors, delegators = get_modules(ensemble)

    return (predictors, predictors_params), (delegators, delegators_params)    

