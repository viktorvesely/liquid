from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner
from liquid_solver import LEsolver, LEInfo


solver = LEsolver()

class LeMlp(nn.Module):
    ...

class LeMnistLearner(Learner[LEInfo]):

    @staticmethod
    def get_model() -> nn.Module:
        
