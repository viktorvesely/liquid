
from dataclasses import dataclass
from typing import Literal, Protocol, Type

from flax import struct
import jax
from task_base import Task
from learner_base import Learner
from atomic_networks import Architecture

@dataclass(frozen=True)
class TrainParams:
    
    batch_size: int
    preload_batches_to_gpu: int
    valid_batches: int
    epochs: int
    lr: float
    task: Type[Task]
    n_predictors: int
    n_delegators: int
    delegators_mixing: Literal["sum", "product"]
    ambiguity_gradient: Literal["both", "delegators", "none"]
    architecture: Architecture
    load_balancing_lambda: float = 0.2
    
type Predictions = jax.Array  # (BS, n_predictors, n_output)
type Delegations = jax.Array  # (BS, n_delegators, n_predictor)

@struct.dataclass
class ForwardReturn:
    delegations: Delegations 
    predictions: Predictions 


@struct.dataclass
class ForwardArgs:
    x: jax.Array


class Ensemble(Protocol):
    def apply(self, params: dict, args: ForwardArgs) -> ForwardReturn:
        ...

class Predictors(Protocol):
    def apply(self, params: dict, x: jax.Array) -> Predictions:
        ...

class Delegators(Protocol):
    def apply(self, params: dict, x: jax.Array) -> Delegations:
        ...

@struct.dataclass
class InOutData:
    
    x: jax.Array
    y: jax.Array