
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
    optimizer: Literal["adam", "sgd"]
    task: Type[Task]
    learner: Type[Learner]
    n_predictors: int
    n_delegators: int
    delegators_mixing: Literal["sum", "product"]
    architecture: Architecture
    param_budget: int | None = None
    

@struct.dataclass
class ForwardReturn:
    delegations: jax.Array # (BS, n_delegators, n_predictor)
    predictions: jax.Array # (BS, n_predictors, n_output)


@struct.dataclass
class ForwardArgs:
    key: jax.Array
    x: jax.Array

class Model(Protocol):
    def apply(self, params: dict, args: ForwardArgs) -> ForwardReturn:
        ...


@struct.dataclass
class InOutData:
    
    x: jax.Array
    y: jax.Array