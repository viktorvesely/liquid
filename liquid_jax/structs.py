
from dataclasses import dataclass
from typing import Literal, Type

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
class TrainReturn:
    delegation: jax.Array # (BS, n_delegators, n_predictor)
    power: jax.Array # (BS, n_predictors)
    ys: jax.Array = None # (BS, n_predictors, n_output)



