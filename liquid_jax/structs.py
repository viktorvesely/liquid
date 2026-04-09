
from dataclasses import dataclass
from typing import Literal, Type
from task_base import Task
from learner_base import Learner

@dataclass(frozen=True)
class TrainParams:
    
    batch_size: int
    preload_batches_to_gpu: int
    valid_batches: int
    epochs: int
    lr: float
    optimizer: Literal["adam", "sgd"]
    performance_loss: Literal["ce"]
    task: Type[Task]
    learner: Type[Learner]
    n_models_in_ensemble: int
    param_budget: int | None = None
    
