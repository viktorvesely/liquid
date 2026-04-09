from abc import ABC, abstractmethod
from typing import Literal

import jax


class Task[TrainData](ABC):
    
    @staticmethod
    @abstractmethod
    def load_cpu(
        split: Literal["train", "test"]
    ) -> TrainData:
        ...

    @staticmethod
    @abstractmethod
    def get_xy(
        data: TrainData
    ) -> tuple[jax.Array, jax.Array]:
        ...

    @staticmethod
    @abstractmethod
    def task_type() -> Literal["classification", "regression"]:
        ...

    @staticmethod
    @abstractmethod
    def out_dim() -> int:
        ...