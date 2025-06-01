from __future__ import annotations

from abc import abstractmethod, ABC
import copy
from pathlib import Path
from typing import Literal, Self
import numpy as np
from optuna import Trial
import time

class Metrics:

    def __init__(self, **metrics):
        self.metrics = {name: [] for name in metrics.keys()}
        self.history = {name: [] for name in metrics.keys()}

    def push(self, **metrics):
        for name, value in metrics.items():
            self.metrics[name].append(value)

    def reset(self):
        for name, values in self.metrics.items():
            if len(values) == 0:
                continue

            self.history[name].append(np.mean(values))
            self.metrics[name] = []

    @classmethod
    def empty_like(cls, other: Metrics, **extra):
        return Metrics(**(other.metrics | extra))

    def __repr__(self):
        out = []
        for name, values in self.metrics.items():
            out.append(f"{name}={np.mean(values):.3f}")
        return ", ".join(out)

    def save_histories(self, folder: Path, prefix: str = "liquid"):
        for name, values in self.history.items():
            with open(folder / f"{prefix}_{name}.txt", "w") as f:
                f.write("\n".join(map(lambda x: f"{x:.4f}", values)))

class Adapter(ABC):


    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        task: str
    ):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.folder = folder
        self.task = task

        self._train_end: float | None = None
        self._train_start: float | None = None


    @abstractmethod
    def inference(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs):
        ...


    @staticmethod
    def now():
        return time.perf_counter()

    @abstractmethod
    def get_size_nbytes(self) -> int:
        ...

    @abstractmethod
    def save(self):
        ...

    @classmethod
    @abstractmethod
    def load(cls, folder: Path) -> Self:
        ...

    @abstractmethod
    def init_model(self, **kwargs):
        ...


    def get_task_type(self) -> Literal["classification", "regression"]:
        if self.task == "cifar10":
            return "classification"
        elif self.task == "protein":
            return "regression"

        raise ValueError(f"Invalid task {self.task}")

    def calc_task_metric(self, y_hat: np.ndarray, y: np.ndarray) -> float:

        task_type = self.get_task_type()

        if task_type == "classification":
            return self.metric_accuracy(y_hat, y)
        elif task_type == "regression":
            return self.metric_rmse(y_hat, y)

    def get_task_metric_name(self) -> str:

        task_type = self.get_task_type()
        if task_type == "classification":
            return "accuracy"
        elif task_type == "regression":
            return "rmse"


    @staticmethod
    def metric_rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.square(y_hat - y).mean())

    @staticmethod
    def metric_accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
        return (y_hat == y).astype(np.float32).mean()

    def save_test_metrics(self, **metrics):


        if self.folder is not None:

            metrics = copy.deepcopy(metrics)
            metrics["nbytes"] = self.get_size_nbytes()
            metrics["train_time"] = self._train_end - self._train_start

            metrics_file = self.folder / f"{self.name()}_test_metrics.txt"
            content = "\n".join(f"{k}={v}" for k, v in metrics.items())
            with open(metrics_file, "w") as f:
                f.write(content)


    # @classmethod
    # @abstractmethod
    # def hyperoptimize_step(cls, trial: Trial) -> float:
    #     ...



    @classmethod
    def name(cls):
        return cls.__name__