from __future__ import annotations

from abc import abstractmethod, ABC
import copy
from pathlib import Path
from typing import Literal, Self
import numpy as np
from optuna import Trial
import time
from scipy.stats import kendalltau

def kendalltau_metric(x: np.ndarray, y: np.ndarray) -> float:
    tau, _ = kendalltau(x, y)
    return float(tau)

class Metrics:

    def __init__(self, **metrics):
        self.metrics = {name: [] for name in metrics.keys()}
        self.history = {name: [] for name in metrics.keys()}

    def metrics_unused(self) -> bool:

        return (
            all([ len(metric) == 0 for metric in self.metrics.values() ]) and
            all([ len(history) == 0 for history in self.history.values() ])
        )

    def add_metrics(self, **metrics):

        assert self.metrics_unused(), "Adding metrics after usage"

        new_metrics = {name: [] for name in metrics.keys()}
        new_histories = {name: [] for name in metrics.keys()}

        self.metrics |= new_metrics
        self.history |= new_histories


    def push(self, **metrics):
        for name, value in metrics.items():
            self.metrics[name].append(value)

    def validate_lengths(self):

        metrics = list(self.metrics.values())
        length = len(metrics[0])

        if not all([ len(metric) == length for metric in metrics ]):
                raise ValueError("Length mismatch between the metrics")

    def reset(self):

        self.validate_lengths()

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

        self._track_confidence: bool = task == "protein"

        self._train_end: float | None = None
        self._train_start: float | None = None
        self._test_metrics: dict | None = None


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

    @abstractmethod
    def calculate_confidence_and_errors(self, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        ...

    def evaluate_confidence_metrics(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()

        confidences, error = self.calculate_confidence_and_errors(x, y)

        i_sorted = np.argsort(error)
        error = error[i_sorted]

        results = dict()
        for name, confidence in confidences.items():

            confidence = confidence[i_sorted]
            cmin, cmax = np.min(confidence), np.max(confidence)

            if cmin < -0.2 or cmax > 1.2:
                confidence_norm = (confidence - cmin) / (cmax - cmin)
            else:
                confidence_norm = confidence

            uncertainty = -confidence
            score = kendalltau_metric(error, uncertainty)
            confidence_std = np.std(confidence)
            # ax.scatter(error, confidence_norm, label=f"{name} {score:0.3f}")
            results[f"{name}_kendall"] = score
            results[f"{name}_spread"] = confidence_std

        # ax.legend()
        # plt.show()

        self._test_metrics |= results

        return results


    def get_task_type(self) -> Literal["classification", "regression"]:
        if self.task == "cifar10":
            return "classification"
        elif self.task == "protein":
            return "regression"

        raise ValueError(f"Invalid task {self.task}")

    def calc_task_metric(self, y_hat: np.ndarray, y: np.ndarray, reduction: Literal["batch", "metric"] = "metric") -> float:

        task_type = self.get_task_type()


        if task_type == "classification":
            same_mask_float = self.metric_accuracy(y_hat, y)

            if reduction == "metric":
                return np.mean(same_mask_float)
            elif reduction == "batch":
                others = tuple(range(1, same_mask_float.ndim))
                return np.mean(same_mask_float, axis=others)


        elif task_type == "regression":
            se_error = self.metric_se(y_hat, y)

            if reduction == "metric":
                return np.sqrt(np.mean(se_error))
            elif reduction == "batch":
                others = tuple(range(1, se_error.ndim))
                return np.sqrt(np.mean(se_error, axis=others))



        return same_mask_float

    def get_task_metric_name(self) -> str:

        task_type = self.get_task_type()
        if task_type == "classification":
            return "accuracy"
        elif task_type == "regression":
            return "rmse"



    @staticmethod
    def metric_se(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.square(y_hat - y) # Per sample

    @staticmethod
    def metric_accuracy(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (y_hat == y).astype(np.float32) # Per sample

    def save_metrics(self):

        metrics = self._test_metrics

        if self.folder is not None:

            metrics_file = self.folder / f"{self.name()}_test_metrics.txt"
            content = "\n".join(f"{k}={v}" for k, v in metrics.items())
            with open(metrics_file, "w") as f:
                f.write(content)

    def set_test_metrics(self, **metrics):

        metrics = copy.deepcopy(metrics)
        metrics["nbytes"] = self.get_size_nbytes()
        metrics["train_time"] = self._train_end - self._train_start

        self._test_metrics = metrics


    # @classmethod
    # @abstractmethod
    # def hyperoptimize_step(cls, trial: Trial) -> float:
    #     ...


    @classmethod
    def name(cls):
        return cls.__name__