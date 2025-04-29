from __future__ import annotations

from abc import abstractmethod, ABC
from pathlib import Path
from typing import Self
import numpy as np


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

    def save_histories(self, folder: Path):
        for name, values in self.history.items():
            with open(folder / f"{name}.txt", "w") as f:
                f.write("\n".join(map(lambda x: f"{x:.4f}", values)))

class Adapter(ABC):


    def __init__(self, folder: Path):
        super().__init__()

        self.folder = folder


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

    @abstractmethod
    def get_size_nbytes(self) -> int:
        ...

    @abstractmethod
    def save(self, folder: Path | None = None):
        ...

    @classmethod
    @abstractmethod
    def load(folder: Path) -> Self:
        ...

    @abstractmethod
    def init_model(**kwargs):
        ...