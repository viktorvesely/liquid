from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import torch.optim as optim

from .adapter import Adapter


class NNAdapter(Adapter):

    synthetic: bool = False

    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        task: str,
        lr: float = 1e-3
    ):
        super().__init__(n_input=n_input, n_output=n_output, folder=folder, task=task)

        self.lr = lr
        self.last_bs: int = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_norm = None
        self.y_norm = None
        self.verbose: int = 0

        self._best_valid_metric = (-float("inf")) if (self.get_task_type() == "classification") else float("inf")


    def register_valid_metric(self, metric: float):

        if self.get_task_type() == "classification":
            self._best_valid_metric = max(self._best_valid_metric, metric)
        else:
            self._best_valid_metric = min(self._best_valid_metric, metric)

    @abstractmethod
    def get_nn(self) -> tuple[nn.Module, optim.Optimizer]:
        ...

    def on_batch(
            self,
            model: nn.Module,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            yhat_batch: torch.Tensor,
            loss: torch.Tensor,
            valid: bool
            ):
        ...

    def on_train(self):
        ...

    def on_epoch(self, epoch: int):
        ...

    def on_end(self, x_val: np.ndarray, y_val: np.ndarray) -> Any:
        ...

    def auxiliary_loss(
            self,
            model: nn.Module,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            yhat_batch: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(1, dtype=x_batch.dtype, device=x_batch.device)

    def get_size_nbytes(self) -> int:

        model, _ = self.get_nn()
        nbytes = 0

        for params in model.parameters():
            nbytes += params.element_size() * params.nelement()

        return nbytes

    @staticmethod
    def _denorm(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray):
        return x_norm * std + mean

    @staticmethod
    def get_norm(x: np.ndarray):
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        return mean, std

    @staticmethod
    def _norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
        x_norm = (x - mean) / std
        return x_norm

    def set_norm(self, x: np.ndarray, y: np.ndarray):
        self.x_norm = self.get_norm(x)
        self.y_norm = self.get_norm(y)

    def norm_y(self, y: np.ndarray):
        return self._norm(y, *self.y_norm)

    def norm_x(self, x: np.ndarray):
        return self._norm(x, *self.x_norm)

    def denorm_y(self, y: np.ndarray):
        return self._denorm(y, *self.y_norm)

    def denorm_x(self, x: np.ndarray):
        return self._denorm(x, *self.x_norm)

    def inference(
            self,
            x: np.ndarray,
            *,
            batch_size: int = 512,
            verbose: int = 0,
            on_batch: Callable[[nn.Module, torch.Tensor, torch.Tensor], None] = None,
            **_
        ) -> np.ndarray:


        if self.task == "protein":
            x = self.norm_x(x)

        self.on_train()
        model, _ = self.get_nn()
        dataset = TensorDataset(torch.tensor(x))
        loader = DataLoader(dataset, batch_size)

        model.eval()

        ys = []

        with torch.no_grad():
            for (x_batch,) in tqdm.tqdm(loader, total=len(loader), disable=(verbose < 1)):

                x_batch = x_batch.to("cuda")
                yhat_batch = model(x_batch)

                ys.append(yhat_batch.cpu().numpy())

                if callable(on_batch):
                    on_batch(model, x_batch, yhat_batch)

        return np.concat(ys, axis=0)


    def calculate_confidence_and_errors(self, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:

        if self.task == "protein":
            x = self.norm_x(x)
            y = self.norm_y(y)

        model, _ = self.get_nn()

        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        loader = DataLoader(dataset, self.last_bs)

        confidence = defaultdict(list)
        yhat = []

        model.eval()
        for (x_batch, y_batch) in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                yhat_batch = model(x_batch)
                results = model.calculate_confidence(x_batch)

                for k, v in results.items():
                    confidence[k].append(v.cpu())

                yhat.append(yhat_batch.cpu())

        dataset = None
        loader = None

        for k, v in confidence.items():
            v = torch.cat(v, dim=0)
            confidence[k] = v.numpy()

        yhat = torch.cat(yhat, dim=0).numpy()

        return confidence, self.calc_task_metric(yhat, y, reduction="batch")

    def calc_task_metric(self, yhat, y,  reduction: Literal["none", "batch", "metric"] = "metric"):

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if isinstance(yhat, torch.Tensor):
            yhat = yhat.cpu().numpy()

        return super().calc_task_metric(yhat, y, reduction)

    def push_task_metric(self, metric, val_metrics):

        task_type = self.get_task_type()
        if task_type == "classification":
            val_metrics.push(accuracy=metric)
        elif task_type == "regression":
            val_metrics.push(rmse=metric)


    def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            *,
            epoch: int = 150,
            batch_size: int = 512,
            verbose: int = 0,
            **_
        ):

        self.verbose = verbose

        if self.task == "protein":

            self.set_norm(x, y)

            x = self.norm_x(x)
            y = self.norm_y(y)

            x_val = self.norm_x(x_val)
            y_val = self.norm_y(y_val)


        self.last_bs = batch_size
        self.on_train()
        model, optimizer = self.get_nn()

        train_dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        valid_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size)

        if self.task in {"cifar10"}:
            criterion = nn.CrossEntropyLoss()
        elif self.task in {"protein"}:
            criterion = nn.MSELoss()

        valid_time_penalty = 0
        self._train_start = self.now()
        for e in range(epoch):

            model.train()

            for x_batch, y_batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", disable=(verbose < 1)):

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                yhat_batch = model(x_batch)

                loss = criterion(yhat_batch, y_batch) + self.auxiliary_loss(model, x_batch, y_batch, yhat_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.on_batch(model, x_batch, y_batch, yhat_batch, loss, valid=False)


            valid_start = self.now()
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", disable=(verbose < 1)):

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    yhat_batch = model(x_batch)
                    loss = criterion(yhat_batch, y_batch) + self.auxiliary_loss(model, x_batch, y_batch, yhat_batch)

                    self.on_batch(model, x_batch, y_batch, yhat_batch, loss, valid=True)
            self.save()
            self.on_epoch(e)

            valid_time_penalty += self.now() - valid_start

        self._train_end = self.now() - valid_time_penalty

        return self.on_end(x_val, y_val)

