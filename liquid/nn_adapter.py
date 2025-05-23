from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        super().__init__(n_input=n_input, n_output=n_output, folder=folder)

        self.lr = lr
        self.last_bs: int = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task

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
    def denorm_features(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray):
        return x_norm * std + mean

    @staticmethod
    def norm_features(x: np.ndarray, mean = None, std = None):

        if mean is None:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8

        x_norm = (x - mean) / std
        return x_norm, mean, std

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
            x, _, _ = self.norm_features(x, self.x_mean, self.x_std)

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


    def get_task_metric_name(self) -> str:
        if self.task in {"cifar10"}:
            return "accuracy"
        elif self.task in {"protein"}:
            return "RMSE"

    def calc_task_metric(self, y_batch, yhat_batch):

        if isinstance(y_batch, torch.Tensor):
            y_batch = y_batch.cpu().numpy()
            yhat_batch = yhat_batch.cpu().numpy()

        if self.task in {"cifar10"}:
            labels_hat = np.argmax(yhat_batch, 1)
            correct = (labels_hat == y_batch).astype(np.float32)
            accuracy = correct.mean()
            return accuracy

        elif self.task in {"protein"}:

            yhat_batch = self.denorm_features(yhat_batch, self.y_mean, self.y_std)
            y_batch = self.denorm_features(y_batch, self.y_mean, self.y_std)
            mse = np.square(y_batch - yhat_batch).mean()
            return np.sqrt(mse)

    def push_task_metric(self, metric, val_metrics):

        if self.task in {"cifar10"}:
            val_metrics.push(accuracy=metric)
        elif self.task in {"protein"}:
            val_metrics.push(RMSE=metric)


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

        if self.task == "protein":
            x, self.x_mean, self.x_std = self.norm_features(x)
            y, self.y_mean, self.y_std = self.norm_features(y)

            x_val, _, _ = self.norm_features(x_val)
            y_val, _, _ = self.norm_features(y_val)


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

        return self.on_end(x_val, y_val)

