from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import torch.optim as optim

from .adapter import Adapter


class NNAdapter(Adapter):

    synthetic: bool = True

    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        lr: float = 1e-3
    ):
        super().__init__(n_input=n_input, n_output=n_output, folder=folder)

        self.lr = lr
        self.last_bs: int = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    def inference(
            self,
            x: np.ndarray,
            *,
            batch_size: int = 512,
            verbose: int = 0,
            on_batch: Callable[[nn.Module, torch.Tensor, torch.Tensor], None] = None,
            **_
        ) -> np.ndarray:


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

        self.last_bs = batch_size
        self.on_train()
        model, optimizer = self.get_nn()

        train_dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        valid_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size)

        criterion = nn.CrossEntropyLoss()

        for e in range(epoch):

            model.train()

            for x_batch, y_batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", disable=(verbose < 1)):

                x_batch = x_batch.to("cuda")
                y_batch = y_batch.to("cuda")

                yhat_batch = model(x_batch)

                loss = criterion(yhat_batch, y_batch) + self.auxiliary_loss(model, x_batch, y_batch, yhat_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.on_batch(model, x_batch, y_batch, yhat_batch, loss, valid=False)

            model.eval()

            with torch.no_grad():
                for x_batch, y_batch in tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", disable=(verbose < 1)):

                    x_batch = x_batch.to("cuda")
                    y_batch = y_batch.to("cuda")

                    yhat_batch = model(x_batch)
                    loss = criterion(yhat_batch, y_batch) + self.auxiliary_loss(model, x_batch, y_batch, yhat_batch)

                    self.on_batch(model, x_batch, y_batch, yhat_batch, loss, valid=True)

            self.save()
            self.on_epoch(e)

        return self.on_end(x_val, y_val)

