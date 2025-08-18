from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Self
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import torch.optim as optim

from .adapter import Adapter, Metrics

from .globals import config
from .visualizer import loss_landscape_2d, plot_landscape_3d, make_two_directions
from matplotlib import pyplot as plt


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
        # self.device = torch.device("cpu")

        self.x_norm = None
        self.y_norm = None
        self.verbose: int = 0

        self.train_metrics: Metrics = None
        self.valid_metric: Metrics = None
        self._best_valid_metric = (-float("inf")) if (self.get_task_type() == "classification") else float("inf")


    def register_valid_metric(self, metric: float):

        if self.get_task_type() == "classification":
            self._best_valid_metric = max(self._best_valid_metric, metric)
        else:
            self._best_valid_metric = min(self._best_valid_metric, metric)

    @abstractmethod
    def get_nn(self) -> tuple[nn.Module, optim.Optimizer]:
        ...

    def p_active_parameters_batch(
        self,
        x_batch: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones(x_batch.shape[0], device=x_batch.device, dtype=torch.float)

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



    def on_dataset_start(self):
        self.train_metrics = Metrics(loss=None)
        self.valid_metric = Metrics.empty_like(self.train_metrics, **{self.get_task_metric_name(): None})

    def on_epoch(self, epoch: int):

        if self.verbose > 0:
            print(f"-------{self.name()} Epoch {epoch}-----------")
            print(f"Train: {self.train_metrics}")
            print(f"Valid: {self.valid_metric}\n")

        self.train_metrics.reset()
        self.valid_metric.reset()

    def on_end(self, x_val: np.ndarray, y_val: np.ndarray) -> Any:

        folder = self.folder
        save_files = self.folder is not None

        if save_files:
            self.valid_metric.save_histories(folder, prefix=self.name())


    def auxiliary_loss(
            self,
            model: nn.Module,
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            yhat_batch: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(1, dtype=x_batch.dtype, device=x_batch.device)

    def get_size_nparams(self) -> int:

        model, _ = self.get_nn()
        nparams = 0

        for params in model.parameters():
            nparams += params.nelement()

        return nparams

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
            regression_norm_x: bool = True,
            classification_output_labels = True,
            **_
        ) -> np.ndarray:


        is_classification = self.get_task_type() == "classification"
        is_regression = self.get_task_type() == "regression"

        if is_regression and regression_norm_x:
            x = self.norm_x(x)

        self.on_dataset_start()
        model, _ = self.get_nn()
        dataset = TensorDataset(torch.tensor(x))
        loader = DataLoader(dataset, batch_size)

        model.eval()

        ys = []

        with torch.no_grad():
            for (x_batch,) in tqdm.tqdm(loader, total=len(loader), disable=(verbose < 1)):

                x_batch = x_batch.to(self.device)
                yhat_batch = model(x_batch)

                if is_classification and classification_output_labels:
                    ys.append(torch.argmax(yhat_batch, dim=1).cpu().numpy())
                else:
                    ys.append(yhat_batch.cpu().numpy())

                if callable(on_batch):
                    on_batch(model, x_batch, yhat_batch)

        return np.concatenate(ys, axis=0)


    def evaluate_p_active_params(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        p_active = []

        def batch(model: nn.Module, x_batch: torch.Tensor, yhat_batch: torch.Tensor):
            p_active_batch = model.p_active_parameters_batch(x_batch)
            p_active.append(p_active_batch.detach().cpu().numpy())

        self.inference(x, batch_size=self.last_bs, on_batch=batch)

        p_active = np.concatenate(p_active)

        p_active_metrics = {}
        q = np.linspace(0, 1, num=7, endpoint=True)
        for qv in zip(q, np.quantile(p_active, q), strict=True):
            p_active_metrics[f"p_active_q_{q:.2f}"] = qv

        self._test_metrics |= p_active_metrics

    def calculate_confidence_and_errors(self, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:

        if self.get_task_type() == "regression":
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


        if self.get_task_type() == "regression":
            y = self.denorm_y(y)
            yhat = self.denorm_y(yhat)

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

        if self.get_task_type() == "regression":

            self.set_norm(x, y)

            x = self.norm_x(x)
            y = self.norm_y(y)

            x_val = self.norm_x(x_val)
            y_val = self.norm_y(y_val)


        self.last_bs = batch_size
        self.on_dataset_start()
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

            if config.display_loss_landscape and (e % config.ll_every == 0) and (e >= config.ll_start):
                self.visualize_landscape(model, train_loader, criterion, resolution=config.ll_resolution, distance=config.ll_distance)

            self.save()
            self.on_epoch(e)

            valid_time_penalty += self.now() - valid_start

        self._train_end = self.now() - valid_time_penalty

        return self.on_end(x_val, y_val)

    def visualize_landscape(
            self,
            model: nn.Module,
            loader: DataLoader,
            criterion: nn.Module,
            distance: float = 0.5,
            resolution: int = 25
        ):

        was_training = model.training

        if was_training:
            model.eval()


        d1, d2 = make_two_directions(model)
        X, Y, Z = loss_landscape_2d(
            model,
            criterion,
            loader,
            device=self.device,
            max_batches=1,
            radius_alpha=distance,
            radius_beta=distance,
            grid_n=resolution,
            d1=d1,
            d2=d2
        )
        plot_landscape_3d(X, Y, Z, title="Normal")
        config.make_delegation_uniform = True

        X, Y, Z = loss_landscape_2d(
            model,
            criterion,
            loader,
            device=self.device,
            max_batches=1,
            radius_alpha=distance,
            radius_beta=distance,
            grid_n=resolution,
            d1=d1,
            d2=d2
        )
        plot_landscape_3d(X, Y, Z, title="Uniform")
        plt.show()
        config.make_delegation_uniform = False

        if was_training:
            model.train()


    def save(self):
        folder = self.folder

        if folder is None:
            return

        file = folder / f"{self.name()}.pt"

        constructor = self.get_constructor()
        constructor["__optimizer_state_dict"] = self.optimizer.state_dict()
        constructor["__model_state_dict"] =  self.model.state_dict()

        torch.save(constructor, file)


    @classmethod
    def load(cls, folder: Path) -> Self:
        constructor = torch.load(folder / f"{cls.name()}.pt", weights_only=False)

        msd = constructor.pop("__model_state_dict")
        osd = constructor.pop("__optimizer_state_dict")

        instance = cls.apply_constructor(constructor)

        instance.model.load_state_dict(msd)
        instance.optimizer.load_state_dict(osd)

        return instance

