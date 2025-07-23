from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from typing import Self

from ..nn_adapter import NNAdapter
from ..adapter import Metrics

from .moe_cifar10architecture import MoeCifar10
from .moe_regression import LongRegression

class Moe(NNAdapter):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        task: str,
        lr: float = 1e-3,
        ):
        super().__init__(lr=lr, n_input=n_input, n_output=n_output, folder=folder, task=task)

        self.model: MoeCifar10 | LongRegression = None
        self.optimizer: AdamW = None
        self.train_metrics: Metrics = None
        self.valid_metric: Metrics = None


    def init_model(
        self,
        model: MoeCifar10 = None,
        model_kwargs: dict = None
    ):

        if model is None:

            if self.task == "cifar10":
                model = MoeCifar10(
                    in_channels=self.n_input,
                    n_output=self.n_output,
                    **model_kwargs
                )
            else:
                model = LongRegression(
                    n_input=self.n_input,
                    n_output=self.n_output,
                    **model_kwargs
                )

        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def get_nn(self):
        return self.model, self.optimizer


    def on_train(self):
        self.train_metrics = Metrics(loss=None, power_entropy=None, speaker_entropy=None)
        self.valid_metric = Metrics.empty_like(self.train_metrics, **{self.get_task_metric_name(): None})

    def auxiliary_loss(self, *args, **kwargs):
        return torch.mean(torch.stack(tuple(moe.auxiliary_loss() for moe in self.model.get_moe_layers())))

    def power_entropy(self):
        return torch.mean(torch.stack(tuple(moe.power_entropy() for moe in self.model.get_moe_layers())))

    def speaker_entropy(self):
        return torch.mean(torch.stack(tuple(moe.speaker_entropy() for moe in self.model.get_moe_layers())))

    def on_batch(self, model, x_batch, y_batch, yhat_batch, loss, valid):

        metrics = self.valid_metric if valid else self.train_metrics

        with torch.no_grad():
            power_entropy = self.power_entropy()
            speaker_entropy = self.speaker_entropy()
            metrics.push(loss=loss.item(), power_entropy=power_entropy.item(), speaker_entropy=speaker_entropy.item())

            if valid:
                metric = self.calc_task_metric(yhat_batch, y_batch)
                self.push_task_metric(metric, metrics)
                self.register_valid_metric(metric)

    def on_epoch(self, epoch: int):

        if self.verbose > 0:
            print(f"\n--------{self.name()} Epoch {epoch}-----------")
            print(f"Train: {self.train_metrics}")
            print(f"Valid: {self.valid_metric}")

        self.train_metrics.reset()
        self.valid_metric.reset()

    def on_end(self, x_val: np.ndarray, y_val: np.ndarray):


        folder = self.folder
        save_files = self.folder is not None

        if save_files:
            self.valid_metric.save_histories(folder, prefix=self.name())

        speaker_entropies = []
        power_entropies = []
        def step(model: nn.Module, x, yhat):
            speaker_entropies.append(self.speaker_entropy().item())
            power_entropies.append(self.power_entropy().item())

        yhat = self.inference(x_val, batch_size=self.last_bs, on_batch=step, norm_x=False)
        power_entropy = np.mean(power_entropies)
        speaker_entropy = np.mean(speaker_entropies)

        task_type = self.get_task_type()
        if task_type == "classification":
            accuracy = self.calc_task_metric(yhat, y_val)
            self.set_test_metrics(accuracy=accuracy, power_entropy=power_entropy, speaker_entropy=speaker_entropy, best_accuracy=self._best_valid_metric)
            return accuracy

        elif task_type == "regression":
            rmse = self.calc_task_metric(yhat, y_val)
            self.set_test_metrics(rmse=rmse, power_entropy=power_entropy, speaker_entropy=speaker_entropy,  best_rmse=self._best_valid_metric)
            return rmse

    def get_constructor(self) -> dict:

        constructor = {
            'n_input': self.n_input,
            'n_output': self.n_output,
            'folder': str(self.folder.resolve()),
            'lr': self.lr,
            'model': self.model.get_constructor(),
            "task": self.task
        }

        return constructor

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        constructor["folder"] = Path(constructor["folder"])
        model = constructor.pop("model")

        instance = cls(**constructor)

        if instance.taks == "cifar10":
            model = MoeCifar10.apply_constructor(model)
        else:
            model = LongRegression.apply_constructor(model)

        instance.init_model(model=model)

        return instance

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