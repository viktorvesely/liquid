
from pathlib import Path
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, RMSprop
from typing import Literal, Self

from ..nn_adapter import NNAdapter
from ..adapter import Metrics

from .le_cifar10architecture import LongCifar10, BlockCifar10
from .le_regression import LongRegression


def task_to_class(task: str, arch_type: str):
    if task == "protein":
        return LongRegression

    elif task == "cifar10":
            ModelClass = LongCifar10 if arch_type == "long" else BlockCifar10
            return ModelClass

class LiquidBase(NNAdapter):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        task: str,
        lr: float = 1e-3,
        ):
        super().__init__(lr=lr, n_input=n_input, n_output=n_output, folder=folder, task=task)

        self.model: LongCifar10 | BlockCifar10 | LongRegression = None
        self.optimizer: AdamW = None
        self.train_metrics: Metrics = None
        self.valid_metric: Metrics = None
        self.arch_type: Literal["long", "block"] = self.child_type()

    def child_type(self) -> str:
        raise NotImplementedError()

    def init_model(
        self,
        model: LongCifar10 | BlockCifar10 = None,
        model_kwargs: dict = None
    ):

        if model is None:

            if self.task == "protein":

                model = LongRegression(
                    n_input=self.n_input,
                    n_output=self.n_output,
                    **model_kwargs
                )

            elif self.task == "cifar10":

                ModelClass = LongCifar10 if self.arch_type == "long" else BlockCifar10
                model = ModelClass(
                    in_channels=self.n_input,
                    n_output=self.n_output,
                    **model_kwargs
                )

        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

    def get_nn(self):
        return self.model, self.optimizer


    def on_train(self):
        self.train_metrics = Metrics(loss=None, power_entropy=None, speaker_entropy=None)
        self.valid_metric = Metrics.empty_like(self.train_metrics, **{self.get_task_metric_name(): None})

    def auxiliary_loss(self, *args, **kwargs):
        return torch.mean(torch.stack(tuple(le.auxiliary_loss() for le in self.model.get_le_layers())))

    def power_entropy(self):
        return torch.mean(torch.stack(tuple(le.power_entropy() for le in self.model.get_le_layers())))

    def speaker_entropy(self):
        return torch.mean(torch.stack(tuple(le.speaker_entropy() for le in self.model.get_le_layers())))

    def on_batch(self, model, x_batch, y_batch, yhat_batch, loss, valid):

        metrics = self.valid_metric if valid else self.train_metrics
        metricor = self.model

        with torch.no_grad():
            power_entropy = metricor.power_entropy()
            speaker_entropy = metricor.speaker_entropy()
            metrics.push(loss=loss.item(), power_entropy=power_entropy.item(), speaker_entropy=speaker_entropy.item())

            if valid:
                metric = self.calc_task_metric(y_batch, yhat_batch)
                self.push_task_metric(metric, metrics)

    def on_epoch(self, epoch: int):

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
            speaker_entropies.append(self.model.speaker_entropy().item())
            power_entropies.append(self.model.power_entropy().item())

        yhat = self.inference(x_val, batch_size=self.last_bs, on_batch=step)

        power_entropy = np.mean(power_entropies)
        speaker_entropy = np.mean(speaker_entropies)

        if self.task in {"cifar10"}:
            accuracy = self.calc_task_metric(y_val, yhat)
            self.save_test_metrics(accuracy=accuracy, power_entropy=power_entropy, speaker_entropy=speaker_entropy)
            return accuracy

        elif self.task in {"protein"}:
            rmse = self.calc_task_metric(y_val, yhat)
            self.save_test_metrics(RMSE=rmse, power_entropy=power_entropy, speaker_entropy=speaker_entropy)
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
        task = constructor["task"]

        instance = cls(**constructor)

        model = task_to_class(task=task, arch_type=instance.arch_type).apply_constructor(model)
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

    @classmethod
    def hyperoptimize_step(cls, trial: optuna.Trial):


        lr = trial.sug
        n_citizens: int = 10,
        solver:  Literal["sink_one", "sink_many"] = "sink_one",
        load_distribution_lambda: float = 0.0,
        specialization_lambda: float = 0.0,


class LiquidLong(LiquidBase):

    def child_type(self):
        return "long"

    @classmethod
    def name(cls):
        return "LongLiquid"

class LiquidBlock(LiquidBase):

    def child_type(self):
        return "block"

    @classmethod
    def name(cls):
        return "BlockLiquid"