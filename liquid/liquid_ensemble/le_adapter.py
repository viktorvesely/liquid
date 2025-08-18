
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

from .le_cifar10architecture import LeLongCifar, LeBlockCifar
from .le_regression import LongRegression

type LeModel = LeLongCifar | LeBlockCifar | LongRegression

def task_to_class(task: str, arch_type: str) -> LeModel:
    if task == "protein":
        return LongRegression

    elif task == "cifar10":
            ModelClass = LeLongCifar if arch_type == "long" else LeBlockCifar
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

        self.model: LeModel = None
        self.optimizer: AdamW = None
        self.arch_type: Literal["long", "block"] = self.child_type()
        self.n_all_params: int = None

    def child_type(self) -> str:
        raise NotImplementedError()

    def init_model(
        self,
        model: LeModel = None,
        model_kwargs: dict = None
    ):

        if model is None:

            ModelClass = task_to_class(self.task, self.arch_type)

            init_kwargs = {
                ("n_input" if self.task == "protein" else "in_channels"): self.n_input,
                "n_output": self.n_output
            }
            all_kwargs = model_kwargs | init_kwargs

            model = ModelClass(
                **all_kwargs
            )

        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.n_all_params = sum(p.numel() for p in self.model.parameters())

    def get_nn(self):
        return self.model, self.optimizer

    def on_dataset_start(self):
        self.train_metrics = Metrics(loss=None, power_entropy=None, speaker_entropy=None)
        self.valid_metric = Metrics.empty_like(self.train_metrics, **{self.get_task_metric_name(): None})

    def p_active_parameters_batch(self, x_batch: torch.Tensor) -> torch.Tensor:

        p_actives = []
        for le_layer in self.model.get_le_layers():
            # (batch,)
            p_active = le_layer.p_active_parameters()
            p_actives.append(p_active)

        # (batch, le_layers)
        p_actives = torch.stack(p_actives)

        # (batch,)
        return p_actives.mean(dim=1)

    def auxiliary_loss(self, *args, **kwargs):
        return torch.mean(torch.stack(tuple(le.auxiliary_loss() for le in self.model.get_le_layers())))

    def power_entropy(self):
        return torch.mean(torch.stack(tuple(le.power_entropy() for le in self.model.get_le_layers())))

    def speaker_entropy(self):
        return torch.mean(torch.stack(tuple(le.speaker_entropy() for le in self.model.get_le_layers())))

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


    def on_end(self, x_val: np.ndarray, y_val: np.ndarray):

        super().on_end(x_val, y_val)

        speaker_entropies = []
        power_entropies = []
        def step(model: nn.Module, x, yhat):
            speaker_entropies.append(self.speaker_entropy().item())
            power_entropies.append(self.power_entropy().item())

        yhat = self.inference(x_val, batch_size=self.last_bs, on_batch=step, regression_norm_x=False)
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
        task = constructor["task"]

        instance = cls(**constructor)

        model = task_to_class(task=task, arch_type=instance.arch_type).apply_constructor(model)
        instance.init_model(model=model)

        return instance



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