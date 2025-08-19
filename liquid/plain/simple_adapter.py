from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from typing import Self

from ..nn_adapter import NNAdapter
from ..adapter import Metrics

from .cifar10 import SimpleCifar

SimpleModel = SimpleCifar

class SimpleNN(NNAdapter):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        task: str,
        lr: float = 1e-3,
        ):
        super().__init__(lr=lr, n_input=n_input, n_output=n_output, folder=folder, task=task)

        self.model: SimpleModel  = None
        self.optimizer: AdamW = None

    def init_model(
        self,
        model: SimpleModel = None,
        model_kwargs: dict = None
    ):

        if model is None:

            if self.task == "protein":

              raise NotImplementedError("Simple NN for protein")

            elif self.task == "cifar10":

                model = SimpleCifar(
                    in_channels=self.n_input,
                    n_output=self.n_output,
                    **model_kwargs
                )

        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr,  weight_decay=0.01)

    def get_nn(self):
        return self.model, self.optimizer

    def on_batch(self, model, x_batch, y_batch, yhat_batch, loss, valid):

        metrics = self.valid_metric if valid else self.train_metrics

        with torch.no_grad():
            metrics.push(loss=loss.item())

            if valid:
                metric = self.calc_task_metric(yhat_batch, y_batch)
                self.push_task_metric(metric, metrics)
                self.register_valid_metric(metric)


    def on_end(self, x_val: np.ndarray, y_val: np.ndarray):

        yhat = self.inference(x_val, batch_size=self.last_bs)

        task_type = self.get_task_type()
        if task_type == "classification":
            accuracy = self.calc_task_metric(yhat, y_val)
            self.set_test_metrics(accuracy=accuracy, best_accuracy=self._best_valid_metric)
            return accuracy

        elif task_type == "regression":
            rmse = self.calc_task_metric(yhat, y_val)
            self.set_test_metrics(rmse=rmse, best_rmse=self._best_valid_metric)
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

        if instance.task == "cifar10":
            model = SimpleCifar.apply_constructor(model)
        else:
            raise NotImplementedError("Only cifar for simple")

        instance.init_model(model=model)

        return instance


