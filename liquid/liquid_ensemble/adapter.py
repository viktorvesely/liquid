
from pathlib import Path
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import normalized_mutual_info_score
from typing import Literal, Self

from ..nn_adapter import NNAdapter
from ..adapter import Metrics

from .cifar10architecture import LongCifar10


def get_regions_classes(X: torch.Tensor) -> torch.Tensor:

    c = torch.zeros(X.shape[0], dtype=torch.int, device=X.device)
    x, y = X[:, 0], X[:, 1]
    mask_ul = (x < 0) & (y > 0)
    mask_ur = (x >= 0) & (y > 0)
    mask_ll = (x < 0) & (y <= 0)
    mask_lr = (x >= 0) & (y <= 0)

    c[mask_ul] = 0
    c[mask_ur] = 1
    c[mask_ll] = 2
    c[mask_lr] = 3

    return c

class Liquid(NNAdapter):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        lr: float = 1e-3,
        ):
        super().__init__(lr=lr, n_input=n_input, n_output=n_output, folder=folder)

        self.model: LongCifar10 = None
        self.optimizer: AdamW = None
        self.train_metrics: Metrics = None
        self.valid_metric: Metrics = None

    def init_model(
        self,
        model: LongCifar10 = None,
        model_kwargs: dict = None
    ):

        if model is None:
            model = LongCifar10(
                in_channels=self.n_input,
                n_output=self.n_output,
                **model_kwargs
            )

        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def get_nn(self):
        return self.model, self.optimizer

    def auxiliary_loss(self, *args, **kwargs):
        return self.model.auxiliary_loss(*args, **kwargs)

    def on_train(self):
        self.train_metrics = Metrics(loss=None, power_entropy=None, speaker_entropy=None)
        self.valid_metric = Metrics.empty_like(self.train_metrics, accuracy=None)

    def on_batch(self, model, x_batch, y_batch, yhat_batch, loss, valid):

        metrics = self.valid_metric if valid else self.train_metrics

        le_layer = self.model.le_layer

        with torch.no_grad():
            power_entropy = le_layer.power_entropy()
            speaker_entropy = le_layer.speaker_entropy()
            metrics.push(loss=loss.item(), power_entropy=power_entropy.item(), speaker_entropy=speaker_entropy.item())

            if valid:
                labels_hat = torch.argmax(yhat_batch, 1)
                correct = (labels_hat == y_batch).float()
                accuracy = correct.mean()
                metrics.push(accuracy=accuracy.item())

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

        Ps = []

        def step(model: nn.Module, x, yhat):
            Ps.append(self.model.le_layer.last_power.cpu().numpy())

        yhat = self.inference(x_val, batch_size=self.last_bs, on_batch=step)
        Ps = np.concat(Ps, axis=0)

        hatlabel = np.argmax(yhat, axis=1)
        accuracy = (y_val  == hatlabel).astype(np.float32).mean()

        if self.synthetic:
            chair = np.argmax(Ps, axis=1)
            region_classes = get_regions_classes(torch.tensor(x_val))
            region_nmi = normalized_mutual_info_score(
                chair, region_classes.cpu().numpy()
            )
        else:
            region_nmi = float("nan")

        if not isinstance(region_nmi, float):
            region_nmi = region_nmi.item()

        power_entropy = self.model.le_layer.power_entropy(torch.tensor(Ps)).item()
        speaker_entropy = self.model.le_layer.speaker_entropy(torch.tensor(Ps)).item()

        self.save_test_metrics(accuracy=accuracy, power_entropy=power_entropy, speaker_entropy=speaker_entropy, region_nmi=region_nmi)

        return accuracy



    def get_constructor(self) -> dict:

        constructor = {
            'n_input': self.n_input,
            'n_output': self.n_output,
            'folder': str(self.folder.resolve()),
            'lr': self.lr,
            'model': self.model.get_constructor(),
        }

        return constructor

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:

        constructor["folder"] = Path(constructor["folder"])
        model = constructor.pop("model")

        instance = cls(**constructor)

        model = LongCifar10.apply_constructor(model)
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

