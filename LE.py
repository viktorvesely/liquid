
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from typing import Literal, Self

from other.nn_adapter import NNAdapter
from other.adapter import Metrics

import utils
from liquid_ensemble import LiquidEnsembleLayer
from synthetic_citizen import DelegatingCitizen
from sklearn.metrics import normalized_mutual_info_score
from synth_metrics import get_regions_classes


class LE(NNAdapter):

    synthetic: bool = True

    def __init__(self, folder: Path):
        super().__init__(folder)

        self.liquid_ensemble: LiquidEnsembleLayer = None
        self.optimizer: AdamW = None
        self.train_metrics: Metrics = None
        self.valid_metric: Metrics = None
        self.n_citizens: int = None
        self.solver: str = None


    def init_model(
        self,
        n_citizens: int = 4,
        load_distribution_lambda: float = 0.0,
        specialization_lambda: float = 0.0,
        solver: Literal["sink_one", "sink_many"] = "sink_one"
    ):

        self.liquid_ensemble = LiquidEnsembleLayer(
            nn.ModuleList([
                DelegatingCitizen(n_citizens, 12, (1, 1, 1), layer_width=50) for _ in range(n_citizens)
            ]),
            load_distribution_lambda=load_distribution_lambda,
            specialization_lambda=specialization_lambda,
            solver=solver
        )

        self.liquid_ensemble = self.liquid_ensemble.to("cuda")

        self.optimizer = AdamW(self.liquid_ensemble.parameters(), lr=2 * 1e-4)
        self.solver = solver
        self.n_citizens = n_citizens

    def get_nn(self):
        return self.liquid_ensemble, self.optimizer

    def auxiliary_loss(self, model, x_batch, y_batch, yhat_batch):
        return self.liquid_ensemble.load_distribution_loss()

    def on_train(self):
        self.train_metrics = Metrics(loss=None, power_entropy=None, speaker_entropy=None)
        self.valid_metric = Metrics.empty_like(self.train_metrics, accuracy=None)

    def on_batch(self, model, x_batch, y_batch, yhat_batch, loss, valid):

        metrics = self.valid_metric if valid else self.train_metrics

        with torch.no_grad():
            power_entropy = self.liquid_ensemble.power_entropy()
            speaker_entropy = self.liquid_ensemble.speaker_entropy()
            metrics.push(loss=loss.item(), power_entropy=power_entropy.item(), speaker_entropy=speaker_entropy.item())

            if valid:
                labels_hat = torch.argmax(yhat_batch, 1)
                correct = (labels_hat == y_batch).float()
                accuracy = correct.mean()
                metrics.push(accuracy=accuracy.item())

    def on_epoch(self, epoch: int):

        print(f"\n--------Epoch {epoch}-----------")
        print(f"Train: {self.train_metrics}")
        print(f"Valid: {self.valid_metric}")

        self.train_metrics.reset()
        self.valid_metric.reset()

    def on_end(self, x_val: np.ndarray, y_val: np.ndarray):

        folder = self.folder
        save_files = self.folder is not None

        if save_files:
            self.valid_metric.save_histories(folder)

        Ps = []

        def step(model: LiquidEnsembleLayer, x, yhat):
            Ps.append(model.last_power.cpu().numpy())

        yhat = self.inference(x_val, batch_size=self.last_bs, on_batch=step)
        Ps = np.concat(Ps, axis=0)

        hatlabel = np.argmax(yhat, axis=1)
        accuracy = (y_val  == hatlabel).astype(np.float32).mean()

        if self.synthetic:
            chair = np.argmax(Ps, axis=1)
            region_classes = get_regions_classes()
            region_nmi = normalized_mutual_info_score(
                chair, region_classes.cpu().numpy()
            )

        if not isinstance(region_nmi, float):
            region_nmi = region_nmi.item()

        power_entropy = self.liquid_ensemble.power_entropy(torch.tensor(Ps)).item()
        speaker_entropy = self.liquid_ensemble.speaker_entropy(torch.tensor(Ps)).item()

        print(save_files, folder)
        if save_files:
            with open(folder / "test_metrics.txt", "w") as f:
                f.write(f"accuracy={accuracy}\npower_entropy={power_entropy}\nspeaker_entropy={speaker_entropy}\nregion_nmi={region_nmi}")

        if self.synthetic:
            return accuracy, power_entropy, speaker_entropy, region_nmi
        else:
            return accuracy, power_entropy, speaker_entropy


    def save(self):
        folder = self.folder

        if folder is None:
            return

        file = folder / "liquid.pt"

        torch.save({
            'model_state_dict': self.liquid_ensemble.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_citizens': self.n_citizens,
            'solver': self.solver,
            'load_distribution_lambda': self.liquid_ensemble.load_distribution_lambda,
            'specialization_lambda': self.liquid_ensemble.specialization_lambda
        }, file)


    @classmethod
    def load(folder: Path) -> Self:

        checkpoint = torch.load(folder / "liquid.pt", weights_only=False)
        model = LiquidEnsembleLayer(
            checkpoint["n_citizens"],
            synthetic=True,
            solver=checkpoint["solver"],
            load_distribution_lambda=checkpoint["load_distribution_lambda"],
            specialization_lambda=checkpoint["specialization_lambda"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        return model