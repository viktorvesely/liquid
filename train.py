from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.optim import AdamW, Optimizer
import torch.nn as nn
import tqdm

import utils

from matplotlib import pyplot as plt

from liquid_council import LiquidCouncil
from council import Council

def load_data_mnist():
    data_dir = Path(__file__).parent / 'mnist_data'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def load_data_cifar10():
    data_dir = Path(__file__).parent / 'cifar10_data'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def save_models(councils: list[Council], optimizers: list[Optimizer], folder: Path):
    for council, optimizer in zip(councils, optimizers, strict=True):
        utils.save_checkpoint(council, optimizer, folder / f"{council.name()}.pt")

class Metrics:

    def __init__(self, **metrics):
        self.metrics = {name: [] for name in metrics.keys()}
        self.history = {name: [] for name in metrics.keys()}

    def push(self, **metrics):
        for name, value in metrics.items():
            self.metrics[name].append(value)

    def reset(self):
        for name, values in self.metrics.items():
            if len(values) == 0:
                continue

            self.history[name].append(np.mean(values))
            self.metrics[name] = []

    @classmethod
    def empty_like(cls, other: Metrics, **extra):
        return Metrics(**(other.metrics | extra))

    def __repr__(self):
        out = []
        for name, values in self.metrics.items():
            out.append(f"{name}={np.mean(values):.3f}")
        return ", ".join(out)

def train(n_citizens: int = 8, experiment_name: str = "mnist_small"):

    batch_size = 1_000
    epoch = 25

    experiment_folder = utils.create_experiment_folder(experiment_name)
    utils.copy_files_to_folder(experiment_folder, "council.py", "dictator_council.py", "liquid_council.py")

    train_dataset, test_dataset = load_data_mnist()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    liquid_council = LiquidCouncil(n_citizens)
    majority_council = liquid_council.cut_delegation_heads()
    dictator = majority_council.make_dictator()

    councils = [
        liquid_council,
        majority_council,
        dictator
    ]

    train_liquid_metric = Metrics(loss=None, power_entropy=None, speaker_entropy=None)

    train_metrics = [
        train_liquid_metric,
        Metrics(loss=None),
        Metrics(loss=None)
    ]

    valid_metrics = [Metrics.empty_like(metric, accuracy=None) for metric in train_metrics]
    valid_liquid_metric = valid_metrics[0]

    optimizers = [AdamW(council.parameters(), lr=1e-4) for council in councils]

    for e in range(epoch):

        for council in councils: council.train()
        for metric in train_metrics: metric.reset()

        for images, labels in tqdm.tqdm(train_loader, total=len(train_loader), desc="Train"):

            images = images.to("cuda")
            labels = labels.to("cuda")

            for council, optimizer, metrics in zip(councils, optimizers, train_metrics, strict=True):
                classifications = council(images)
                loss = council.loss(labels, classifications)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics.push(loss=loss.item())


            with torch.no_grad():
                power_entropy = liquid_council.power_entropy()
                speaker_entropy = liquid_council.speaker_entropy()
                train_liquid_metric.push(power_entropy=power_entropy.item(), speaker_entropy=speaker_entropy.item())


        for council in councils: council.eval()
        for metric in valid_metrics: metric.reset()

        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader, total=len(test_loader), desc="Valid"):

                images = images.to("cuda")
                labels = labels.to("cuda")

                for council, optimizer, metrics in zip(councils, optimizers, valid_metrics, strict=True):

                    classifications = council(images)
                    loss = council.loss(labels, classifications)
                    labels_hat = council.vote(classifications)
                    correct = (labels_hat == labels).float()
                    accuracy = correct.mean()

                    metrics.push(loss=loss.item(), accuracy=accuracy.item())

                power_entropy = liquid_council.power_entropy()
                speaker_entropy = liquid_council.speaker_entropy()

                valid_liquid_metric.push(power_entropy=power_entropy.item(), speaker_entropy=speaker_entropy.item())

        print(f"\n--------Epoch {e}-----------")
        for i, council in enumerate(councils):
            print(f"\n {council.name()}")
            print(f"Train: {train_metrics[i]}")
            print(f"Valid: {valid_metrics[i]}")


    for metric in (train_metrics + valid_metrics): metric.reset()
    fig, ax = plt.subplots()

    for metric, name in zip(valid_metrics, ["liquid", "majority", "dictator"]):
        ax.plot(metric.history["accuracy"], label=name)

    ax.legend()
    plt.show()

    save_models(councils, optimizers, experiment_folder)



if __name__ == "__main__":
    train()