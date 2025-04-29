from __future__ import annotations
from typing import Literal

import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import TensorDataset

import utils

from synthetic import sample
from LE import LE

import argparse

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


def load_data_synthetic(n_train: int = 50_000, n_valid: int = 5_000):

    x_train, y_train = sample(n_train)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

    x_test, y_test = sample(n_valid)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    return train_dataset, test_dataset


def train(
        n_citizens: int = 4,
        experiment_name: str = "experiment",
        load_distribution_lambda: float = 0.0,
        specialization_lambda: float = 0.0,
        epoch: int = 150,
        batch_size: int = 1_500,
        verbose: int = 2,
        solver: Literal["sink_one", "sink_many"] = "sink_one",
        save_files: bool = True
        ):

    if save_files:
        experiment_folder = utils.create_experiment_folder(experiment_name)
        # utils.copy_files_to_folder(experiment_folder, "liquid_council.py")
    else:
        experiment_folder = None

    train_dataset, val_dataset = load_data_synthetic()

    x_train = train_dataset.tensors[0].numpy()
    y_train = train_dataset.tensors[1].numpy()
    train_dataset = None

    x_val = val_dataset.tensors[0].numpy()
    y_val = val_dataset.tensors[1].numpy()
    val_dataset = None

    le = LE(experiment_folder)
    le.init_model(
        n_citizens=n_citizens,
        load_distribution_lambda=load_distribution_lambda,
        specialization_lambda=specialization_lambda,
        solver=solver
    )

    val_metrics = le.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)

    return val_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--habrok", action="store_true")
    args = parser.parse_args()

    train(
        experiment_name="best_nmi",
        load_distribution_lambda=0.75,
        specialization_lambda=0.5,
        solver="sink_many",
        epoch=5
    )