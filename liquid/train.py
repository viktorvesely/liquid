from __future__ import annotations
from typing import Literal

import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import TensorDataset


import liquid.utils as utils
from .synthetic import sample
from .liquid_ensemble.adapter import Liquid
from .moe.adapter import Moe
from .forests.bagging import RandomForest
from .forests.lgbm import LightGBM

import argparse

def load_data_mnist():
    data_dir = Path(__file__).parent.parent / 'mnist_data'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def load_data_cifar10():
    data_dir = Path(__file__).parent.parent / 'cifar10_data'
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

    le = Liquid(
        n_input=2,
        n_output=3,
        folder=experiment_folder,
        n_citizens=n_citizens,
        lr=5 * 1e-4,
        load_distribution_lambda=load_distribution_lambda,
        specialization_lambda=specialization_lambda,
        solver=solver
    )
    le.init_model()
    val_metrics = le.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)

    me = Moe(
        n_input=2,
        n_output=3,
        folder=experiment_folder,
        n_experts=n_citizens,
        lr=5 * 1e-4
    )
    me.init_model()
    val_metrics = me.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)


    bagging = RandomForest(
        n_input=2,
        n_output=3,
        folder=experiment_folder,
        n_estimators=100
    )
    bagging.init_model()
    val_metrics = bagging.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)
    bagging.save()

    lgbm = LightGBM(
        n_input=2,
        n_output=3,
        folder=experiment_folder,
        n_estimators=100
    )
    lgbm.init_model()
    val_metrics = lgbm.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)
    lgbm.save()

    return val_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--habrok", action="store_true")
    args = parser.parse_args()

    train(
        experiment_name="all_together",
        epoch=80
    )