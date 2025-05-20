from __future__ import annotations
from copy import deepcopy
import json

import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import TensorDataset, Subset


import liquid.utils as utils
from .synthetic import sample
from .liquid_ensemble.le_adapter import LiquidLong, LiquidBlock
from .moe.moe_adapter import Moe
from .forests.bagging import RandomForest
from .forests.lgbm import LightGBM

import argparse

def load_data_mnist():
    data_dir = Path(__file__).parent.parent / 'mnist_data'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def load_data_cifar10(reduction: float = 0.25):
    data_dir = Path(__file__).parent.parent / 'cifar10_data'
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    train_len = int(len(train_dataset) * reduction)
    test_len = int(len(test_dataset) * reduction)

    train_indices = np.random.choice(len(train_dataset), train_len, replace=False)
    test_indices = np.random.choice(len(test_dataset), test_len, replace=False)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset

def dataset_to_numpy(dataset):

    x = []
    y = []

    for img, label in dataset:
        x.append(img.numpy())
        y.append(label)

    return np.stack(x), np.array(y)


def load_data_synthetic(n_train: int = 50_000, n_valid: int = 5_000):

    x_train, y_train = sample(n_train)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

    x_test, y_test = sample(n_valid)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    return train_dataset, test_dataset


def load_params(task: str):

    with open(Path(__file__).parent / f"{task}.json", "r") as f:
        params = json.load(f)

    return params

def train(
        params: dict,
        task: str,
        experiment_name: str = "experiment",
        save_files: bool = True,
        verbose: int = 1
        ):


    epoch = params["epoch"]
    batch_size = params["batch_size"]
    n_input = params["n_input"]
    n_output = params["n_output"]

    if save_files:
        experiment_folder = utils.create_experiment_folder(task, experiment_name)
    else:
        experiment_folder = None

    train_dataset, val_dataset = load_data_cifar10()

    x_train, y_train = dataset_to_numpy(train_dataset)
    train_dataset = None

    x_val, y_val = dataset_to_numpy(val_dataset)
    val_dataset = None


    # model_params = params[LiquidLong.name()]
    # le = LiquidLong(
    #     n_input=n_input,
    #     n_output=n_output,
    #     folder=experiment_folder,
    #     lr=model_params["lr"]
    # )
    # le.init_model(model_kwargs=model_params["architecture"])
    # le.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)
    # le = None


    # moe = Moe(
    #     n_input=n_input,
    #     n_output=n_output,
    #     folder=experiment_folder,
    #     lr=params["Moe"]["lr"],
    # )
    # moe.init_model(model_kwargs=params["Moe"]["MoeCifar10"])
    # val_metrics = moe.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)


    # bagging = RandomForest(
    #     n_input=2,
    #     n_output=3,
    #     folder=experiment_folder,
    #     n_estimators=100
    # )
    # bagging.init_model()
    # val_metrics = bagging.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)
    # bagging.save()

    lgbm = LightGBM(
        n_input=n_input,
        n_output=n_output,
        folder=experiment_folder,
        n_estimators=1_000
    )
    lgbm.init_model()
    val_metrics = lgbm.train(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, verbose=verbose)
    lgbm.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cifar10")
    args = parser.parse_args()

    task = args.task
    assert task in {"cifar10"}
    params = load_params(task)

    train(
        experiment_name="check_sizes",
        params=params,
        task=task
    )