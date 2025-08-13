from __future__ import annotations
from copy import deepcopy
import json

import numpy as np
import pandas as pd
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
from torch.utils.data import random_split

import argparse


def load_protein():

    df = pd.read_csv(Path(__file__).parent.parent / "protein_train.csv")
    df = df.reset_index(drop=True)

    i_val = np.random.choice(df.shape[0], size=int(df.shape[0] * 0.1), replace=False)
    df_val = df.iloc[i_val, ].copy()
    df_train = df.drop(index=i_val)

    x_col = df.columns[df.columns.str.contains("F")]
    y_col = "RMSD"

    train_dataset = TensorDataset(
        torch.tensor(df_train[x_col].to_numpy(), dtype=torch.float32),
        torch.tensor(df_train[y_col].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    )

    val_dataset = TensorDataset(
        torch.tensor(df_val[x_col].to_numpy(), dtype=torch.float32),
        torch.tensor(df_val[y_col].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    )

    return train_dataset, val_dataset

def load_data_cifar10(reduction: float = 0.25):
    data_dir = Path(__file__).parent.parent / 'cifar10_data'
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    train_len = int(len(train_dataset) * reduction)
    train_indices = np.random.choice(len(train_dataset), train_len, replace=False)
    train_dataset = Subset(train_dataset, train_indices)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset

task_to_data = {
    "cifar10": load_data_cifar10,
    "protein": load_protein
}


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
        experiment_name: str = "experiment",
        save_files: bool = True
        ):


    task = params["name"]

    if save_files:
        experiment_folder = utils.create_experiment_folder(task, experiment_name)
    else:
        experiment_folder = None

    train_dataset, val_dataset = task_to_data[task]()

    x_train, y_train = dataset_to_numpy(train_dataset)
    train_dataset = None

    x_val, y_val = dataset_to_numpy(val_dataset)
    val_dataset = None

    # for init_func in [init_le, init_moe, init_rf, init_lgbm]:
    for init_func in [init_moe]:

        instance, train_kwargs = init_func(params, experiment_folder)
        instance.train(x_train, y_train, x_val, y_val, **train_kwargs)
        instance.save()

        if instance._track_confidence:
            instance.evaluate_confidence_metrics(x_val, y_val)
        instance.save_metrics()


def init_lgbm(params, experiment_folder):

    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]

    lgbm = LightGBM(
        n_input=n_input,
        n_output=n_output,
        task=task,
        folder=experiment_folder,
        estimate_confidence=True,
        **params["lgbm"]
    )
    lgbm.init_model()
    return lgbm, {
        "verbose": verbose
    }

def init_rf(params, experiment_folder):

    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]

    bagging = RandomForest(
        n_input=n_input,
        n_output=n_output,
        task=task,
        folder=experiment_folder,
        **params["rf"],
    )
    bagging.init_model()
    return bagging, {
        "verbose": verbose
    }

def init_moe(params, experiment_folder):

    epoch = params["epoch"]
    batch_size = params["batch_size"]
    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]

    moe = Moe(
        n_input=n_input,
        task=task,
        n_output=n_output,
        folder=experiment_folder,
        lr=params["Moe"]["lr"],
    )
    moe.init_model(model_kwargs=params["Moe"]["architecture"])

    return moe, {
        "epoch": epoch,
        "batch_size": batch_size,
        "verbose": verbose
    }

def init_le(params, experiment_folder):

    epoch = params["epoch"]
    batch_size = params["batch_size"]
    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]
    model_params = params[LiquidLong.name()]

    le = LiquidLong(
        n_input=n_input,
        n_output=n_output,
        folder=experiment_folder,
        task=task,
        lr=model_params["lr"]
    )
    le.init_model(model_kwargs=model_params["architecture"])
    return le, {
        "epoch": epoch,
        "batch_size": batch_size,
        "verbose": verbose
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="protein")
    parser.add_argument("--name", type=str, default="protein")
    args = parser.parse_args()

    task = args.task
    assert task in set(task_to_data.keys())
    params = load_params(task)

    train(
        experiment_name=args.name,
        params=params
    )