from __future__ import annotations
from copy import deepcopy
import json
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import TensorDataset, Subset


import liquid.utils as utils
from .synthetic import sample
from .liquid_ensemble.le_adapter import LiquidLong, LiquidBlock
from .moe.moe_adapter import MoeBlock, MoeLong
from .plain.simple_adapter import SimpleNN
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

def load_data_cifar10(reduction: float = 0.05):
    data_dir = Path(__file__).parent.parent / 'cifar10_data'
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    train_len = int(len(train_dataset) * reduction)
    train_indices = np.random.choice(len(train_dataset), train_len, replace=False)
    train_dataset = Subset(train_dataset, train_indices)

    train_size = int(0.95 * len(train_dataset))
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
        save_files: bool = True,
        algos: list | None = None,
        folder_kwargs: dict | None = None
        ):


    task = params["name"]

    if save_files:
        folder_kwargs = folder_kwargs or dict()
        experiment_folder = utils.create_experiment_folder(task, experiment_name, **folder_kwargs)
    else:
        experiment_folder = None

    train_dataset, val_dataset = task_to_data[task]()

    x_train, y_train = dataset_to_numpy(train_dataset)
    train_dataset = None

    x_val, y_val = dataset_to_numpy(val_dataset)
    val_dataset = None

    if algos is not None:
        ...
    elif task == "protein":
        algos = [init_long_le, init_moe, init_rf, init_lgbm]
    elif task == "cifar10":
        algos = [init_long_le, init_block_le, init_long_moe, init_block_moe, init_simple]

    for init_func in algos:

        instance, train_kwargs = init_func(params, experiment_folder)
        instance.train(x_train, y_train, x_val, y_val, **train_kwargs)
        instance.save()

        if task == "protein":
            instance.evaluate_confidence_metrics(x_val, y_val)
        elif task == "cifar10":
            instance.evaluate_p_active_params(x_val, y_val)

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


def init_moe(params, experiment_folder,  variation: Literal["block", "long"] = "long"):


    ModelClass = {
        "block": MoeBlock,
        "long": MoeLong
    }[variation]

    epoch = params["epoch"]
    batch_size = params["batch_size"]
    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]
    model_params = params[ModelClass.name()]

    moe = ModelClass(
        n_input=n_input,
        task=task,
        n_output=n_output,
        folder=experiment_folder,
        lr=model_params["lr"],
    )
    moe.init_model(model_kwargs=model_params["architecture"])

    return moe, {
        "epoch": epoch,
        "batch_size": batch_size,
        "verbose": verbose
    }

def init_le(params, experiment_folder, variation: Literal["block", "long"] = "long"):

    ModelClass = LiquidLong if variation == "long" else LiquidBlock

    epoch = params["epoch"]
    batch_size = params["batch_size"]
    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]
    model_params = params[ModelClass.name()]

    le = ModelClass(
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


def init_long_moe(*args, **kwargs):
    return init_moe(*args, **kwargs, variation="long")

def init_block_moe(*args, **kwargs):
    return init_moe(*args, **kwargs, variation="block")

def init_long_le(*args, **kwargs):
    return init_le(*args, **kwargs, variation="long")

def init_block_le(*args, **kwargs):
    return init_le(*args, **kwargs, variation="block")

def init_simple(params, experiment_folder):

    epoch = params["epoch"]
    batch_size = params["batch_size"]
    n_input = params["n_input"]
    n_output = params["n_output"]
    verbose = params["verbose"]
    task = params["name"]
    model_params = params[SimpleNN.name()]

    le = SimpleNN(
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

    default_task = "cifar10"
    parser.add_argument("--task", type=str, default=default_task)
    parser.add_argument("--name", type=str, default=default_task)
    parser.add_argument("--algos", type=str, default="")
    args = parser.parse_args()

    task = args.task
    assert task in set(task_to_data.keys())
    params = load_params(task)


    algos_spec = {
        "le": init_long_le, "moe": init_moe, "rf": init_rf, "lgbm": init_lgbm
    }

    if args.algos == "":
        algos = None
    else:
        algos = []
        for alg in args.algos.split(","):
            algos.append(algos_spec[alg])
     
    train(
        experiment_name=args.name,
        params=params,
        algos=algos
    )