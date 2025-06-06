import argparse
import csv
from pathlib import Path
from typing import Any, Callable
import numpy as np
import traceback

from .train import init_le, init_rf, init_lgbm, init_moe, task_to_data, dataset_to_numpy
from .hyper_protein import h_le, h_lgbm, h_rf, h_moe
from .adapter import Adapter
from .utils import create_experiment_folder

MAX_TRIALS = 100_000
DEBUG = False

def get_init(algorithm: str) -> Callable[[dict, Path], tuple[Adapter, dict]]:

    if algorithm == "moe":
        return init_moe
    elif algorithm == "le":
        return init_le
    elif algorithm == "rf":
        return init_rf
    elif algorithm == "lgbm":
        return init_lgbm

    raise ValueError(algorithm)


def get_hyper(algorithm: str):

    if algorithm == "moe":
        return h_moe
    elif algorithm == "le":
        return h_le
    elif algorithm == "rf":
        return h_rf
    elif algorithm == "lgbm":
        return h_lgbm

    raise ValueError(algorithm)

def flatten_dict(results: dict, existing: dict = None) -> dict:

    if existing is None:
        existing = dict()

    for k, v in results.items():

        assert k not in existing

        if isinstance(v, dict):
            flatten_dict(v, existing)
        else:
            existing[k] = v

    return existing

def dict_to_row(results: dict, header: bool = False) -> list[str]:
    flat = flatten_dict(results)
    items = list(flat.items())
    items = sorted(items, key=lambda item: item[0])
    index = 0 if header else 1
    values = [str(item[index]) for item in items]
    return values

def write_header(path: str, results: dict):

    header = dict_to_row(results, header=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def write_row(path: str, results: dict, lock):
    row = dict_to_row(results)
    if lock is not None:
        with lock, open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    else:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def trial(
        params: dict,
        algorithm: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> dict:

    init_func = get_init(algorithm)

    instance, train_kwargs = init_func(params, None)
    instance.train(x_train, y_train, x_val, y_val, **train_kwargs)
    instance.evaluate_confidence_metrics(x_val, y_val)

    return {f"{k}_results": v for k, v in instance._test_metrics.items()}

def worker(args: tuple[str, str, str, Any], warmup: bool = False):

    algorithm, task, path, lock = args
    h_func = get_hyper(algorithm)

    train_dataset, val_dataset = task_to_data[task]()

    x_train, y_train = dataset_to_numpy(train_dataset)
    del train_dataset

    x_val, y_val = dataset_to_numpy(val_dataset)
    del val_dataset

    for _ in range(MAX_TRIALS):
        params = h_func()
        params["verbose"] = 0 if not DEBUG else 1

        if DEBUG:
            params["epoch"] = 1

        try:
            results = trial(params, algorithm, x_train, y_train, x_val, y_val)
            row = params | results
            if warmup:
                return row
            write_row(path, row, lock)
        except Exception as e:
            if warmup:
                raise

            traceback.print_exc()
            print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, required=False, default=1)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cpu = args.cpu
    algorithm = args.algorithm

    if args.debug:
        DEBUG = True
        MAX_TRIALS = 4

    if cpu > 1:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        lock = manager.Lock()
    else:
        lock = None

    h_func = get_hyper(algorithm)
    dummy = h_func()
    task = dummy["name"]

    folder = create_experiment_folder(task, name=algorithm, hyper=True)
    result_path = str(folder / f"{algorithm}.csv")

    params = []
    for _ in range(cpu):
        params.append((algorithm, task, result_path, lock))

    row = worker(params[0], warmup=True)
    write_header(result_path, row)
    write_row(result_path, row, lock)

    if cpu == 1:
        worker(params[0])
    else:
        with mp.Pool(processes=cpu) as pool:
            for _ in pool.imap_unordered(worker, params):
                ...



