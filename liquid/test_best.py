import json
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from .train import train, init_long_le, init_moe

from .liquid_ensemble.le_adapter import LiquidLong
from .moe.moe_adapter import MoeLong

def load_params(task: str = "protein"):

    with open(Path(__file__).parent / f"{task}_best.json", "r") as f:
        params = json.load(f)

    return params


def gather_best(
    metric: str,
    task: str = "protein"
) -> list[Path]:

    exp_folders = Path(__file__).parent.parent / "experiments" / task / "best"

    good = []
    for ent in exp_folders.iterdir():

        if not ent.is_dir() or (f"best_{metric}" not in ent.name):
            continue

        good.append(ent)

    return good

def dataset() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(Path(__file__).parent.parent / "protein_train.csv")
    df = df.reset_index(drop=True)

    x_col = df.columns[df.columns.str.contains("F")]
    y_col = "RMSD"

    return df[x_col].to_numpy(), df[y_col].to_numpy()

def generate_partition_dataset(
    x: np.ndarray,
    y: np.ndarray,
    bests: list[Path]
):

    results_powers = {
        LiquidLong: [],
        MoeLong: []
    }


    def on_batch(model: LiquidLong | MoeLong, partitions, *args, **kwargs):

        if isinstance(model, LiquidLong):
            architecture, _ = model.get_nn()
            power = architecture.get_le_layers()[0].last_power
            partitions.append(power.cpu().numpy())

        if isinstance(model, MoeLong):
            architecture, _ = model.get_nn()
            power = architecture.get_moe_layers()[0].last_gate
            partitions.append(power.cpu().numpy())


    for best in bests:

        for ModelClass in [LiquidLong, MoeLong]:

            ModelClass: LiquidLong | MoeLong = ModelClass
            model = ModelClass.load(best)

            partitions = []
            model.inference(x, batch_size=1_000, on_batch=partial(on_batch, model, partitions))
            partitions = np.concatenate(partitions, axis=0)

            results_powers[ModelClass].append(partitions)


    par_folder = Path(__file__).parent / "partitions"
    par_folder.mkdir(exist_ok=True)

    np.save(par_folder / "x.npy", x)
    np.save(par_folder / "y.npy", y)

    for ModelClass, runs in results_powers.items():
        name = ModelClass.name()
        for i, partitions in enumerate(runs):
            np.save(par_folder / f"{name}_{i}.npy", partitions)

def train_n_best(
    metric: str,
    n: int = 1
):

    params = load_params()

    for _ in range(n):
        train(
            experiment_name=f"best_{metric}",
            params=params,
            algos=[init_moe, init_long_le],
            folder_kwargs={"inner": "best"}
        )


if __name__ == "__main__":

    #train_n_best("rmse")

    x, y = dataset()
    generate_partition_dataset(x, y, gather_best("rmse"))
