from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from pathlib import Path

from task_base import Task
from regression import dict_to_x


@struct.dataclass
class EnergyData:
    x: jax.Array
    y: jax.Array

class Energy(Task[EnergyData]):
    
    folder = Path(__file__).parent.parent / "data" / "energy"

    @staticmethod
    def process_raw():

        import pandas as pd
        import numpy as np
        
        folder = Energy.folder
        df = pd.read_csv(folder / "data.csv")
        df = df[df["Source"] != "Mixed"]

        df = df.sample(len(df), random_state=123)

        categorize_features = ("Source", "Day_Name", "Day_Name", "Month_Name", "Season")
        for cf in categorize_features:
            df[cf] = df[cf].astype("category").cat.codes

        normalize_features = ("Start_Hour", "End_Hour", "Day_of_Year")
        noop_features = ()
        onehot_features = categorize_features
        target = ("Production",)

        def isint(x):
            assert x == int(x) 

        for of in onehot_features:
            low, high = df[of].min(), df[of].max()
            isint(low), isint(high)
            df[of] = df[of] - low

        cols = normalize_features + noop_features + onehot_features + target
        data = {c: jnp.array(df[c].to_numpy()) for c in cols}
        
        x = dict_to_x(data, normalize_features=normalize_features, onehot_features=onehot_features, noop_features=noop_features)
        y = dict_to_x(data, normalize_features=target)

        assert (not jnp.isnan(x).any())
        assert (not jnp.isnan(y).any())

        n_test = int(len(df) * 0.1)
        x_train = x[:-n_test]
        y_train = y[:-n_test]

        x_test = x[-n_test:]
        y_test = y[-n_test:]

        np.savez(
            folder / "train.npz",
            x=x_train,
            y=y_train
        )

        np.savez(
            folder / "test.npz",
            x=x_test,
            y=y_test
        )

    
    @staticmethod
    def load_cpu(split: Literal["train", "test"]) -> EnergyData:
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):
            data_dict = jnp.load(Energy.folder / f"{split}.npz")
            data = EnergyData(**data_dict)
            return data
        
    @staticmethod
    def get_xy(
        data: EnergyData
    ) -> tuple[jax.Array, jax.Array]:
        return data.x, data.y
    
    @staticmethod
    def task_type() -> Literal["classification", "regression"]:
        return "regression"
    
    @staticmethod
    def out_dim() -> int:
        return 1
    
if __name__ == "__main__":

    Energy.process_raw()

    # data = Cifar10.load_cpu(split="test")
    # print(data.img.shape, data.label.shape, data.img.max())
