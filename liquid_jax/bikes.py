from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from pathlib import Path

from task_base import Task
from regression import dict_to_x


@struct.dataclass
class BikesData:
    x: jax.Array
    y: jax.Array

class Bikes(Task[BikesData]):
    
    folder = Path(__file__).parent.parent / "data" / "bikes"

    @staticmethod
    def process_raw():

        import pandas as pd
        import numpy as np
        
        folder = Bikes.folder
        df = pd.read_csv(folder / "hour.csv")

        df = df.sample(len(df), random_state=123)

        normalize_features = ("hr", )
        noop_features = ("yr", "holiday", "workingday", "temp", "atemp", "hum", "windspeed",)
        onehot_features = ("season", "mnth", "weekday", "weathersit")
        target = ("cnt",)

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
    def load_cpu(split: Literal["train", "test"]) -> BikesData:
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):
            data_dict = jnp.load(Bikes.folder / f"{split}.npz")
            data = BikesData(**data_dict)
            return data
        
    @staticmethod
    def get_xy(
        data: BikesData
    ) -> tuple[jax.Array, jax.Array]:
        return data.x, data.y
    
    @staticmethod
    def task_type() -> Literal["classification", "regression"]:
        return "regression"
    
    @staticmethod
    def out_dim() -> int:
        return 1
    
if __name__ == "__main__":

    Bikes.process_raw()

    # data = Cifar10.load_cpu(split="test")
    # print(data.img.shape, data.label.shape, data.img.max())
