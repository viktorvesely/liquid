from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from pathlib import Path

from task_base import Task


@struct.dataclass
class Cifar10Data:
    img: jax.Array
    label: jax.Array

class Cifar10(Task[Cifar10Data]):
    
    folder = Path(__file__).parent.parent / "data" / "cifar"

    @staticmethod
    def load_from_tf():

        import tensorflow_datasets as tfds
        import numpy as np
        
        folder = Cifar10.folder
        folder.mkdir(parents=True, exist_ok=True)

        train = tfds.as_numpy(
            tfds.load("cifar10", split="train", batch_size=-1, as_supervised=True)
        )
        img, label = train
        np.savez(
            folder / "train.npz",
            img=img,
            label=label
        )

        test = tfds.as_numpy(
            tfds.load("cifar10", split="test", batch_size=-1, as_supervised=True)
        )
        img, label = test
        np.savez(
            folder / "test.npz",
            img=img,
            label=label
        )

    
    @staticmethod
    def load_cpu(split: Literal["train", "test"]) -> Cifar10Data:
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):
            data_dict = jnp.load(Cifar10.folder / f"{split}.npz")
            data = Cifar10Data(**data_dict)
            return data.replace(
                img=data.img / 255.0
            )
        
    @staticmethod
    def get_xy(
        data: Cifar10Data
    ) -> tuple[jax.Array, jax.Array]:
        return data.img, data.label
    
    @staticmethod
    def task_type() -> Literal["classification", "regression"]:
        return "classification"
    
    @staticmethod
    def out_dim() -> int:
        return 10
    
if __name__ == "__main__":

    Cifar10.load_from_tf()

    # data = Cifar10.load_cpu(split="test")
    # print(data.img.shape, data.label.shape, data.img.max())
