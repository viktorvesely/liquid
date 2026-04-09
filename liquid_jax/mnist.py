from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from pathlib import Path

from task_base import Task


@struct.dataclass
class MnistData:

    img: jax.Array
    label: jax.Array

class Mnist(Task[MnistData]):

    
    folder = Path(__file__).parent.parent / "data" / "mnist"

    @staticmethod
    def load_from_tf():

        import tensorflow_datasets as tfds
        import numpy as np
        
        folder = Mnist.folder
        folder.mkdir(parents=True, exist_ok=True)

        train = tfds.as_numpy(
            tfds.load("mnist", split="train", batch_size=-1, as_supervised=True)
        )
        img, label = train
        np.savez(
            folder / "train.npz",
            img=img,
            label=label
        )


        test = tfds.as_numpy(
            tfds.load("mnist", split="test", batch_size=-1, as_supervised=True)
        )
        img, label = test
        np.savez(
            folder / "test.npz",
            img=img,
            label=label
        )

    
    @staticmethod
    def load_cpu(split: Literal["train", "test"]) -> MnistData:
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):
            data_dict = jnp.load(Mnist.folder / f"{split}.npz")
            data = MnistData(**data_dict)
            return data.replace(
                img=data.img / 255.0
            )
        
    @staticmethod
    def get_xy(
        data: MnistData
    ) -> tuple[jax.Array, jax.Array]:
        return data.img, data.label
    
    @staticmethod
    def task_type() -> Literal["classification", "regression"]:
        return "classification"
    
    @staticmethod
    def out_dim() -> int:
        return 10
    
if __name__ == "__main__":

    # Mnist.load_from_tf()

    data = Mnist.load_cpu(split="test")
    print(data.img.shape, data.label.shape, data.img.max())
