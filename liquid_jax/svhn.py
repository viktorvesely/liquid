from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from pathlib import Path

from task_base import Task


@struct.dataclass
class SvhnData:
    img: jax.Array
    label: jax.Array

class Svhn(Task[SvhnData]):
    
    folder = Path(__file__).parent.parent / "data" / "svhn"

    @staticmethod
    def load_from_tf():

        import tensorflow_datasets as tfds
        import numpy as np
        
        folder = Svhn.folder
        folder.mkdir(parents=True, exist_ok=True)

        train = tfds.as_numpy(
            tfds.load("svhn_cropped", split="train", batch_size=-1, as_supervised=True)
        )
        img, label = train
        np.savez(
            folder / "train.npz",
            img=img,
            label=label
        )

        test = tfds.as_numpy(
            tfds.load("svhn_cropped", split="test", batch_size=-1, as_supervised=True)
        )
        img, label = test
        np.savez(
            folder / "test.npz",
            img=img,
            label=label
        )

    
    @staticmethod
    def load_cpu(split: Literal["train", "test"]) -> SvhnData:
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):
            data_dict = jnp.load(Svhn.folder / f"{split}.npz")
            data = SvhnData(**data_dict)
            
            data = data.replace(
                img=data.img / 255.0
            )
            
            return data
        
    @staticmethod
    def get_xy(
        data: SvhnData
    ) -> tuple[jax.Array, jax.Array]:
        return data.img, data.label
    
    @staticmethod
    def task_type() -> Literal["classification", "regression"]:
        return "classification"
    
    @staticmethod
    def out_dim() -> int:
        return 10
    
if __name__ == "__main__":

    Svhn.load_from_tf()