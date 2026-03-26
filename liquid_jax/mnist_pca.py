from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from sklearn.decomposition import PCA

from task_base import Task
from mnist import Mnist


@struct.dataclass
class MnistPcaData:
    img: jax.Array
    label: jax.Array


def make_pca_task(n_components: int) -> type[Task]:
    """Create a Task class for MNIST reduced to n_components PCA dims.

    PCA is fit on the train split only.
    """
    # Fit PCA on train
    train_raw = Mnist.load_cpu(split="train")
    x_train = np.asarray(train_raw.img).reshape((-1, 784))

    pca = PCA(n_components=n_components)
    pca.fit(x_train)

    # Transform both splits and cache
    cpu = jax.devices("cpu")[0]
    cache = {}
    for split in ("train", "test"):
        raw = Mnist.load_cpu(split=split) if split == "test" else train_raw
        x = np.asarray(raw.img).reshape((-1, 784))
        x_pca = pca.transform(x).astype(np.float32)
        with jax.default_device(cpu):
            cache[split] = MnistPcaData(
                img=jnp.array(x_pca),
                label=raw.label,
            )

    class MnistPca(Task[MnistPcaData]):

        @staticmethod
        def load_cpu(split: Literal["train", "test"]) -> MnistPcaData:
            return cache[split]

        @staticmethod
        def get_xy(data: MnistPcaData) -> tuple[jax.Array, jax.Array]:
            return data.img, data.label

    MnistPca.__name__ = f"MnistPca_{n_components}"
    return MnistPca, pca
