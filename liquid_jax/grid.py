

from dataclasses import replace
from itertools import product
from typing import Type

import jax
from flax import struct
from matplotlib import pyplot as plt

from cifar10 import Cifar10
from bikes import Bikes
from svhn import Svhn
from learner_le import LeLearner
from structs import TrainParams
from task_base import Task
from train import finish_run, make_train_folder, train

@struct.dataclass
class Architecture:
    spec_predictors: tuple[int, ...]
    spec_delegators: tuple[int, ...]
    widths_predictors: tuple[int, ...]
    widths_delegators: tuple[int, ...]

big_mlp = Architecture(
    spec_predictors = (2, 4, 8, 16, 32, 64),
    spec_delegators = (0, 1, 2, 4, 8, 16, 32, 64),
    widths_predictors = (8, 64),
    widths_delegators = (4, 32),
)

small_mlp = Architecture(
    spec_predictors = (2, 4, 8, 16, 32, 64),
    spec_delegators = (0, 1, 2, 4, 8, 16, 32, 64),
    widths_predictors = (8, 16),
    widths_delegators = (4, 8),
)

@struct.dataclass
class GridSpec:

    preload_batches_to_gpu: int
    valid_batches: int
    epochs: int
    architecture: Architecture
    task: Type[Task]


specs: dict[str, GridSpec] = {
    "cifar10": GridSpec(
        preload_batches_to_gpu=20,
        valid_batches=4,
        epochs=50,
        task=Cifar10,
        architecture=big_mlp
    ),
    "svhn": GridSpec(
        preload_batches_to_gpu=20,
        valid_batches=4,
        epochs=50,
        task=Svhn,
        architecture=big_mlp
    ),
    "bikes": GridSpec(
        preload_batches_to_gpu=50,
        valid_batches=4,
        epochs=1_000,
        task=Bikes,
        architecture=small_mlp
    ),
}



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()

    tasks: list[str] = args.tasks

    for task in tasks:

        gridspec = specs[task]

        g_params = TrainParams(
            batch_size=512,
            preload_batches_to_gpu=gridspec.preload_batches_to_gpu,
            valid_batches=gridspec.valid_batches,
            epochs=gridspec.epochs,
            lr=1e-3,
            optimizer="adam",
            task=gridspec.task,
            n_predictors=-1,
            n_delegators=-1,
            learner=LeLearner
        )

        folder = make_train_folder(f"grid_{task}")
        key = jax.random.key(123)

        spec_predictors = gridspec.architecture.spec_predictors
        spec_delegators = gridspec.architecture.spec_delegators
        widths_predictors = gridspec.architecture.widths_predictors
        widths_delegators = gridspec.architecture.widths_delegators

        specs = list(product(spec_predictors, spec_delegators, widths_predictors, widths_delegators))

        finished = 0

        for n_predictors, n_delegators, predictor_width, delegator_width in reversed(specs):

            # Reset
            jax.clear_caches()
            plt.close("all")

            prefix = f"predictors_{n_predictors}_delegators_{n_delegators}_pwidth_{predictor_width}_dwidth_{delegator_width}"
            print(prefix)

            metrics = train(
                key=key,
                train_params=replace(
                    g_params,
                    learner=LeLearner,
                    n_predictors=n_predictors,
                    n_delegators=n_delegators,
                    predictor=(predictor_width, predictor_width // 2, g_params.task.out_dim()),
                    delegator=(delegator_width, n_predictors)
                )
            )
            finish_run(metrics, folder, prefix=prefix)
            
            finished += 1
            print(f"{finished} / {len(specs)}")