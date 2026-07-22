

from dataclasses import replace
from itertools import product
from typing import Type

import jax
from flax import struct
from matplotlib import pyplot as plt

from cifar10 import Cifar10
from bikes import Bikes
from energy import Energy
from svhn import Svhn
from learner_le import LeLearner
from structs import TrainParams
from task_base import Task
from train import finish_run, make_train_folder, train
from atomic_networks import Architecture, small_cnn, two_layer_mlp, three_layer_mlp



@struct.dataclass
class ArchitectureSpecs:
    spec_predictors: tuple[int, ...]
    spec_delegators: tuple[int, ...]
    widths_predictors: tuple[int, ...]
    widths_delegators: tuple[int, ...]
    base: Architecture

cnn_specs = ArchitectureSpecs(
    spec_predictors = (2, 4, 8, 16, 32, 64),
    spec_delegators = (0, 1, 2, 4, 8, 16, 32, 64),
    widths_predictors = (1, 4, 8),
    widths_delegators = (1, 4, 8),
    base = small_cnn
)

small_mlp = ArchitectureSpecs(
    spec_predictors = (2, 4, 8, 16, 32, 64),
    spec_delegators = (0, 1, 2, 4, 8, 16, 32, 64),
    widths_predictors = (4, 8, 16),
    widths_delegators = (4, 8, 16),
    base = two_layer_mlp
)

big_mlp = ArchitectureSpecs(
    spec_predictors = (2, 4, 8, 16, 32, 64),
    spec_delegators = (0, 1, 2, 4, 8, 16, 32, 64),
    widths_predictors = (4, 8, 16),
    widths_delegators = (4, 8, 16),
    base = three_layer_mlp
)

@struct.dataclass
class GridSpec:

    batch_size: int
    preload_batches_to_gpu: int
    valid_batches: int
    epochs: int
    architecture_specs: ArchitectureSpecs
    task: Type[Task]


specs: dict[str, GridSpec] = {
    "cifar10": GridSpec(
        batch_size = 64,
        preload_batches_to_gpu=20,
        valid_batches=20,
        epochs=50,
        task=Cifar10,
        architecture_specs=cnn_specs
    ),
    "svhn": GridSpec(
        batch_size = 64,
        preload_batches_to_gpu=20,
        valid_batches=20,
        epochs=50,
        task=Svhn,
        architecture_specs=cnn_specs
    ),
    "bikes": GridSpec(
        batch_size = 256,
        preload_batches_to_gpu=50,
        valid_batches=7,
        epochs=1_000,
        task=Bikes,
        architecture_specs=small_mlp
    ),
    "energy": GridSpec(
        batch_size = 256,
        preload_batches_to_gpu=50,
        valid_batches=7,
        epochs=2_000,
        task=Energy,
        architecture_specs=big_mlp
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
    parser.add_argument(
        "--agg",
        type=str,
        choices=("sum", "product"),
        required=True
    )
    args = parser.parse_args()

    
    delegators_mixing = args.agg
    tasks: list[str] = args.tasks

    for task in tasks:

        gridspec = specs[task]

        g_params = TrainParams(
            batch_size=gridspec.batch_size,
            preload_batches_to_gpu=gridspec.preload_batches_to_gpu,
            valid_batches=gridspec.valid_batches,
            epochs=gridspec.epochs,
            lr=1e-3,
            optimizer="adam",
            task=gridspec.task,
            n_predictors=-1,
            n_delegators=-1,
            delegators_mixing=delegators_mixing,
            architecture=None,
            learner=LeLearner,
        )

        folder = make_train_folder(f"grid_{task}_agg_{delegators_mixing}")
        key = jax.random.key(123)

        spec_predictors = gridspec.architecture_specs.spec_predictors
        spec_delegators = gridspec.architecture_specs.spec_delegators
        widths_predictors = gridspec.architecture_specs.widths_predictors
        widths_delegators = gridspec.architecture_specs.widths_delegators


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
                    delegators_mixing=delegators_mixing,
                    architecture=gridspec.architecture_specs.base.determine_size(
                        predictor_base=predictor_width,
                        delegator_base=delegator_width,
                        out_dim=gridspec.task.out_dim(),
                        n_predictors=n_predictors
                    )
                )
            )
            finish_run(metrics, folder, prefix=prefix)
            
            finished += 1
            print(f"{finished} / {len(specs)}")