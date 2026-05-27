

from dataclasses import replace
from itertools import product

import jax
from matplotlib import pyplot as plt

from cifar10 import Cifar10
from learner_le import LeLearner
from structs import TrainParams
from train import finish_run, make_train_folder, train

g_params = TrainParams(
    batch_size=512,
    preload_batches_to_gpu=10,
    valid_batches=4,
    epochs=50,
    lr=1e-3,
    optimizer="adam",
    task=Cifar10,
    n_predictors=-1,
    n_delegators=-1,
    learner=LeLearner
)


if __name__ == "__main__":


    folder = make_train_folder("grid")
    learners = [LeLearner]
    key = jax.random.key(123)
    param_budget = None

    spec_predictors = [2, 4, 8, 16, 32, 64]
    spec_delegators = [0, 1, 2, 4, 8, 16, 32, 64]
    widths_predictors = [8, 32, 64]
    widths_delegators = [2, 32, 64]

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