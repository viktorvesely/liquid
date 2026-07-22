from dataclasses import dataclass
from itertools import product
from math import prod
from random import Random
from typing import Literal

import jax
from matplotlib import pyplot as plt

from atomic_networks import Architecture, big_cnn, three_layer_mlp, two_layer_mlp
from bikes import Bikes
from cifar10 import Cifar10
from energy import Energy
from learner_le import LeLearner
from structs import TrainParams
from svhn import Svhn
from task_base import Task
from train import finish_run, make_train_folder, train


type Mode = Literal["grid", "random"]
type Mixing = Literal["sum", "product"]
type AmbiguityGradient = Literal["both", "delegators", "none"]
type TaskType = type[Task]


@dataclass(frozen=True, slots=True)
class Pool[T]:
    values: tuple[T, ...]
    mode: Mode

    @classmethod
    def grid(cls, *values: T) -> "Pool[T]":
        return cls(values, "grid")

    @classmethod
    def random(cls, *values: T) -> "Pool[T]":
        return cls(values, "random")
    
    @classmethod
    def constant(cls, value: T) -> "Pool[T]":
        return cls.random(value)

    def largest(self) -> T:
        return max(self.values) if all(isinstance(x, int | float) for x in self.values) else self.values[0]

    def ordered(self) -> tuple[T, ...]:
        return tuple(sorted(self.values, reverse=True)) if all(
            isinstance(x, int | float) for x in self.values
        ) else self.values

    def sample(self, rng: Random) -> T:
        return rng.choice(self.values)


@dataclass(frozen=True, slots=True)
class TaskProfile:
    batch_size: int
    preload_batches_to_gpu: int
    valid_batches: int
    epochs: int
    architecture: Architecture


TASKS: tuple[TaskType, ...] = Cifar10, Svhn, Bikes, Energy
TASK_BY_NAME = {task.__name__: task for task in TASKS}

TASK_PROFILES: dict[TaskType, TaskProfile] = {
    Cifar10: TaskProfile(
        batch_size=64,
        preload_batches_to_gpu=25,
        valid_batches=20,
        epochs=100,
        architecture=big_cnn,
    ),
    Svhn: TaskProfile(64, 20, 20, 50, small_cnn),
    Bikes: TaskProfile(256, 50, 7, 1_000, two_layer_mlp),
    Energy: TaskProfile(256, 50, 7, 2_000, three_layer_mlp),
}

N_PREDICTORS = 2, 4, 8, 16, 32, 64
N_DELEGATORS = 0, 1, 2, 4, 8, 16, 32, 64
CNN_WIDTHS = 1, 4, 8
MLP_WIDTHS = 4, 8, 16
MIXING: tuple[Mixing, ...] = "sum", "product"
AMBIGUITY_GRADIENTS: tuple[AmbiguityGradient, ...] = "both", "delegators", "none"


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    run_id: int
    task: TaskType
    n_predictors: int
    n_delegators: int
    width_predictors: int
    width_delegators: int
    delegators_mixing: Mixing
    ambiguity_gradient: AmbiguityGradient

    @property
    def name(self) -> str:
        return (
            f"run_{self.run_id:05d}"
            f"_task_{self.task.__name__}"
            f"_predictors_{self.n_predictors}"
            f"_delegators_{self.n_delegators}"
            f"_pwidth_{self.width_predictors}"
            f"_dwidth_{self.width_delegators}"
            f"_mixing_{self.delegators_mixing}"
            f"_ambiguity_{self.ambiguity_gradient}"
        )


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    task: Pool[TaskType]
    n_predictors: Pool[int]
    n_delegators: Pool[int]
    width_predictors: Pool[int]
    width_delegators: Pool[int]
    delegators_mixing: Pool[Mixing]
    ambiguity_gradient: Pool[AmbiguityGradient]
    max_iterations: int | None = None
    seed: int = 123

    @property
    def pools(self) -> dict[str, Pool]:
        return {
            "task": self.task,
            "n_predictors": self.n_predictors,
            "n_delegators": self.n_delegators,
            "width_predictors": self.width_predictors,
            "width_delegators": self.width_delegators,
            "delegators_mixing": self.delegators_mixing,
            "ambiguity_gradient": self.ambiguity_gradient,
        }

    def cases(self) -> list[ExperimentCase]:
        rng = Random(self.seed)
        grid = {name: pool for name, pool in self.pools.items() if pool.mode == "grid"}
        random = {name: pool for name, pool in self.pools.items() if pool.mode == "random"}

        if not grid:
            if self.max_iterations is None:
                raise ValueError("max_iterations is required for a fully random experiment")

            values = [
                {name: pool.largest() for name, pool in self.pools.items()},
                *(
                    {
                        name: pool.sample(rng)
                        for name, pool in self.pools.items()
                    }
                    for _ in range(self.max_iterations - 1)
                ),
            ]
        else:
            names = tuple(grid)
            combinations = product(*(pool.ordered() for pool in grid.values()))
            values = []

            for run_id, combination in enumerate(combinations):
                case = dict(zip(names, combination, strict=True))
                case.update(
                    {
                        name: pool.largest() if run_id == 0 else pool.sample(rng)
                        for name, pool in random.items()
                    }
                )
                values.append(case)

        return [
            ExperimentCase(run_id=run_id, **case)
            for run_id, case in enumerate(values, start=1)
        ]

    @property
    def iterations(self) -> int:
        grid = [pool for pool in self.pools.values() if pool.mode == "grid"]
        return prod(len(pool.values) for pool in grid) if grid else self.max_iterations or 0

    def params(self, case: ExperimentCase) -> TrainParams:
        profile = TASK_PROFILES[case.task]

        return TrainParams(
            batch_size=profile.batch_size,
            preload_batches_to_gpu=profile.preload_batches_to_gpu,
            valid_batches=profile.valid_batches,
            epochs=profile.epochs,
            lr=1e-3,
            optimizer="adam",
            task=case.task,
            n_predictors=case.n_predictors,
            n_delegators=case.n_delegators,
            delegators_mixing=case.delegators_mixing,
            ambiguity_gradient=case.ambiguity_gradient,
            architecture=profile.architecture.determine_size(
                predictor_base=case.width_predictors,
                delegator_base=case.width_delegators,
                out_dim=case.task.out_dim(),
                n_predictors=case.n_predictors,
            ),
            learner=LeLearner,
        )

    def run(self) -> None:
        folder = make_train_folder(self.name)
        key = jax.random.key(self.seed)
        cases = self.cases()

        for index, case in enumerate(cases, start=1):
            jax.clear_caches()
            plt.close("all")
            key, run_key = jax.random.split(key)

            print(case.name)
            metrics = train(run_key, self.params(case))
            finish_run(metrics, folder, prefix=case.name)
            print(f"{index} / {len(cases)}")


experiment_aggregation_method = Experiment(
    name="exp_aggregation",
    task=Pool.random(Cifar10, Svhn, Bikes, Energy),
    n_predictors=Pool.random(*N_PREDICTORS),
    n_delegators=Pool.random(*N_DELEGATORS),
    width_predictors=Pool.random()
)