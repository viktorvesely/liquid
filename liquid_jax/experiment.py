from dataclasses import dataclass
from itertools import product
from math import prod
from pathlib import Path
from random import Random
import secrets
from typing import Literal

import jax
from matplotlib import pyplot as plt

from atomic_networks import Architecture, big_cnn, three_layer_mlp, two_layer_mlp
from bikes import Bikes
from cifar10 import Cifar10
from energy import Energy
from structs import TrainParams
from svhn import Svhn
from task_base import Task
from train import finish_run, make_train_folder, train


type Mode = Literal["grid", "random", "paired"]
type Mixing = Literal["sum", "product"]
type AmbiguityGradientPredictor = Literal["both", "delegators", "none"]
type AmbiguityGradientDelegator = bool
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
    def paired(cls, *values: T) -> "Pool[T]":
        return cls(values, "paired")
    
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

    def __len__(self):
        return len(self.values)


@dataclass(frozen=True, slots=True)
class TaskProfile:
    batch_size: int
    preload_batches_to_gpu: int
    valid_batches: int
    epochs: int
    architecture: Architecture


TASKS: tuple[TaskType, ...] = Cifar10, Svhn, Bikes, Energy
TASK_BY_NAME = {task.__name__: task for task in TASKS}

img_task_profile = TaskProfile(
    batch_size=128,
    preload_batches_to_gpu=25,
    valid_batches=20,
    epochs=100,
    architecture=three_layer_mlp,
)

tab_task_profile_energy  = TaskProfile(
    batch_size=256,
    preload_batches_to_gpu=13,
    valid_batches=10,
    epochs=2_000,
    architecture=two_layer_mlp,
)

tab_task_profile_bikes  = TaskProfile(
    batch_size=256,
    preload_batches_to_gpu=13,
    valid_batches=10,
    epochs=1_000,
    architecture=two_layer_mlp,
)

TASK_PROFILES: dict[TaskType, TaskProfile] = {
    Cifar10: img_task_profile,
    Svhn: img_task_profile,
    Bikes: tab_task_profile_bikes,
    Energy: tab_task_profile_energy
}


N_PREDICTORS = 2, 4, 8, 16, 32
N_DELEGATORS_ENSEMBLE = 2, 4, 8, 16, 32
N_DELEGATORS_VALID = (1,) + N_DELEGATORS_ENSEMBLE
N_DELEGATORS = (0,) + N_DELEGATORS_VALID
N_DELEGATORS_MULTIPLE = 2, 4, 8, 16, 32
CNN_WIDTHS = 1, 4, 8
MLP_WIDTHS = 4, 8, 16
MIXING: tuple[Mixing, ...] = "sum", "product"
AMBIGUITY_GRADIENTS_PREDICTORS: tuple[AmbiguityGradientPredictor, ...] = "both", "delegators", "none"
AMBIGUITY_GRADIENTS_DELEGATORS: tuple[AmbiguityGradientDelegator, ...] = True, False

@dataclass(frozen=True, slots=True)
class ExperimentCase:
    run_id: int
    n_predictors: int
    n_delegators: int
    width_predictors: int
    width_delegators: int
    delegators_mixing: Mixing
    ambiguity_gradient_predictor: AmbiguityGradientPredictor
    ambiguity_gradient_delegator: AmbiguityGradientDelegator
    key: jax.Array

    @classmethod
    def func_same(cls, a: "ExperimentCase", b: "ExperimentCase") -> bool:
        return (
            a.n_predictors == b.n_predictors and
            a.n_delegators == b.n_delegators and
            a.width_predictors == b.width_predictors and
            a.width_delegators == b.width_delegators and
            a.delegators_mixing == b.delegators_mixing and
            a.ambiguity_gradient_predictor == b.ambiguity_gradient_predictor and
            a.ambiguity_gradient_delegator == b.ambiguity_gradient_delegator
        )

    @property
    def name(self) -> str:
        return (
            f"run_{self.run_id:05d}"
            f"_predictors_{self.n_predictors}"
            f"_delegators_{self.n_delegators}"
            f"_pwidth_{self.width_predictors}"
            f"_dwidth_{self.width_delegators}"
            f"_mixing_{self.delegators_mixing}"
            f"_ambiguitypred_{self.ambiguity_gradient_predictor}"
            f"_ambiguitydele_{self.ambiguity_gradient_delegator}"
        )


def random_hex(n: int) -> str:
    return secrets.token_hex((n + 1) // 2)[:n]

@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    n_predictors: Pool[int]
    n_delegators: Pool[int]
    width_predictors: Pool[int]
    width_delegators: Pool[int]
    delegators_mixing: Pool[Mixing]
    ambiguity_gradient_predictor: Pool[AmbiguityGradientPredictor]
    ambiguity_gradient_delegator: Pool[AmbiguityGradientDelegator]
    max_iterations: int | None = None
    seed: int = 123
        

    @property
    def pools(self) -> dict[str, Pool]:
        return {
            "n_predictors": self.n_predictors,
            "n_delegators": self.n_delegators,
            "width_predictors": self.width_predictors,
            "width_delegators": self.width_delegators,
            "delegators_mixing": self.delegators_mixing,
            "ambiguity_gradient_predictor": self.ambiguity_gradient_predictor,
            "ambiguity_gradient_delegator": self.ambiguity_gradient_delegator,
        }


    def cases(self, key: jax.Array) -> list[ExperimentCase]:
        rng = Random(self.seed)
        grid = {name: pool for name, pool in self.pools.items() if pool.mode == "grid"}
        random = {name: pool for name, pool in self.pools.items() if pool.mode == "random"}
        paired = {name: pool for name, pool in self.pools.items() if pool.mode == "paired"}


        if len(paired) > 0:
            assert len(paired) < 2, f"Don't you wanna use grid or split it to multiple experiments? {paired}"
            paired_name = tuple(paired.keys())[0]
            paired_pool = tuple(paired.values())[0]
            pool_size = len(paired_pool)

            assert len(grid) == 0, "Don't know what to do both with grid and paired"

            if self.max_iterations is None:
                raise ValueError("max_iterations is required for a fully random experiment")

            assert (self.max_iterations % pool_size) == 0, "Max iterations needs to be divisible by the #unique values"

            # Largest one for all uniques

            first_key = jax.random.fold_in(key, self.max_iterations + 1)
            values = [
                ({name: pool.largest() for name, pool in random.items()} | {paired_name: pool_value, "key": first_key}) for pool_value in paired_pool.values 
            ]

            # Rest
            for i_iteration in range((self.max_iterations - pool_size) // pool_size):

                same_across = {
                    name: pool.sample(rng)
                    for name, pool in random.items()
                }

                same_across["key"] = jax.random.fold_in(key, i_iteration)

                for pool_value in paired_pool.values:
                    value = same_across | {paired_name: pool_value}
                    values.append(value)

        elif len(grid) == 0:
            assert False, "Not supported anymore"
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
                case["key"] = key
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

    def params(self, case: ExperimentCase, task: type[Task]) -> TrainParams:
        profile = TASK_PROFILES[task]

        return TrainParams(
            batch_size=profile.batch_size,
            preload_batches_to_gpu=profile.preload_batches_to_gpu,
            valid_batches=profile.valid_batches,
            epochs=profile.epochs,
            lr=1e-3,
            task=task,
            n_predictors=case.n_predictors,
            n_delegators=case.n_delegators,
            delegators_mixing=case.delegators_mixing,
            ambiguity_gradient_predictors=case.ambiguity_gradient_predictor,
            ambiguity_gradient_delegators=case.ambiguity_gradient_delegator,
            architecture=profile.architecture.determine_size(
                predictor_base=case.width_predictors,
                delegator_base=case.width_delegators,
                out_dim=task.out_dim(),
                n_predictors=case.n_predictors,
            ),
        )


    def is_paired(self) -> bool:
        return len({name: pool for name, pool in self.pools.items() if pool.mode == "paired"}) > 0

    def is_grid(self) -> bool:
        return len({name: pool for name, pool in self.pools.items() if pool.mode == "grid"}) > 0

    def get_finished_cases(self, resume_folder: Path) -> list[ExperimentCase]:

        cases = []

        name_mapper = {
            "run": "run_id",
            "delegators": "n_delegators",
            "predictors": "n_predictors",
            "dwidth": "width_delegators",
            "pwidth": "width_predictors",
            "mixing": "delegators_mixing",
            "ambiguitypred": "ambiguity_gradient_predictor",
            "ambiguitydele": "ambiguity_gradient_delegator"
        }

        for mf in resume_folder.glob("*_metrics.json"):

            info = mf.stem.removesuffix("_metrics").split("_")
            case_dict = {}

            for name, str_value in zip(info[::2], info[1::2], strict=True):

                name = name_mapper[name]

                try:
                    value = int(str_value)
                except ValueError:

                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                    else:
                        value = str_value

                case_dict[name] = value

            case_dict["key"] = None

            cases.append(ExperimentCase(**case_dict))

        cases = sorted(cases, key=lambda case: case.run_id)
        return cases
            

    def run(self, task: type[Task], resume_folder: Path | None = None) -> None:


        new = resume_folder is None

        if new:
            launch_id = random_hex(n=10)
            folder = make_train_folder(f"{self.name}_{launch_id}_{task.__name__}")
        else:
            folder = resume_folder
            launch_id = resume_folder.name.split("_")[2]

        key = jax.random.key(self.seed)
        cases = self.cases(key)

        n_finished_cases = 0
        n_all_cases = len(cases) 

        if not new:
            finished_cases = self.get_finished_cases(resume_folder)
            n_finished_cases = len(finished_cases)

            for proposed_case, finished_case in zip(cases, finished_cases, strict=False):

                if not ExperimentCase.func_same(proposed_case, finished_case):
                    print(proposed_case)
                    print(finished_cases)
                    raise ValueError(f"Not equal cases")

            cases = cases[len(finished_cases):]

        for visual_index, case in enumerate(cases, start=(1 + n_finished_cases)):
            jax.clear_caches()
            plt.close("all")

            print(case.name)
            metrics, eval_metrics = train(
                key=case.key,
                train_params=self.params(case, task)
            )
            
            finish_run(metrics, eval_metrics, folder, prefix=case.name)
            print(f"{visual_index} / {n_all_cases}")


experiment_aggregation_method = Experiment(
    name="exp_aggregation",
    n_predictors=Pool.random(*N_PREDICTORS),
    n_delegators=Pool.random(*N_DELEGATORS_ENSEMBLE),
    width_predictors=Pool.random(*MLP_WIDTHS),
    width_delegators=Pool.random(*MLP_WIDTHS),
    delegators_mixing=Pool.paired(*MIXING),
    ambiguity_gradient_predictor=Pool.constant("none"),
    ambiguity_gradient_delegator=Pool.constant(True), # Only valid there
    max_iterations=200
)

exp_ambiguity_gradient = Experiment(
    name="exp_ambiguity_gradient",
    n_predictors=Pool.random(*N_PREDICTORS),
    n_delegators=Pool.random(*N_DELEGATORS_VALID),
    width_predictors=Pool.random(*MLP_WIDTHS),
    width_delegators=Pool.random(*MLP_WIDTHS),
    delegators_mixing=Pool.constant("sum"),
    ambiguity_gradient_delegator=Pool.random(True),
    ambiguity_gradient_predictor=Pool.paired(*AMBIGUITY_GRADIENTS_PREDICTORS),
    max_iterations=300
)

exp_ambiguity_delgradient = Experiment(
    name="exp_ambiguity_delgradient",
    n_predictors=Pool.random(*N_PREDICTORS),
    n_delegators=Pool.random(*N_DELEGATORS_VALID),
    width_predictors=Pool.random(*MLP_WIDTHS),
    width_delegators=Pool.random(*MLP_WIDTHS),
    delegators_mixing=Pool.constant("sum"),
    ambiguity_gradient_predictor=Pool.constant("delegators"),
    ambiguity_gradient_delegator=Pool.paired(*AMBIGUITY_GRADIENTS_DELEGATORS),
    max_iterations=200
)

experiment_scaling = Experiment(
    name="exp_scaling",
    n_predictors=Pool.grid(*N_PREDICTORS),
    n_delegators=Pool.grid(*N_DELEGATORS),
    width_predictors=Pool.grid(*MLP_WIDTHS),
    width_delegators=Pool.grid(*MLP_WIDTHS),
    delegators_mixing=Pool.constant("sum"), 
    ambiguity_gradient_predictor=Pool.constant("delegators"),
    ambiguity_gradient_delegator=Pool.constant(False) 
)

if __name__ ==  "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=("agg", "gradientpred", "gradientdel", "scaling"),
    )
    parser.add_argument(
        "task",
        choices=list(TASK_BY_NAME.keys()),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    experiments = {
        "agg": experiment_aggregation_method,
        "gradientpred": exp_ambiguity_gradient,
        "gradientdel":  exp_ambiguity_delgradient,
        "scaling": experiment_scaling,
    }

    this_experiment = experiments[args.experiment]
    this_task = TASK_BY_NAME[args.task]

    this_experiment.run(this_task, resume_folder=args.resume)


