from functools import partial
from pathlib import Path
from typing import Callable
import jax
import numpy as np
import optuna

from task_base import Task
from structs import TrainParams
from learner_base import Learner

import time
from train import train, make_train_folder
from cifar10 import Cifar10
from learner_le import LeLearner

def objective_validation_performance(
        trial: optuna.Trial,
        train_params: TrainParams,
        objective_metric: str,
        training_metric: str,
        invert_objective_metric: bool = False
    ) -> float:
    
    key = jax.random.key(time.perf_counter_ns())
    jax.clear_caches()
    
    metrics = train(
        key=key,
        train_params=train_params,
        trial=trial
    )

    assert objective_metric in metrics, f"{objective_metric} not in {list(metrics.keys())}"
    assert training_metric in metrics, f"{training_metric} not in {list(metrics.keys())}"

    metric_values = np.array(metrics[objective_metric])
    if invert_objective_metric:
        metric_values = -metric_values
    
    metric = np.min(metric_values)
    when = np.argmin(metric)
    t_loss = np.array(metrics[training_metric])
    inds_p = np.array([0.25, 0.5, 0.75, 1.0])
    inds = ((len(t_loss) - 1) * inds_p).astype(int)
    training_loss = t_loss[inds]
    
    trial.set_user_attr("when_best_metri", when.item())
    for tl, p in zip(training_loss, inds_p, strict=True):
        p = int(p * 100)
        trial.set_user_attr(f"training_loss_{p}", tl.item())

    return metric
    

def optimize(n_trials: int, method: str, folder: Path, objective: Callable) -> None:
    storage_path: str = f"sqlite:///{folder / f'study_{method}.db'}"
    
    study: optuna.Study = optuna.create_study(
        study_name=method,
        storage=storage_path,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=n_trials)
    print(study.best_value)

if __name__ == "__main__":
    
    method = "le"

    folder = make_train_folder(f"hyper_{method}")

    optimize(
        n_trials=1_000,
        method=method,
        folder=folder,
        objective=partial(
            objective_validation_performance,
            train_params= TrainParams(
                batch_size=512,
                preload_batches_to_gpu=5,
                valid_batches=3,
                epochs=50,
                lr=5e-4,
                optimizer="adam",
                performance_loss="ce",
                task=Cifar10,
                n_models_in_ensemble=-1, # Decided by optuna
                learner=LeLearner
            ),
            training_metric="ce_loss",
            objective_metric="validation_accuracy_metric",
            invert_objective_metric=True
        )
    )