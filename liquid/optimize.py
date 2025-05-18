import optuna
import csv
import os

from experiments import setup_multiple_cpu, run_my_task


def objective(trial: optuna.Trial):
    b = trial.suggest_int("b", 1, 4)
    c = trial.suggest_int("c", 1, 4)
    d = trial.suggest_int("d", 1, 4)
    load_lambda = trial.suggest_float("load_lambda", 0.0, 1.0)
    layer_times_step = trial.suggest_int("layer_times_step", 1, 10)
    network_width = trial.suggest_int("network_width", 10, 100)
    return run_my_task(b, c, d, load_lambda, layer_times_step, network_width)

def save_trial_callback(study, trial):
    file_path = "./nmi_optimize.csv"
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        fieldnames = ["trial", "value", "b", "c", "d", "load_lambda", "layer_times_step", "network_width"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "trial": trial.number,
            "value": trial.value,
            "b": trial.params.get("b"),
            "c": trial.params.get("c"),
            "d": trial.params.get("d"),
            "load_lambda": trial.params.get("load_lambda"),
            "layer_times_step": trial.params.get("layer_times_step"),
            "network_width": trial.params.get("network_width")
        })

if __name__ == "__main__":
    setup_multiple_cpu()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, callbacks=[save_trial_callback])
