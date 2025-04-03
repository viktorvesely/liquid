from itertools import product
import itertools
import math
import numpy as np
import torch.multiprocessing as mp

from train import train

def experiment_load_balancer():

    base_name = "exp_load_solver_{}_{}_{}"

    solvers = ["stable_point", "iterative"]
    balance_lambdas = np.linspace(0, 1.0, num=4, endpoint=True)

    for solver,  balance_lambda in product(solvers, balance_lambdas):

        for variation in range(4):
            exp_name = base_name.format(solver, balance_lambda, variation)

            train(
                experiment_name=exp_name,
                load_distribution_lambda=balance_lambda,
                verbose=1,
                solver=solver,
                epoch=200
            )



def launch_layer(args: tuple[str, int]):
    exp_name, layers = args

    train(
        experiment_name=exp_name,
        load_distribution_lambda=0,
        epoch=300,
        batch_size=2_000,
        verbose=0,
        layers=layers,
        habrok=True,
        citizens_ratio=(1, 1, 1),
    )


def experiment_param_budget():

    base_name = "exp_layer_budget_{}_{}"
    n_variations = 5

    layer_step = 3 * 4
    up_to = 8

    layer_spec = list(range(layer_step, layer_step * (up_to + 1), layer_step))
    runs = []

    for layers, i_variation in product(layer_spec, list(range(n_variations))):
        exp_name = base_name.format(layers, i_variation)
        runs.append((exp_name, layers))


    with mp.Pool(processes=8) as pool:
        for _ in pool.imap_unordered(launch_layer, runs):
            ...


def launch_ratio(args: tuple[str, int]):
    exp_name, ratios = args

    train(
        experiment_name=exp_name,
        load_distribution_lambda=0,
        epoch=300,
        batch_size=2_000,
        verbose=0,
        habrok=True,
        citizens_ratio=ratios
    )

def experiment_ratio():

    base_name = "exp_ratio_{}_{}_{}_{}"
    n_variations = 4

    ratios_specs = list(itertools.permutations([1, 1, 2]))
    ratios_specs += list(itertools.permutations([1, 1, 4]))
    ratios_specs += list(itertools.permutations([2, 1, 2]))
    ratios_specs += list(itertools.permutations([4, 1, 4]))
    ratios_specs.append((1, 1, 1))

    ratios_specs = list(set(ratios_specs))

    runs = []

    for ratios, i_variation in product(ratios_specs, list(range(n_variations))):
        b, c, d = ratios
        exp_name = base_name.format(b, c, d, i_variation)
        runs.append((exp_name, ratios))


    with mp.Pool(processes=8) as pool:
        for _ in pool.imap_unordered(launch_ratio, runs):
            ...

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    experiment_param_budget()

