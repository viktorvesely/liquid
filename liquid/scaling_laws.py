import argparse
import json
import math, random, copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import tqdm

from .train import init_block_le, init_long_le, init_block_moe, init_long_moe, init_simple, dataset_to_numpy, load_data_cifar10
from .nn_adapter import NNAdapter
from .utils import create_experiment_folder



def step(min: int = -1, max: int = 1):
    return random.randint(min, max)

def rstep(val: int, r: float = 0.2):

    return step(int(val * (-r)), int(val * r))

def pertube_le_long(params: dict) -> dict:
    params = copy.deepcopy(params)
    architecture = params["LongLiquid"]["architecture"]
    architecture["le_cnn_kwargs"]["layers_y"] += step()
    architecture["le_cnn_kwargs"]["layers_d"] += step()
    architecture["le_cnn_kwargs"]["layers_body"] += step(-3, 1)

    architecture["le_fc_kwargs"]["layers_y"] += step()
    architecture["le_fc_kwargs"]["layers_d"] += step(0, 1)
    architecture["le_fc_kwargs"]["layers_body"] += step()

    architecture["le_fc_kwargs"]["width_body"] += rstep(architecture["le_fc_kwargs"]["width_body"])
    architecture["le_fc_kwargs"]["width_y"] += rstep(architecture["le_fc_kwargs"]["width_y"])
    architecture["le_fc_kwargs"]["width_d"] += rstep(architecture["le_fc_kwargs"]["width_d"])

    return params

def setup_le_long(
        base_params: dict,
        last_channels: int,
        fc_width: int,
        n_citizens: int,
    ) -> dict:


        params = copy.deepcopy(base_params)
        architecture = params["LongLiquid"]["architecture"]
        architecture["last_channels"] = last_channels
        architecture["le_fc_kwargs"]["width_body"] = fc_width
        architecture["le_fc_kwargs"]["width_y"] = fc_width
        architecture["le_fc_kwargs"]["width_d"] = fc_width // 4
        architecture["n_citizens"] = n_citizens

        return params


def pertube_le_block(params: dict) -> dict:
    params = copy.deepcopy(params)
    architecture = params["BlockLiquid"]["architecture"]
    architecture["n_cnn_le_blocks"] += step(-2, 2)
    architecture["n_fc_le_blocks"] += step()

    architecture["le_fc_kwargs"]["width_body"] += rstep(architecture["le_fc_kwargs"]["width_body"])
    architecture["le_fc_kwargs"]["width_y"] += rstep(architecture["le_fc_kwargs"]["width_y"])
    architecture["le_fc_kwargs"]["width_d"] += rstep(architecture["le_fc_kwargs"]["width_d"])

    return params

def setup_le_block(
        base_params: dict,
        last_channels: int,
        fc_width: int,
        n_citizens: int,
    ) -> dict:


        params = copy.deepcopy(base_params)
        architecture = params["BlockLiquid"]["architecture"]
        architecture["last_channels"] = last_channels
        architecture["le_fc_kwargs"]["width_body"] = fc_width
        architecture["le_fc_kwargs"]["width_y"] = fc_width
        architecture["le_fc_kwargs"]["width_d"] = fc_width // 2
        architecture["n_citizens"] = n_citizens

        return params



def pertube_moe_long(params: dict) -> dict:
    params = copy.deepcopy(params)
    architecture = params["LongMoe"]["architecture"]
    architecture["moe_cnn_kwargs"]["layers"] += step(-2, 2)
    architecture["moe_fc_kwargs"]["layers"] += step()
    architecture["router_cnn_kwargs"]["layers"] += step()

    architecture["moe_fc_kwargs"]["width"] += rstep(architecture["moe_fc_kwargs"]["width"])
    architecture["router_fc_kwargs"]["width"] += rstep(architecture["router_fc_kwargs"]["width"])

    return params

def setup_moe_long(
        base_params: dict,
        last_channels: int,
        fc_width: int,
        n_citizens: int,
    ) -> dict:


        params = copy.deepcopy(base_params)
        architecture = params["LongMoe"]["architecture"]
        architecture["last_channels"] = last_channels
        architecture["moe_fc_kwargs"]["width"] = fc_width
        architecture["n_citizens"] = n_citizens

        return params

def pertube_moe_block(params: dict) -> dict:
    params = copy.deepcopy(params)
    architecture = params["BlockMoe"]["architecture"]
    architecture["n_cnn_moe_blocks"] += step(-2, 2)
    architecture["n_fc_moe_blocks"] += step()

    architecture["moe_fc_kwargs"]["width"] += rstep(architecture["moe_fc_kwargs"]["width"])
    architecture["router_fc_kwargs"]["width"] += rstep(architecture["router_fc_kwargs"]["width"])

    return params

def setup_moe_block(
        base_params: dict,
        last_channels: int,
        fc_width: int,
        n_citizens: int,
    ) -> dict:


        params = copy.deepcopy(base_params)
        architecture = params["BlockMoe"]["architecture"]
        architecture["last_channels"] = last_channels
        architecture["moe_fc_kwargs"]["width"] = fc_width
        architecture["n_citizens"] = n_citizens

        return params


def pertube_simple(params: dict) -> dict:
    params = copy.deepcopy(params)
    architecture = params["SimpleNN"]["architecture"]
    architecture["n_cnn_layers"] += step(-2, 2)
    architecture["n_fc_layers"] += step(-2, 2)

    architecture["fc_kwargs"]["width"] += rstep(architecture["fc_kwargs"]["width"])

    return params

def setup_simple(
        base_params: dict,
        last_channels: int,
        fc_width: int,
        n_citizens: int,
    ) -> dict:

        params = copy.deepcopy(base_params)
        architecture = params["SimpleNN"]["architecture"]
        architecture["last_channels"] = last_channels
        architecture["fc_kwargs"]["width"] = fc_width

        return params

INIT_FNS: dict[str, Callable[[dict, Path], tuple[NNAdapter, dict]]] = {
    "LongLiquid": init_long_le,
    "BlockLiquid": init_block_le,
    "LongMoe": init_long_moe,
    "BlockMoe": init_block_moe,
    "SimpleNN": init_simple
}

params_funcs = {
    "LongLiquid": (setup_le_long, pertube_le_long),
    "BlockLiquid": (setup_le_block, pertube_le_block),
    "LongMoe": (setup_moe_long, pertube_moe_long),
    "BlockMoe": (setup_moe_block, pertube_moe_block),
    "SimpleNN": (setup_simple, pertube_simple),
}


def yield_architectures(
    arch_name: str,
    cnn_fc_widths: list[tuple[int, int]],
    n_citizens: int,
    exp_name: str | None = None,
    variations: int = 1
):

    with open(Path(__file__).parent / "cifar10.json", "r") as f:
        base_params = json.load(f)

    init_func = INIT_FNS[arch_name]
    setup_func, _ = params_funcs[arch_name]

    for last_channels, fc_width in tqdm.tqdm(list(cnn_fc_widths), disable=True):

        set_up_params: dict = setup_func(base_params, last_channels=last_channels, fc_width=fc_width, n_citizens=n_citizens)

        for i_var in range(variations):

            if exp_name is not None:
                path = create_experiment_folder(task="cifar10", name=exp_name, hyper=True, rand=True)
            else:
                path = None
            yield init_func(set_up_params, path), set_up_params, path


def train_arch_variations(
        exp_name: str,
        _range: tuple[int, int],
        n_citizens: int,
        N: int,
        variations: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        arch_name: str
    ):

    cnns = np.linspace(*_range, num=N, endpoint=True).astype(int)
    fcs = np.linspace(*_range, num=N, endpoint=True).astype(int)

    cnns = [int(x) for x in cnns]
    fcs = [int(x) for x in fcs]


    for (model, train_kwargs), params, path in yield_architectures(arch_name, zip(cnns, fcs), variations=variations, n_citizens=n_citizens, exp_name=exp_name):

        save_params = {k: v for k, v in params.items() if (not isinstance(v, dict) or (k == arch_name))}

        with open(path / "params.json", "w") as f:
            json.dump(save_params, f)

        try:
            model.train(x_train, y_train, x_val, y_val, **train_kwargs)
            model.evaluate_p_active_params(x_val, y_val)
            model.save_metrics()
        except Exception as e:
            print(e)


def count_arch_variations(
        _range: tuple[int, int],
        n_citizens: int,
        N: int,
        arch_name: str
    ) -> list[int]:

    cnns = np.linspace(*_range, num=N, endpoint=True).astype(int)
    fcs = np.linspace(*_range, num=N, endpoint=True).astype(int)

    cnns = [int(x) for x in cnns]
    fcs = [int(x) for x in fcs]

    params = []
    for (model, _), _, _ in yield_architectures(arch_name, zip(cnns, fcs), n_citizens=n_citizens):
        params.append(model.get_size_nparams())

    return params


def train(
    arch_name: str,
    experiment_prefix: str,
    N: int = 16,
    variations: int = 3
):

    train_dataset, val_dataset = load_data_cifar10(reduction=1.0)

    x_train, y_train = dataset_to_numpy(train_dataset)
    del train_dataset

    x_val, y_val = dataset_to_numpy(val_dataset)
    del val_dataset

    exp_base = f"{experiment_prefix}_{arch_name}"

    if arch_name == "LongLiquid":
        train_arch_variations(f"{exp_base}_{5}", (27, 67), n_citizens=5,  N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{10}", (20, 47), n_citizens=10, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{15}", (16, 37), n_citizens=15, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
    elif arch_name == "BlockLiquid":
        train_arch_variations(f"{exp_base}_{5}", (25, 62), n_citizens=5,  N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{10}", (18, 42), n_citizens=10, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{15}", (13, 33), n_citizens=15, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
    elif arch_name == "LongMoe":
        train_arch_variations(f"{exp_base}_{5}", (38, 86), n_citizens=5,  N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{10}", (25, 62), n_citizens=10, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{15}", (22, 50), n_citizens=15, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
    elif arch_name == "BlockMoe":
        train_arch_variations(f"{exp_base}_{5}", (30, 70), n_citizens=5,  N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{10}", (22, 52), n_citizens=10, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
        train_arch_variations(f"{exp_base}_{15}", (17, 42), n_citizens=15, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
    elif arch_name == "SimpleNN":
        train_arch_variations(exp_base, (75, 180), n_citizens=None, N=N, variations=variations, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, arch_name=arch_name)
    else:
        raise ValueError(arch_name)


def plot_sizes():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    N = 10

    # collect all parameter distributions
    lle_param_c_5  = count_arch_variations((27, 67), n_citizens=5,  N=N, arch_name="LongLiquid")
    lle_param_c_10 = count_arch_variations((20, 47), n_citizens=10, N=N, arch_name="LongLiquid")
    lle_param_c_15 = count_arch_variations((16, 37), n_citizens=15, N=N, arch_name="LongLiquid")

    ble_param_c_5  = count_arch_variations((25, 62), n_citizens=5,  N=N, arch_name="BlockLiquid")
    ble_param_c_10 = count_arch_variations((18, 42), n_citizens=10, N=N, arch_name="BlockLiquid")
    ble_param_c_15 = count_arch_variations((13, 33), n_citizens=15, N=N, arch_name="BlockLiquid")

    lmoe_param_c_5  = count_arch_variations((38, 86), n_citizens=5,  N=N, arch_name="LongMoe")
    lmoe_param_c_10 = count_arch_variations((25, 62), n_citizens=10, N=N, arch_name="LongMoe")
    lmoe_param_c_15 = count_arch_variations((22, 50), n_citizens=15, N=N, arch_name="LongMoe")

    bmoe_param_c_5  = count_arch_variations((30, 70), n_citizens=5,  N=N, arch_name="BlockMoe")
    bmoe_param_c_10 = count_arch_variations((22, 52), n_citizens=10, N=N, arch_name="BlockMoe")
    bmoe_param_c_15 = count_arch_variations((17, 42), n_citizens=15, N=N, arch_name="BlockMoe")

    simple_param = count_arch_variations((75, 180), n_citizens=None, N=N, arch_name="SimpleNN")

    fig, ax = plt.subplots()

    # mapping
    colors = {
        "LongLiquid": "tab:blue",
        "BlockLiquid": "tab:orange",
        "LongMoe": "tab:green",
        "BlockMoe": "tab:red",
        "SimpleNN": "tab:purple",
    }
    markers = {5: "o", 10: "s", 15: "D"}

    # data container for labeling
    ytick_labels = []
    ytick_pos = []

    # iterate and plot
    index = 0
    for label, params in [
        ("LongLiquid", [lle_param_c_5, lle_param_c_10, lle_param_c_15]),
        ("BlockLiquid", [ble_param_c_5, ble_param_c_10, ble_param_c_15]),
        ("LongMoe", [lmoe_param_c_5, lmoe_param_c_10, lmoe_param_c_15]),
        ("BlockMoe", [bmoe_param_c_5, bmoe_param_c_10, bmoe_param_c_15]),
        ("SimpleNN", [simple_param]),
    ]:
        for n_citizens, param in zip([5, 10, 15][:len(params)], params):

            if label == "SimpleNN":
                n_citizens = None

            ytick_labels.append(f"{label}:{n_citizens}" if n_citizens is not None else f"{label}")
            ytick_pos.append(index)
            ax.scatter(
                param, [index] * len(param),
                color=colors[label],
                marker=markers.get(n_citizens, "x"),
                alpha=0.7,
            )
            index += 1

    # apply yticks
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels)

    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Architecture:n_citizens")

    ax.axvline(200_000, linestyle="dashed", color="black")
    ax.axvline(1_000_000, linestyle="dashed", color="black")

    # optional legend
    # color_handles = [Line2D([0], [0], color=c, lw=0, marker="o") for c in colors.values()]
    # color_labels = list(colors.keys())
    # legend1 = ax.legend(color_handles, color_labels, title="Architecture", loc="upper right")

    # marker_handles = [Line2D([0], [0], color="black", lw=0, marker=m) for m in markers.values()]
    # marker_labels = [f"{n} citizens" for n in markers.keys()]
    # legend2 = ax.legend(marker_handles, marker_labels, title="Citizens", loc="lower right")

    # ax.add_artist(legend1)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--prefix", type=str, default="experiment")
    args = parser.parse_args()

    train(
        arch_name=args.algorithm,
        experiment_prefix=args.prefix
    )

    # plot_sizes()

