import numpy as np
import tqdm
from .liquid_ensemble.le_adapter import LiquidLong, LiquidBlock
from .moe.moe_adapter import Moe
import json

def sample_config(config: dict) -> dict:
    def parse_key_value(key, value):
        parts = key.split(":")
        base_key = parts[0]
        key_type = parts[1:] if len(parts) > 1 else []

        if isinstance(value, list) and len(value) == 2:
            low, high = value
            if "int" in key_type:
                return base_key, int(np.random.randint(low, high + 1))
            elif "float" in key_type and "log" in key_type:
                return base_key, float(10 ** np.random.uniform(np.log10(low), np.log10(high)))
            elif "float" in key_type:
                return base_key, float(np.random.uniform(low, high))
        elif isinstance(value, dict):
            return base_key, {k.split(":")[0]: parse_key_value(k, v)[1] for k, v in value.items()}
        else:
            return base_key, value

    return {k.split(":")[0]: parse_key_value(k, v)[1] for k, v in config.items()}



if __name__ == "__main__":

    N = 1_000

    from pathlib import Path
    import pandas as pd

    with open(Path(__file__).parent / "cifar10_hyper.json", "r") as f:
        wave_params = json.load(f)

    n_input = wave_params["n_input"]
    n_output = wave_params["n_output"]

    sizes = {
        LiquidLong.name(): [],
        LiquidBlock.name(): [],
        Moe.name(): []
    }
    for _ in tqdm.tqdm(list(range(N)), disable=False):
        collapsed_params = sample_config(wave_params)

        ll = collapsed_params[LiquidLong.name()]
        le = LiquidLong(
            n_input=n_input,
            n_output=n_output,
            folder=None,
            lr=ll["lr"],
        )
        le.init_model(model_kwargs=ll["architecture"])
        sizes[LiquidLong.name()].append(le.get_size_nbytes())

        ll = collapsed_params[LiquidBlock.name()]
        le = LiquidBlock(
            n_input=n_input,
            n_output=n_output,
            folder=None,
            lr=ll["lr"],
        )
        le.init_model(model_kwargs=ll["architecture"])
        sizes[LiquidBlock.name()].append(le.get_size_nbytes())


        ll = collapsed_params[Moe.name()]
        le = Moe(
            n_input=n_input,
            n_output=n_output,
            folder=None,
            lr=ll["lr"],
        )
        experts = ll["architecture"]["n_citizens"]
        ll["architecture"]["moe_kwargs"]["k_active"] = min(ll["architecture"]["moe_kwargs"]["k_active"], experts)
        le.init_model(model_kwargs=ll["architecture"])
        sizes[Moe.name()].append(le.get_size_nbytes())

    for name, szs in sizes.items():
        print(f"\n\n {name}")
        print(pd.Series(szs).describe())




