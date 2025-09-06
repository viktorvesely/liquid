import numpy as np

from .train import init_le, init_moe

def common():
    return {
        "name": "protein",
        "verbose": 0,
        "batch_size": 1800,
        "n_input": 9,
        "n_output": 1,
    }


def rand_int(low, high, log = False):

    if log:
        return int(round(rand_float(low, high, log=True)))

    return int(np.random.randint(low, high))

def rand_float(low, high, log = False):

    if log:
        return float(10 ** np.random.uniform(np.log10(low), np.log10(high)))

    return float(np.random.uniform(low, high))

def rand_cat(choices, w = None):

    if w is not None:
        w = np.array(w)
        w = w / w.sum()

    return np.random.choice(choices, p=w)


def dropout(zero_p = 0.05):
    if np.random.random() <= zero_p:
        return 0

    return rand_float(0, 0.4)

def h_le():

    p_zero_load = 0.2
    p_zero_specialize = 0.2

    load_distribution_lambda = rand_float(0, 1)
    specialization_lambda = rand_float(0, 1)

    if np.random.random() <= p_zero_load:
        load_distribution_lambda = 0.0

    if np.random.random() <= p_zero_specialize:
        specialization_lambda = 0.0

    params = {
        "epoch": rand_int(100, 450),
        "LongLiquid": {
            "lr": rand_float(9 * 1e-5, 5 * 1e-3),
            "architecture": {
                "n_citizens": rand_int(2, 50, log=True),
                "layers_body": rand_cat([0, 1, 2, 3, 4], w=[1, 5, 5, 2, 1]),
                "layers_y": rand_cat([1, 2, 3, 4], w=[5, 5, 2, 1]),
                "layers_d": rand_cat([1, 2, 3, 4], w=[5, 2, 2, 1]),
                "width_body": rand_int(10, 300),
                "width_y": rand_int(10, 300),
                "width_d": rand_int(10, 300),
                "dropout_body": dropout(),
                "dropout_y": dropout(),
                "dropout_d": dropout(),
                "le_kwargs": {
                    "load_distribution_lambda": load_distribution_lambda,
                    "specialization_lambda": specialization_lambda,
                    "solver": "sink_many"
                }
            }
        }
    }

    params = common() | params

    return params

def h_moe():

    p_zero_load = 0.2
    p_zero_specialize = 0.2

    load_distribution_lambda = rand_float(0, 1)
    specialization_lambda = rand_float(0, 1)

    if np.random.random() <= p_zero_load:
        load_distribution_lambda = 0.0

    if np.random.random() <= p_zero_specialize:
        specialization_lambda = 0.0

    params = {
        "epoch": rand_int(100, 450),
        "LongMoe": {
            "lr": rand_float(9 * 1e-5, 5 * 1e-3),
            "architecture": {
                "layers_body": rand_cat([1, 2, 3, 4, 5, 6, 7, 8], w=[1, 2, 3, 4, 4, 3, 2, 1]),
                "width_body": rand_int(10, 300),
                "n_citizens": rand_int(2, 50, log=True),
                "layers_router": rand_cat([1, 2, 3, 4, 5, 6, 7, 8], w=[5, 2, 2, 2, 1, 1, 0.5, 0.5]),
                "width_router": rand_int(10, 800),
                "body_dropout": dropout(),
                "router_dropout": dropout(),
                "moe_kwargs": {
                    "load_distribution_lambda": load_distribution_lambda,
                    "specialization_lambda": specialization_lambda
                }
            }
        }
    }

    params = common() | params

    return params


def zero(zero_p, other_value, zero_value = 0):

    if np.random.random() <= zero_p:
        return zero_value

    return other_value

def h_rf():

    params = {
        "rf": {
            "n_estimators": rand_int(10, 800),
            "max_depth": zero(0.2, rand_int(5, 100), None),
            "min_samples_split": rand_int(2, 40, log=True),
            "min_samples_leaf": rand_int(1, 40, log=True),
            "max_features": 1.0,
            "max_leaf_nodes": zero(0.8, rand_int(10, 1000), None)
        }
    }

    params = common() | params

    return params


def h_lgbm():

    params = {
        "lgbm": {
            "learning_rate": rand_float(0.005, 0.2, log=True),
            "n_estimators": rand_int(10, 800),
            "max_depth": zero(0.2, rand_int(5, 100), -1),
            "num_leaves": rand_int(15, 100),
            "subsample": rand_float(0.7, 1.0),
            "feature_fraction": 1.0,
            "lambda_l2" : rand_float(0, 2),
            "boosting": rand_cat(["gbdt", "dart"], w=[0.8, 0.2]),
            "min_data_in_leaf": rand_int(1, 50)
        }
    }

    params = common() | params

    return params



if __name__ == "__main__":

    from .liquid_ensemble.le_adapter import LiquidLong, LiquidBlock

    params = h_moe()

    moe, _ = init_moe(params, None)

    m, _ = moe.get_nn()
    print(m)
    print("GB", moe.get_size_nbytes() / 1_000_000_000)