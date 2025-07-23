import copy
import json
from pathlib import Path
import random
import re
from typing import Literal, Self
import numpy as np

import lightgbm as lgb
from lightgbm import log_evaluation

from ..adapter import Adapter

class LightGBM(Adapter):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        task: str,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int | None = None,
        num_leaves: int | None = None,
        subsample: float = 1.0,
        feature_fraction: float = 1.0,
        lambda_l2 : float = 0.0,
        boosting: Literal["gbdt", "dart"] = "gbdt",
        min_data_in_leaf: int = 20,
        estimate_confidence: bool = False
    ):
        super().__init__(n_input, n_output, folder, task=task)

        self.constructor = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "feature_fraction": feature_fraction,
            "lambda_l2": lambda_l2,
            "boosting": boosting,
            "min_data_in_leaf": min_data_in_leaf
        }

        self.estimate_confidence = estimate_confidence
        self.n_estimators = n_estimators
        self.model: lgb.Booster | None = None


    def init_model(self):

        objective ="multiclass" if self.get_task_type() == "classification" else "mse"
        metric = "multi_logloss" if self.get_task_type() == "classification" else "rmse"

        params = self.constructor | {
            "objective": objective,
            "metric": metric,
            "num_class": self.n_output
        }

        self._params = {k: v for k, v in params.items() if v is not None}

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        verbose: int,
        **kwargs
    ) -> float:

        y = np.squeeze(y)
        y_val = np.squeeze(y_val)

        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], -1))
            x_val = np.reshape(x_val, (x_val.shape[0], -1))

        train_data = lgb.Dataset(x, label=y)
        valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

        params = copy.deepcopy(self._params)
        params["verbosity"] = verbose

        self._train_start = self.now()
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data],
        callbacks=[log_evaluation(period=0)]
        )
        self._train_end = self.now()

        if self.estimate_confidence:
            up = params.copy()
            up["objective"] = "quantile"
            up["alpha"] = 0.95
            self.upper = lgb.train(
                up,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[valid_data],
                callbacks=[log_evaluation(period=0)]
            )

            lp = params.copy()
            lp["objective"] = "quantile"
            lp["alpha"] = 0.05
            self.lower = lgb.train(
                lp,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[valid_data],
                callbacks=[log_evaluation(period=0)]
            )


        y_hat_train = self.inference(x)
        y_hat_val = self.inference(x_val)

        train_metric = self.calc_task_metric(y_hat_train, y)
        val_metric = self.calc_task_metric(y_hat_val, y_val)

        metric_name = self.get_task_metric_name()
        metrics = {
            f"train_{metric_name}": train_metric,
            metric_name: val_metric
        }

        self.set_test_metrics(**metrics)

    def inference(self, x: np.ndarray) -> np.ndarray:

        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], -1))

        preds = self.model.predict(x)

        if self.get_task_type() == "classification":
            raise NotImplementedError("Check if this is correct implementation for classification")
            if preds.ndim > 1:
                return np.argmax(preds, axis=1)
            return (preds > 0.5).astype(int)

        else:
            return preds



    def calculate_confidence_and_errors(self, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:

        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], -1))

        upper = self.upper.predict(x)
        lower = self.lower.predict(x)
        yhat = self.model.predict(x)

        difference = upper - lower
        confidence = {
            "confidence_quantile": 1 / (difference + 1e-6)
        }

        return confidence, self.calc_task_metric(yhat, y, reduction="batch")

    def get_constructor(self) -> dict:
        return self.constructor


    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        data = constructor.copy()
        data["folder"] = Path(data["folder"])
        return cls(**data)

    def save(self):

        if self.folder is None:
            return

        constructor = self.get_constructor()

        model_file = self.folder / f"{self.name()}.txt"
        self.model.save_model(model_file)

        if self.estimate_confidence:
            self.upper.save_model(self.folder / f"{self.name()}_upper.txt")
            self.lower.save_model(self.folder / f"{self.name()}_lower.txt")

        constructor_file = self.folder / f"{self.name()}.json"
        with open(constructor_file, "w") as f:
            json.dump(constructor, f)

    @classmethod
    def load(cls, folder: Path) -> Self:
        constructor_file = folder / f"{cls.name()}.json"
        model_file = folder / f"{cls.name()}.txt"
        with open(constructor_file, "r") as f:
            constructor = json.load(f)
        instance = cls.apply_constructor(constructor)

        if instance.estimate_confidence:
            instance.upper = lgb.Booster(model_file=folder / f"{cls.name()}_upper.txt")
            instance.lower = lgb.Booster(model_file=folder / f"{cls.name()}_lower.txt")

        instance.model = lgb.Booster(model_file=model_file)
        return instance

    @staticmethod
    def calculate_tree_params_memory(file_path: Path, int_size: int =8, float_size: int=8):
        int_count = 0
        float_count = 0
        reading = False
        pattern = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\b")
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('Tree='):
                    reading = True
                    continue
                if reading and line.strip().startswith('end of trees'):
                    break
                if not reading:
                    continue
                stripped = line.strip()
                if stripped.startswith('shrinkage=') or stripped.startswith('is_linear='):
                    continue
                if '=' in line:
                    vals = line.split('=', 1)[1]
                    for num in pattern.findall(vals):
                        if '.' in num or 'e' in num.lower():
                            float_count += 1
                        else:
                            int_count += 1
        return int_count * int_size + float_count * float_size

    def get_size_nbytes(self) -> int:

        rand = random.randint(1000, 10_000)
        model_file = Path(__file__).parent / f"delete_me_lgbm_{rand}.txt"
        self.model.save_model(model_file)
        nbytes = self.calculate_tree_params_memory(model_file)
        model_file.unlink()
        return nbytes
