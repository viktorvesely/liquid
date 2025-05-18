import json
from pathlib import Path
import re
from typing import Self
import numpy as np
import lightgbm as lgb

from ..adapter import Adapter

class LightGBM(Adapter):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        folder: Path,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int | None = None,
        num_leaves: int | None = None,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int | None = None
    ):
        super().__init__(n_input, n_output, folder)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model: lgb.Booster | None = None

    def init_model(self):
        params = {
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'seed': self.random_state,
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": self.n_output
        }

        self._params = {k: v for k, v in params.items() if v is not None}

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs
    ) -> float:
        train_data = lgb.Dataset(x, label=y)
        valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
        self.model = lgb.train(
            self._params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data],
        )

        y_hat_train = self.inference(x)
        y_hat_val = self.inference(x_val)

        train_accuracy = (y_hat_train == y).mean()
        val_accuracy = (y_hat_val == y_val).mean()

        self.save_test_metrics(accuracy=val_accuracy, train_accuracy=train_accuracy)

        return val_accuracy

    def inference(self, x: np.ndarray) -> np.ndarray:
        preds = self.model.predict(x)
        if preds.ndim > 1:
            return np.argmax(preds, axis=1)
        return (preds > 0.5).astype(int)

    def get_constructor(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "folder": str(self.folder.resolve()),
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state
        }

    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        data = constructor.copy()
        data["folder"] = Path(data["folder"])
        return cls(**data)

    def save(self):
        if self.folder is None:
            return

        model_file = self.folder / f"{self.name()}.txt"
        self.model.save_model(model_file)

        constructor_file = self.folder / f"{self.name()}.json"
        with open(constructor_file, "w") as f:
            json.dump(self.get_constructor(), f)

    @classmethod
    def load(cls, folder: Path) -> Self:
        constructor_file = folder / f"{cls.name()}.json"
        model_file = folder / f"{cls.name()}.txt"
        with open(constructor_file, "r") as f:
            constructor = json.load(f)
        instance = cls.apply_constructor(constructor)
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
        model_file = self.folder / "delete_me_lgbm.txt"
        self.model.save_model(model_file)
        nbytes = self.calculate_tree_params_memory(model_file)
        model_file.unlink()
        return nbytes
