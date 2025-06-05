import json
from pathlib import Path
from typing import Literal, Self
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import scipy.stats as stats

from ..adapter import Adapter

class RandomForest(Adapter):

    def __init__(
            self,
            n_input: int,
            n_output: int,
            folder: Path,
            task: str,
            n_estimators: int,
            max_depth: int,
            min_samples_split: int,
            min_samples_leaf: int,
            max_features: Literal["sqrt", "log2"] | int | float,
            max_leaf_nodes: int
        ):
        super().__init__(n_input, n_output, folder, task=task)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

        self.rf: RandomForestClassifier | RandomForestRegressor = None


    def init_model(
            self,
            **kwargs
        ):

        task_type = self.get_task_type()
        if task_type == "classification":
            RFClass = RandomForestClassifier
        elif task_type == "regression":
            RFClass = RandomForestRegressor

        self.rf = RFClass(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes
        )

    def calculate_confidence_and_errors(self, x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:

        if self.get_task_type() == "regression":

            all_tree_preds = np.array([tree.predict(x) for tree in self.rf.estimators_])
            all_tree_preds = all_tree_preds.T # (batch, n_estimators)

            y_std = all_tree_preds.std(axis=1) # shape: (batch,)

            confidence =  {"confidence_std": 1 / (y_std + 1e-6)}
        else:
            all_tree_preds = np.array([tree.predict(x) for tree in self.rf.estimators_])
            all_tree_preds = all_tree_preds.T # (batch, n_estimators)
            n_estimators = all_tree_preds.shape[1]

            result = stats.mode(all_tree_preds, axis=1, keepdims=True)
            mode = result.mode

            n_agree = (all_tree_preds == mode).sum(1)
            confidence = n_agree / n_estimators

            confidence = {"confidence_mode": confidence}

        yhat = self.rf.predict(x)
        return confidence, self.calc_task_metric(yhat, y, reduction="batch")



    def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            verbose: int,
            **kwargs
        ):

        self.rf.verbose = verbose

        y = np.squeeze(y)
        y_val = np.squeeze(y_val)

        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], -1))
            x_val = np.reshape(x_val, (x_val.shape[0], -1))

        self._train_start = self.now()
        self.rf.fit(x, y)
        self._train_end = self.now()

        y_train_hat = self.rf.predict(x)
        train_metric = self.calc_task_metric(y_train_hat, y)

        y_val_hat = self.rf.predict(x_val)
        val_metric = self.calc_task_metric(y_val_hat, y_val)

        metric_name = self.get_task_metric_name()
        metrics = {
            f"train_{metric_name}": train_metric,
            metric_name: val_metric
        }

        self.set_test_metrics(**metrics)

        return val_metric


    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.rf.predict(x)

    def get_constructor(self) -> dict:

        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "folder": str(self.folder.resolve()),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
        }


    @classmethod
    def apply_constructor(cls, constructor: dict) -> Self:
        constructor["folder"] = Path(constructor["folder"])
        return cls(**constructor)

    def save(self):

        folder = self.folder

        if folder is None:
            return

        file_constructor = folder / "rf.json"
        file_tree = folder / "rf.pickle"

        joblib.dump(self.rf, file_tree)
        with open(file_constructor, "w") as f:
            json.dump(self.get_constructor(), f)


    @classmethod
    def load(cls, folder: Path) -> Self:

        file_constructor = folder / "rf.json"
        file_tree = folder / "rf.pickle"

        with open(file_constructor, "r") as f:
            constructor = json.load(f)

        instance = cls.apply_constructor(constructor)
        instance.rf = joblib.load(file_tree)

        return instance

    @classmethod
    def count_tree_values(cls, tree):
        return (
            tree.feature.nbytes +
            tree.threshold.nbytes +
            tree.children_left.nbytes +
            tree.children_right.nbytes +
            tree.value.nbytes +
            tree.impurity.nbytes +
            tree.n_node_samples.nbytes +
            tree.weighted_n_node_samples.nbytes
        )

    @classmethod
    def count_random_forest_values(cls, forest: RandomForestClassifier):
        return sum(cls.count_tree_values(est.tree_) for est in forest.estimators_)

    def get_size_nbytes(self):
        return self.count_random_forest_values(self.rf)





