import json
from pathlib import Path
from typing import Callable, Literal, Self
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

from ..adapter import Adapter

class RandomForest(Adapter):

    def __init__(
            self,
            n_input: int,
            n_output: int,
            folder: Path,
            n_estimators: int = 100,
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: Literal["sqrt", "log2"] | int | float = "sqrt",
            max_leaf_nodes: int | None = None
        ):
        super().__init__(n_input, n_output, folder)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

        self.rfc: RandomForestClassifier = None
        self.criterion: Callable[[np.ndarray, np.ndarray], float] = self.accuracy


    @staticmethod
    def accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
        return (y_hat == y).mean()

    def init_model(
            self,
            **kwargs
        ):

        self.rfc = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
        )


    def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            **kwargs
        ):

        self.rfc.fit(x, y)

        y_train_hat = self.rfc.predict(x)
        train_accuracy = self.criterion(y_train_hat, y)

        y_val_hat = self.rfc.predict(x_val)
        val_accuracy = self.criterion(y_val_hat, y_val)

        folder = self.folder
        save_files = self.folder is not None

        self.save_test_metrics(accuracy=val_accuracy, train_accuracy=train_accuracy)

        return val_accuracy


    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.rfc.predict(x)

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

        joblib.dump(self.rfc, file_tree)
        with open(file_constructor, "w") as f:
            json.dump(self.get_constructor(), f)


    @classmethod
    def load(cls, folder: Path) -> Self:

        file_constructor = folder / "rf.json"
        file_tree = folder / "rf.pickle"

        with open(file_constructor, "r") as f:
            constructor = json.load(f)

        instance = cls.apply_constructor(constructor)
        instance.rfc = joblib.load(file_tree)

        return instance

    @classmethod
    def count_tree_values(cls, tree):
        return (
            tree.feature.nbytes +
            tree.threshold.nbytes +
            tree.children_left.nbytes +
            tree.children_right.nbytes +
            tree.value.nbytes
        )

    @classmethod
    def count_random_forest_values(cls, forest: RandomForestClassifier):
        return sum(cls.count_tree_values(est.tree_) for est in forest.estimators_)

    def get_size_nbytes(self):
        return self.count_random_forest_values(self.rfc)





