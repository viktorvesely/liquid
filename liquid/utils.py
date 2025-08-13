from pathlib import Path
from datetime import datetime
from string import ascii_letters
import numpy as np

def create_experiment_folder(task: str, name: str = "experiment", hyper: bool = False, rand: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rand_string = np.random.choice([char for char in ascii_letters], size=6, replace=True)
    rand_string = ''.join(rand_string)

    exp_folder = Path(__file__).parent.parent / "experiments"
    if hyper:
        folder = exp_folder / task / "hyper" / f"{timestamp}_{name}"
    else:
        folder = exp_folder / task / f"{timestamp}_{name}"

    if rand:
        folder = folder.parent / f"{folder.name}_{rand_string}"

    folder.mkdir(parents=True, exist_ok=False)
    return folder

