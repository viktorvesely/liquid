from pathlib import Path
from datetime import datetime

def create_experiment_folder(task: str, name: str = "experiment", hyper: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_folder = Path(__file__).parent.parent / "experiments"
    if hyper:
        folder = exp_folder / task / "hyper" / f"{timestamp}_{name}"
    else:
        folder = exp_folder / task / f"{timestamp}_{name}"

    folder.mkdir(parents=True, exist_ok=False)
    return folder

