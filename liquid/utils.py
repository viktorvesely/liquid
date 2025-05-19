from pathlib import Path
from datetime import datetime

def create_experiment_folder(task: str, name: str = "experiment") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent.parent / "experiments" / task / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

