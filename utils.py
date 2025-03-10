from pathlib import Path
from datetime import datetime
import shutil
import torch
from torch import nn, optim

def create_experiment_folder(name: str = "experiment") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent  / "experiments" / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def copy_files_to_folder(folder: Path, *files: str) -> None:
    src_dir = Path(__file__).parent
    for file in files:
        shutil.copy(src_dir / file, folder)

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, file: Path) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, file)
