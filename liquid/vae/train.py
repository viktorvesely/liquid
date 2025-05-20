import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from .architecture import ConvVAE

def create_experiment_folder(name: str = "vae") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent / "runs" / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def now() -> str:
    return str(datetime.now())

BS = 1024

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    folder = create_experiment_folder()
    loss_path = folder / "loss.txt"

    full_train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_train_data))
    val_size = len(full_train_data) - train_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BS, shuffle=False, num_workers=4)

    model = ConvVAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2 * 1e-4)

    best_val_loss = float('inf')

    with open(loss_path, "w") as f:
        f.write(f"start:{now()}\n")

    model.train()
    for epoch in range(1, 1_000):
        model.train()
        train_loss = 0
        for data, _ in tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        tag = "loss"

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), folder / 'vae.pth')
            print(f'New best model saved with val loss {best_val_loss:.4f}')
            tag = "best"

        with open(loss_path, "a") as f:
            f.write(f"{tag}:{now()}:{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

if __name__ == '__main__':
    train()
