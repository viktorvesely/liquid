import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.optim import AdamW
import torch.nn as nn
import tqdm

from citizen import Citizen

def load_data():
    data_dir = Path(__file__).parent / 'mnist_data'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset



def train():

    batch_size = 1024
    epoch = 20

    train_dataset, test_dataset = load_data()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = Citizen()
    model = model.to("cuda")

    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for e in range(epoch):

        model.train()
        tl = []
        for images, labels in tqdm.tqdm(train_loader, total=len(train_loader), desc="Train"):

            images = images.to("cuda")
            labels = labels.to("cuda")

            label_dist = model(images)
            loss = criterion(label_dist, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl.append(loss.item())

        vl = []
        va = []
        model.eval()
        for images, labels in tqdm.tqdm(test_loader, total=len(test_loader), desc="Valid"):

            images = images.to("cuda")
            labels = labels.to("cuda")

            with torch.no_grad():
                label_dist = model(images)
                loss = criterion(label_dist, labels)

                labels_hat = torch.argmax(label_dist, dim=1)
                correct = (labels_hat == labels).float()
                accuracy = correct.mean()


            vl.append(loss.item())
            va.append(accuracy.item())

        print(f"Epoch {e} tl={np.mean(tl):.3f} vl={np.mean(vl):.3f} accuracy={np.mean(va):.3f}")





if __name__ == "__main__":
    train()