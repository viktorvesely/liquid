import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path
from torch.optim import AdamW
import torch.nn as nn
import tqdm

from council import Council

def load_data():
    data_dir = Path(__file__).parent / 'mnist_data'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    return train_dataset, test_dataset



def train(n_citizens: int = 4):

    batch_size = 800
    epoch = 20

    train_dataset, test_dataset = load_data()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    council = Council(n_citizens=n_citizens)
    optimizer = AdamW(council.parameters(), lr=1e-4)

    for e in range(epoch):

        council.train()

        train_losses = []
        train_entropies = []
        train_self_delegations = []

        for images, labels in tqdm.tqdm(train_loader, total=len(train_loader), desc="Train"):

            images = images.to("cuda")
            labels = labels.to("cuda")

            classifications, power, D = council(images)
            loss = council.loss(labels, classifications, power)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self_delegation = council.self_delegation(D)
                entropy = council.entropy(power)

            train_losses.append(loss.item())
            train_entropies.append(entropy.item())
            train_self_delegations.append(self_delegation.item())

        validation_losses = []
        validation_accuracy = []
        valid_entropies = []
        valid_self_delegations = []

        council.eval()

        for images, labels in tqdm.tqdm(test_loader, total=len(test_loader), desc="Valid"):

            images = images.to("cuda")
            labels = labels.to("cuda")

            with torch.no_grad():

                classifications, power, D = council(images)
                loss = council.loss(labels, classifications, power)
                self_delegation = council.self_delegation(D)
                entropy = council.entropy(power)

                label_dist = council.vote_distribution(classifications, power)

                labels_hat = torch.argmax(label_dist, dim=1)
                correct = (labels_hat == labels).float()
                accuracy = correct.mean()


            validation_losses.append(loss.item())
            validation_accuracy.append(accuracy.item())
            valid_entropies.append(entropy.item())
            valid_self_delegations.append(self_delegation.item())

        print(f" --------Epoch {e}-----------")
        print(f"tl={np.mean(train_losses):.3f} te={np.mean(train_entropies):.3f} tsd={np.mean(train_self_delegations):.3f}")
        print(f"accuracy={np.mean(validation_accuracy):.3f} vl={np.mean(validation_losses):.3f} ve={np.mean(valid_entropies):.3f} vsd={np.mean(valid_self_delegations):.3f}")




if __name__ == "__main__":
    train()