import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

#from models import SimpleMLP, train_model, evaluate_model, plot_results, smooth_signal
import copy


def permute_mnist(data, permutation):
    """
    Applies a permutation to the flattened MNIST images.

    Args:
        data: Torch tensor of shape (N, 28, 28) or (N, 784).
        permutation: List or numpy array specifying the pixel shuffling.

    Returns:
        Permuted data of the same shape.
    """
    flattened_data = data.view(data.size(0), -1)  # Flatten the images
    permuted_data = flattened_data[:, permutation]  # Apply permutation
    return permuted_data.view(data.size(0), 28, 28)  # Reshape back to (N, 28, 28)


class PermutedMNIST(Dataset):
    def __init__(self, mnist_data, permutation):
        """
        Custom dataset for Permuted MNIST.

        Args:
            mnist_data: Torch dataset (e.g., training or testing MNIST data).
            permutation: List or numpy array specifying the pixel shuffling.
        """
        self.data = permute_mnist(mnist_data.data, permutation)
        self.targets = mnist_data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_MNIST_dataset():
    mnist_mean = 0.1307
    mnist_std = 0.3081
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean, ), (mnist_std, )),
    ])

    mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=mnist_transform)
    return mnist_train, mnist_test


def main():
    #model = SimpleMLP()
    #lop_experiment_experiment1(model)
    train_data, test_data = get_MNIST_dataset()

if __name__ == "__main__":
    main()

