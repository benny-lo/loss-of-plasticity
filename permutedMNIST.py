import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SimpleMLP, train_model, evaluate_model


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
    


def plot_results(task_performance):
    

    # Plot training and testing accuracy across tasks
    plt.figure(figsize=(12, 5))
    plt.plot(task_performance["train_acc"], label="Train Accuracy")
    plt.plot(task_performance["test_acc"], label="Test Accuracy")
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Tasks")
    plt.legend()
    plt.show()

    


def main():
    num_tasks = 10
    permutations = [np.random.permutation(28 * 28) for _ in range(num_tasks)]

    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    task_datasets = []

    for task_id, perm in enumerate(permutations):
        train_dataset = PermutedMNIST(mnist_train, perm)
        test_dataset = PermutedMNIST(mnist_test, perm)
        task_datasets.append((train_dataset, test_dataset))

    batch_size = 64

    # DataLoaders for each task
    task_loaders = [
        (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
        for train_dataset, test_dataset in task_datasets
    ]



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    task_performance = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} 

    for task_id, (train_dataset, test_dataset) in enumerate(task_datasets):
        print(f"\nTask {task_id + 1}/{len(task_datasets)}")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)

        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        # Log performance
        task_performance["train_loss"].append(train_loss)
        task_performance["train_acc"].append(train_acc)
        task_performance["test_loss"].append(test_loss)
        task_performance["test_acc"].append(test_acc)

        print(f"Task {task_id + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Task {task_id + 1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    plot_results(task_performance)




if __name__ == "__main__":
    main()

