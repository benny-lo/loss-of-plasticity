import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SimpleMLP, train_model, evaluate_model
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
    

def smooth_signal(signal, window_size):

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to ensure symmetry.")
    
    half_window = window_size // 2
    padded_signal = np.pad(signal, (half_window, half_window), mode='edge')
    smoothed_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    return smoothed_signal





def plot_results(task_performance,smoothing=True,window=11):
    

    # Plot training and testing accuracy across tasks
    plt.figure(figsize=(12, 5))

    train_acc = task_performance["train_acc"]
    test_acc = task_performance["test_acc"]

    if smoothing:
        train_acc = smooth_signal(train_acc,window)
        test_acc = smooth_signal(test_acc,window)


    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Tasks")
    plt.legend()
    plt.show()



def lop_experiment1(model, num_tasks=500, batch_size= 32):    
    permutations = [np.random.permutation(28 * 28) for _ in range(num_tasks)]

    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    task_performance = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} 

    snapshots = {"early": None, "mid": None, "end": None}


    for task_id, perm in enumerate(permutations):
        print(f"\nTask {task_id + 1}/{len(permutations)}")
        train_dataset = PermutedMNIST(mnist_train, perm)
        test_dataset = PermutedMNIST(mnist_test, perm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)

        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        # Log performance
        task_performance["train_loss"].append(train_loss)
        task_performance["train_acc"].append(train_acc)
        task_performance["test_loss"].append(test_loss)
        task_performance["test_acc"].append(test_acc)

        print(f"Task {task_id + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Task {task_id + 1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")



        if task_id == 4:  # After 5 iterations
            snapshots["early"] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),  "snapshot_early.pth")
        elif task_id == num_tasks // 2:  # Midway through tasks
            snapshots["mid"] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),  "snapshot_mid.pth")
        elif task_id == num_tasks - 1:  # End of all tasks
            snapshots["end"] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),  "snapshot_late.pth")



    plot_results(task_performance)

    return snapshots





def main():

    
    model = SimpleMLP()

    lop_experiment1(model)
    




if __name__ == "__main__":
    main()

