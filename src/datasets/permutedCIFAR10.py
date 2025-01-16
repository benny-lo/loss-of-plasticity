import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SimpleVGG, train_model, evaluate_model, plot_results
import copy


def permute_image(image, perm):

    img_flat = image.view(-1)
    permuted_img = img_flat[perm]
    
    permuted_img = permuted_img.view(3, 32, 32)
    return permuted_img


class PermutedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, permutation):
        self.dataset = dataset
        self.permutation = permutation

    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        
        #permuted_image = permute_image(image, self.permutation)
        permuted_image = image
        return permuted_image, label

    def __len__(self):
        return len(self.dataset)
    



def lop_experiment1(model, num_tasks=500, batch_size= 32):    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    task_performance = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} 

    snapshots = {"early": None, "mid": None, "end": None}


    permutations = [np.random.permutation(3 * 32 * 32) for _ in range(num_tasks)]

    train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

    batch_size = 16


    for task_id, perm in enumerate(permutations):
        print(f"\nTask {task_id + 1}/{len(permutations)}")

        train_dataset = PermutedCIFAR10(train_set,perm)
        test_dataset = PermutedCIFAR10(test_set,perm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, num_epochs=3)

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
            torch.save(model.state_dict(),  "snapshot_early_cifar.pth")
        elif task_id == num_tasks // 2:  # Midway through tasks
            snapshots["mid"] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),  "snapshot_mid_cifar.pth")
        elif task_id == num_tasks - 1:  # End of all tasks
            snapshots["end"] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),  "snapshot_late_cifar.pth")



    plot_results(task_performance)

    return snapshots



def main():

    
    model = SimpleVGG()

    lop_experiment1(model)
    


if __name__ == "__main__":
    main()
    

