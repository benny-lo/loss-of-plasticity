import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time

from models import train_model, evaluate_model, SimpleMLP

class MNIST(Dataset):
    def __init__(self, mnist_data):
        
        self.data = mnist_data.data
        self.targets = mnist_data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def prune_model(model, mask, s):
    """Prune s% of the parameters."""
    with torch.no_grad():
        weights = torch.cat([param[mask[idx] > 0].abs().view(-1) for idx, param in enumerate(model.parameters())])
        threshold = torch.quantile(weights, s / 100)

        if threshold == 0:
            print("warning very sparse parameters")
            threshold = 1e-6

        for idx, param in enumerate(model.parameters()):
            mask[idx][param.abs() < threshold] = 0
            param[mask[idx] == 0] = 0
    return mask

def reset_weights(model, initial_state, mask):
    """Reset only the unpruned weights to their initial state."""
    
    for idx, param in enumerate(model.parameters()):
        with torch.no_grad():
            param[mask[idx] > 0] = initial_state[idx][mask[idx] > 0]


def initialize_mnist():
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())



    train_dataset = MNIST(mnist_train)
    test_dataset = MNIST(mnist_test)
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    criterion = nn.CrossEntropyLoss()

    return train_loader, test_loader, criterion




def find_winning_ticket(model,train_loader=None, test_loader=None, criterion=None, performance_threshold=None, s=20, num_iter=5):

    if train_loader is None:
        train_loader, test_loader, criterion = initialize_mnist()
        


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3

    initial_state = []

    for param in model.parameters():
        initial_state.append(param)

    mask = [torch.ones_like(param) for param in model.parameters()]

    print("\nFinding the Winning Ticket")
    current_performance = evaluate_model(model, test_loader=test_loader, criterion=criterion,device="cpu")[1]
    print(f"Initial performance: {current_performance}")


    if performance_threshold == None:
        performance_threshold = current_performance

    for iter in range(num_iter):
        # Prune the model
        mask = prune_model(model, mask, s)
        
        # Reset weights to initial state
        reset_weights(model, initial_state,mask)

        # Evaluate performance
        current_performance = evaluate_model(model, test_loader=test_loader, criterion=criterion, device="cpu")[1]
        print(f"Performance after pruning: {current_performance}")

        for _ in range(num_epochs):
            train_model(model, train_loader=train_loader, optimizer=optimizer, criterion=criterion,device="cpu",mask= mask)

        curr_performance = evaluate_model(model, test_loader=test_loader, criterion=criterion, device="cpu")[1]
        print(f"Performance before pruning: {curr_performance}")


        pruning_percentage(model)


    if curr_performance > 0.9:
        print("Winning ticket found!")

    return mask



def pruning_percentage(model):
    total_weights = 0
    non_zero_weights = 0

    for param in model.parameters():
        total_weights += param.numel()
        non_zero_weights += torch.count_nonzero(param)

    # Calculate and print percentage
    percentage_non_zero = (non_zero_weights / total_weights) * 100
    print(f"Percentage of non-zero weights in the model: {percentage_non_zero:.2f}%")




def main():

    model = SimpleMLP()
    


    train_loader, test_loader, criterion = initialize_mnist()


    performance_threshold = 0.95
    s = 20

    train_model(model,train_loader,optimizer=optim.Adam(model.parameters(), lr=0.001),criterion=criterion,device="cpu")

    find_winning_ticket(model, train_loader, test_loader, criterion, performance_threshold, s)





if __name__ == "__main__":
    main()

