import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time

from models import train_model, evaluate_model, SimpleMLP
from lottery_tickets import MNIST, initialize_mnist


def compute_entk(model,x1,y1, x2,y2):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()



    outputs = model(x1.to(torch.float))
    loss = criterion(outputs, y1)
    loss.backward()

    grad1 = torch.cat([param.grad.view(-1) for param in model.parameters()])


    optimizer.zero_grad()
    outputs = model(x2.to(torch.float))
    loss = criterion(outputs, y2)
    loss.backward()

    grad2 = torch.cat([param.grad.view(-1) for param in model.parameters()])


    return torch.dot(grad1, grad2)


def lop_entk(state_dict):

    model = SimpleMLP()
    model.load_state_dict(state_dict)

    train_loader, test_loader, _ = initialize_mnist()

    n = 10

    entk = 0
    data = []

    i = 0
    for x,y in test_loader:
        data.append((x,y))
        i+= 1
        if i >= n:
            break


    for i in range(n):
        for j in range(i+1,n):
            entk += compute_entk(model, data[i][0],data[i][1],data[j][0],data[j][1])

    entk /= n*(n-1)/2


    return entk


def main():

    
    # early, mid, late = lop_experiment1(model)

    # alternatively, load the saved checkpoints

    early = torch.load("snapshot_early.pth")
    mid = torch.load("snapshot_mid.pth")
    late = torch.load("snapshot_late.pth")


    early_entk  = lop_entk(early)
    

    mid_entk  = lop_entk(mid)
    late_entk  = lop_entk(late)

    print(early_entk, mid_entk, late_entk)




if __name__ == "__main__":
    main()







    
