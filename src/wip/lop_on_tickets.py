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


from lottery_tickets import find_winning_ticket
from permutedMNIST import lop_experiment1


        ###   Goal is to compare the progression of normal model vs winning ticket  ###


def ticket_from_mask(model, mask):
    pruned_model = copy.deepcopy(model)  # Create a deep copy of the model
    with torch.no_grad():
        for idx, param in enumerate(pruned_model.parameters()):
            param *= mask[idx]  # Apply the mask to the copied model
    return pruned_model


def main():

    model = SimpleMLP()

    mask = find_winning_ticket(model)

    ticket = ticket_from_mask(model,mask)


    lop_experiment1(model)
    lop_experiment1(ticket)






if __name__ == "__main__":
    main()