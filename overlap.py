import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SimpleMLP, train_model, evaluate_model
from permutedMNIST import lop_experiment1
from lottery_tickets import find_winning_ticket


     ###   Goal is to find the relationship between overlap of winning tickets and plasticity ###


def compute_overlap(mask1, mask2):

    assert(len(mask1)== len(mask2))

    intersection = 0
    union = 0

    for idx in range(len(mask1)):
        m1 = mask1[idx]
        m2 = mask2[idx]

        # Ensure masks are binary
        m1 = (m1 != 0).float()
        m2 = (m2 != 0).float()
        
        # Intersection: non-zero weights in both masks
        intersection += torch.sum(m1 * m2)
        
        # Union: non-zero weights in either mask
        union += torch.sum((m1 + m2) > 0).float()
        
    # Overlap as percentage
    overlap_percentage = (intersection / union) * 100 if union > 0 else 0.0
    
    return overlap_percentage



def overlap_lop(state_dict):

    num_tickets = 2

    tickets = []

    for i in range(num_tickets):
        model = SimpleMLP()
        model.load_state_dict(state_dict)
        tickets.append(find_winning_ticket(model))
        
    n = len(tickets)
    overlap = 0

    for i in range(n):
        for j in range(i+1,n):
            overlap += compute_overlap(tickets[i],tickets[j])

    overlap /= n*(n-1)/2


    return overlap






def main():

    
    # early, mid, late = lop_experiment1(model)

    # alternatively, load the saved checkpoints

    early = torch.load("snapshot_early.pth")
    mid = torch.load("snapshot_mid.pth")
    late = torch.load("snapshot_late.pth")


    early_overlap  = overlap_lop(early)
    

    mid_overlap  = overlap_lop(mid)
    late_overlap  = overlap_lop(late)

    print(early_overlap, mid_overlap, late_overlap)




if __name__ == "__main__":
    main()



