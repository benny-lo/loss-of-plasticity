import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import models
from training import train_model, evaluate_model
from actual_lottery_tickets import find_winning_ticket


     ###   Goal is to find the relationship between overlap of winning tickets and plasticity ###


def compute_pairwise_overlap(mask1, mask2):

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


def compute_overlap(masks):

    assert all(len(mask) == len(masks[0]) for mask in masks), "All masks must have the same length."

    intersection = 0
    union = 0

    for idx in range(len(masks[0])):
        bin_masks = [ (mask[idx] != 0).float() for mask in masks ]

        intersection += torch.sum(torch.prod(torch.stack(bin_masks), dim=0))

        union += torch.sum(torch.any(torch.stack(bin_masks), dim=0).float())

    overlap_percentage = (intersection / union) * 100 if union > 0 else 0.0

    return overlap_percentage


def main():

   pass




if __name__ == "__main__":
    main()



