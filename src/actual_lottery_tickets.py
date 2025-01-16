import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time
import copy
import numpy as np

import models

def initialize_mask(model):
    mask = []

    for param in model.parameters():
        mask.append(torch.ones_like(param))

    return mask
    

def pruning(model,mask,pruning_per_round):
    idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                weights = param[mask[idx] > 0].abs().view(-1) 
                threshold = torch.quantile(weights, 1 - pruning_per_round)

                if threshold == 0:
                    raise ValueError("Very sparse parameters")
                    

                for idx, param in enumerate(model.parameters()):
                    mask[idx][param.abs() < threshold] = 0
        idx += 1
    return mask



     

def find_winning_ticket(model, target_percentage, pruning_rounds, num_epochs):
    initial_params = copy.deepcopy(model.state_dict)

    pruning_per_round = target_percentage ** (1 / pruning_rounds)

    accuracies = np.zeros(pruning_rounds)

    mask = initialize_mask(model)

    for r in range(pruning_rounds):
        pruning(model,mask,pruning_per_round)
        model.load_state_dict(initial_params)
        for _ in range(num_epochs):
            models.train_model(model,mask)

        accuracies[r] = models.evaluate_model(model,mask)[1]

    
    return mask, accuracies