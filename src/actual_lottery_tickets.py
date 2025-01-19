import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time
import copy
import numpy as np

import models
import training

def initialize_mask(model):
    mask = []

    for name,param in model.named_parameters():
        if 'weight' in name:
            mask.append(torch.ones_like(param))

    return mask
    

def pruning(model,mask,pruning_per_round):
    idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            with torch.no_grad():
                assert(param.shape == mask[idx].shape)
                weights = param[mask[idx] > 0].abs().view(-1) 
                threshold = torch.quantile(weights, 1 - pruning_per_round)

                if threshold == 0:
                    raise ValueError("Very sparse parameters")
                    
                mask[idx][param.abs() < threshold] = 0
            idx += 1
    return mask



     

def find_winning_ticket(cfg, model, data_loader, test_loader, device):
    initial_params = copy.deepcopy(model.state_dict())

    pruning_per_round = cfg.winning_tickets_masks.target_percentage ** (1 / cfg.winning_tickets_masks.pruning_rounds)

    accuracies = np.zeros(cfg.winning_tickets_masks.pruning_rounds + 1)

    mask = initialize_mask(model)

    for r in range(cfg.winning_tickets_masks.pruning_rounds):
        model.load_state_dict(initial_params)
        
        training.train_model(cfg, model, data_loader, device, cfg.general.num_epochs,mask=mask)        
        accuracies[r] = training.evaluate_model(cfg, model, test_loader, device, mask=mask)[1]

        pruning(model, mask, pruning_per_round)

    accuracies[cfg.winning_tickets_masks.pruning_rounds] = training.evaluate_model(cfg, model, test_loader, device, mask=mask)[1]
    
    return mask, accuracies