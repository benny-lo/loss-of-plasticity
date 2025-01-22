import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import copy
import numpy as np

import models
import training
import utils

def mask_selection(cfg,masks):
    criterion  = cfg.overlap_parameters_tickets.mask_selection_criterion
    if criterion == 'first':
        return masks[0]
    
    num_parameters = len(masks[0])

    ones_percentage = utils.percentage_of_ones(masks[0])

    aggregations = []

    for i in range(num_parameters):
        param_aggregation = masks[0][i]

        for mask in masks[1:]:
            if criterion == 'intersection':
                param_aggregation = param_aggregation & mask[i]
            if criterion == 'union':
                param_aggregation = param_aggregation | mask[i]
            if criterion == 'average':
                raise NotImplementedError         

        aggregations.append(param_aggregation)

    return aggregations

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

    return [x.to('cpu') for x in mask], accuracies

def winning_tickets_helper(cfg, dataset, task_id, device):
    print(f'task_id: {task_id}')
    
    perm = np.load(cfg.winning_tickets_masks.models_dir + f'permutation_task{task_id}.npy')
    
    perm_train = datasets.mnist.PermutedMNIST(dataset[0], perm)
    perm_test = datasets.mnist.PermutedMNIST(dataset[1], perm)

    train_loader = torch.utils.data.DataLoader(perm_train, batch_size=cfg.general.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(perm_test, batch_size=cfg.general.batch_size, shuffle=False)

    masks = []
    for ticket_id in range(cfg.winning_tickets_masks.num_tickets):
        if cfg.general.dataset == 'mnist':
            model = models.SimpleMLP()
        else:
            raise NotImplementedError
    
        model = model.to(device)
        model.load_state_dict(torch.load(cfg.winning_tickets_masks.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
        
        print(f"finding {ticket_id} winning ticket")
        mask, _ = find_winning_ticket(cfg, model, train_loader, test_loader, device)
        masks.append(mask)
    utils.dump_pickle_obj(masks,f"./results/winning_tickets_masks/masks/masks_task_{task_id}_target_percentage_{cfg.winning_tickets_masks.target_percentage}_pruning_rounds_{cfg.winning_tickets_masks.pruning_rounds}")


    if cfg.winning_tickets_masks.pairwise:
        overlaps = []
        for i in range(cfg.winning_tickets_masks.num_tickets):
            for j in range(i+1, cfg.winning_tickets_masks.num_tickets):
                overlaps.append(utils.compute_pairwise_overlap(masks[i],masks[j]).cpu().item())
        overlaps = np.array(overlaps)
        avg_overlap, std_overlap = np.mean(overlaps), np.std(overlaps)
        return {'average pairwise overlap':avg_overlap, 'pairwise overlap std':std_overlap}

    else:
        total_overlap = utils.compute_overlap(masks)
        return {'total_overlap': total_overlap}