import numpy as np
import matplotlib.pyplot as plt
import yaml
import box
import random
import torch
import pickle
import models
import os
import re
from actual_lottery_tickets import find_winning_ticket
from overlap import compute_pairwise_overlap, compute_overlap
import datasets

def load_config():
    with open('./src/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return box.Box(config)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pickle_obj(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

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

def get_unique_ids(directory, pattern=r'\d+'):
    ids = set()
    for filename in os.listdir(directory):
        if not filename.startswith('snapshot'): 
            continue
        match = re.findall(pattern, filename)
        if match:
            ids.update(match)
    return sorted(ids)


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
    pickle_obj(masks,f"./results/winning_tickets_masks/masks/masks_task_{task_id}_target_percentage_{cfg.winning_tickets_masks.target_percentage}_pruning_rounds_{cfg.winning_tickets_masks.pruning_rounds}")


    if cfg.winning_tickets_masks.pairwise:
        overlaps = []
        for i in range(cfg.winning_tickets_masks.num_tickets):
            for j in range(i+1, cfg.winning_tickets_masks.num_tickets):
                overlaps.append(compute_pairwise_overlap(masks[i],masks[j]).cpu().item())
        overlaps = np.array(overlaps)
        avg_overlap, std_overlap = np.mean(overlaps), np.std(overlaps)
        return {'average pairwise overlap':avg_overlap, 'pairwise overlap std':std_overlap}

    else:
        total_overlap = compute_overlap(masks)
        return {'total_overlap': total_overlap}
    
def aggregate_gradient_stats(gradients):
    num_tasks = len(gradients)
    stats_dict = {}

    avg_grad = [torch.zeros_like(param_grad) for param_grad in gradients[0] ]

    for grad in gradients:
        for param_idx,param_grad in enumerate(grad):
            avg_grad[param_idx] += param_grad / num_tasks
        
    stats_dict['avg_grad'] = avg_grad
    return stats_dict
        




