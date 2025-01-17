import numpy as np
import matplotlib.pyplot as plt
import yaml
import box
import random
import torch
import pickle
from actual_lottery_tickets import find_winning_ticket
from overlap import compute_pairwise_overlap, compute_overlap

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

def winning_tickets_helper(file_name, num_tickets, pairwise,target_percentage,pruning_rounds,num_epochs):
    model = torch.load(file_name)
    masks = []
    for ticket_id in range(num_tickets):
        mask, _ = find_winning_ticket(model,target_percentage,pruning_rounds,num_epochs)
        masks.append(mask)

    if pairwise:
        overlaps = []
        for i in range(num_tickets):
            for j in range(i+1,num_tickets):
                overlaps.append(compute_pairwise_overlap(masks[i],masks[j]))
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
        




