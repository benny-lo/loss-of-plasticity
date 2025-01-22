import numpy as np
import matplotlib.pyplot as plt
import yaml
import box
import random
import torch
import pickle
import os
import re

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

def dump_pickle_obj(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle_obj(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj

def smooth_signal(signal, window_size):

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to ensure symmetry.")
    
    half_window = window_size // 2
    padded_signal = np.pad(signal, (half_window, half_window), mode='edge')
    smoothed_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    return smoothed_signal

def get_unique_ids(directory, pattern=r'\d+'):
    ids = set()
    for filename in os.listdir(directory):
        if not filename.startswith('snapshot'): 
            continue
        match = re.findall(pattern, filename)
        if match:
            ids.update(match)
    return sorted(ids)
    
def aggregate_gradient_stats(gradients):
    num_tasks = len(gradients)
    stats_dict = {}

    avg_grad = [torch.zeros_like(param_grad) for param_grad in gradients[0] ]

    for grad in gradients:
        for param_idx,param_grad in enumerate(grad):
            avg_grad[param_idx] += param_grad / num_tasks
        
    stats_dict['avg_grad'] = avg_grad
    return stats_dict
        

def percentage_of_ones(mask: list):
    total_elements = sum(t.numel() for t in mask)
    total_ones = sum(t.sum() for t in mask)

    return total_ones / total_elements


