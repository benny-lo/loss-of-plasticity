import numpy as np
import matplotlib.pyplot as plt
import yaml
import box
import random
import torch
import pickle

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