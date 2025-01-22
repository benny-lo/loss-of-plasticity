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

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dump_pickle_obj(obj, path: str):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle_obj(path: str):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj       

def percentage_of_ones(mask: list):
    total_elements = sum(t.numel() for t in mask)
    total_ones = sum(t.sum() for t in mask)

    return total_ones / total_elements


