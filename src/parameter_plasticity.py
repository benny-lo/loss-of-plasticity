import utils
import datasets.mnist
import numpy as np
import torch
import models
import training
import os
from overlap import compute_pairwise_overlap, compute_overlap
from actual_lottery_tickets import find_winning_ticket
import box

from experiments import parameter_plasticity


def overlap_parameters_tickets(cfg):
    masks_dir = cfg.masks_dir
    gradients_dir = cfg.gradients_dir

    
    # i assume here i take a single mask and grad

    mask = ...  
    grad = ...
    ones_percentage = utils.percentage_of_ones(mask)

    torch_grad = torch.cat(grad)
    threshold = torch.quantile(torch_grad,1-ones_percentage)

    grad_mask = []

    for param_grad in grad:
        grad_mask.append(torch.ones_like(param_grad))

        grad_mask[-1][param_grad.abs() < threshold] = 0

    ticket_parameters_overlap = utils.compute_pairwise_overlap(mask,grad_mask)
    print(f"overlap between ticket and most important parameters : {ticket_parameters_overlap}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.load_config()
    utils.seed_everything(cfg.general.seed)

    if cfg.general.dataset == 'mnist':
        dataset = datasets.mnist.get_MNIST_dataset()
    else:
        raise NotImplementedError
    
    ###  code for the parameter_plasticity experiment ###

    models_dir = cfg.winning_tickets_masks.models_dir

    task_ids = utils.get_unique_ids(models_dir)
    task_ids = [int(x) for x in task_ids]

    for task_id in task_ids:
        print(f"\nTask {task_id + 1}")
        if cfg.general.dataset == 'mnist':
            model = models.SimpleMLP()
        else:
            raise NotImplementedError

        model = model.to(device)
        model.load_state_dict(torch.load(cfg.winning_tickets_masks.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
        model_name = f'snapshot_start_task{task_id}'
        parameter_plasticity(cfg, model=model, model_name=model_name, dataset=dataset, device=device)
    

if __name__ == '__main__':
    main()
