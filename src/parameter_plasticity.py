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
import logging
logging.basicConfig(filename='error_log.txt', level=logging.DEBUG)

from experiments import parameter_plasticity
import pickle

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
                param_aggregation = param_aggregation.to(torch.int32) & mask[i].to(torch.int32)
            if criterion == 'union':
                param_aggregation = param_aggregation.to(torch.int32) | mask[i].to(torch.int32)
                
            if criterion == 'average':
                raise NotImplementedError         

        aggregations.append(param_aggregation)

    return aggregations


def overlap_parameters_tickets(cfg):
    
    masks_dir = cfg.overlap_parameters_tickets.masks_dir
    gradients_dir = cfg.overlap_parameters_tickets.gradients_dir

    for idx in range(3):
        stats_dict = {}
        target_percentage = target_percentage_values[idx]
        pruning_rounds = pruning_rounds_values[idx]

        #task_ids = utils.get_unique_ids(masks_dir)
        task_ids = range(0,550,50)
        for task_id in task_ids:
            target_percentage_values = [0.2,0.1,0.05]
            pruning_rounds_values = [3,5,10]

            logging.info(f"running mask {task_id} with params {target_percentage} {pruning_rounds}")

            mask_file = open(masks_dir + f'masks_task_{task_id}_target_percentage_{target_percentage}_pruning_rounds_{pruning_rounds}','rb')
            masks = torch.load(mask_file, weights_only=True)
            mask = mask_selection(cfg,masks)
            grad_file = open(gradients_dir + f'gradients_stats_snapshot_start_task{task_id}','rb')
            grad = pickle.load(grad_file)['avg_grad']

            if len(grad) > 2:
                grad = grad[:2]

            ones_percentage = utils.percentage_of_ones(mask)

            grad_mask = []

            for param_grad in grad:
                grad_mask.append(torch.ones_like(param_grad))

                threshold = torch.quantile(param_grad.abs().view(-1) ,1-ones_percentage)

                grad_mask[-1][param_grad.abs() < threshold] = 0

            ticket_parameters_overlap = utils.compute_pairwise_overlap(mask,grad_mask)
            stats_dict[task_id]= ticket_parameters_overlap
            logging.info(f"overlap between ticket and most important parameters : {ticket_parameters_overlap} for task id {task_id}")
        utils.pickle_obj(stats_dict,cfg.overlap_parameters_tickets.save_dir + f'overlap_parameters_tickets_target_percentage_{target_percentage}_pruning_rounds_{pruning_rounds}' )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.load_config()
    utils.seed_everything(cfg.general.seed)

    if cfg.general.dataset == 'mnist':
        dataset = datasets.mnist.get_MNIST_dataset()
    else:
        raise NotImplementedError
    
    overlap_parameters_tickets(cfg)

    
    ###  code for the parameter_plasticity experiment ###

    """models_dir = cfg.winning_tickets_masks.models_dir

    task_ids = utils.get_unique_ids(models_dir)
    task_ids = [int(x) for x in task_ids]

    for task_id in task_ids:
        logging.info(f"\nTask {task_id + 1}")
        if cfg.general.dataset == 'mnist':
            model = models.SimpleMLP()
        else:
            raise NotImplementedError

        model = model.to(device)
        model.load_state_dict(torch.load(cfg.winning_tickets_masks.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
        model_name = f'snapshot_start_task{task_id}'
        parameter_plasticity(cfg, model=model, model_name=model_name, dataset=dataset, device=device)"""
    

if __name__ == '__main__':
    main()
