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
import pickle


def lop(cfg, model, dataset, num_tasks, save_folder, save, save_freq, device, mask=None):
    if cfg.general.dataset == 'mnist':
        permutations = [np.random.permutation(28 * 28) for _ in range(num_tasks)]
    
    if save:
        for task_id, perm in enumerate(permutations):
            np.save(save_folder + f'permutation_task{task_id}', arr=perm)

    train = dataset[0]
    test = dataset[1]

    model = model.to(device)

    task_performance = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for task_id, perm in enumerate(permutations):
        print(f"\nTask {task_id + 1}/{len(permutations)}")

        if save and task_id % save_freq == 0 :
            np.save(file=save_folder + f'permutation_task{task_id}', arr=perm)
            torch.save(model.state_dict(), save_folder + f'snapshot_start_task{task_id}')

        if cfg.general.dataset == 'mnist':
            perm_train = datasets.mnist.PermutedMNIST(train, perm)
            perm_test = datasets.mnist.PermutedMNIST(test, perm)
        
        train_loader = torch.utils.data.DataLoader(perm_train, batch_size=cfg.general.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(perm_test, batch_size=cfg.general.batch_size, shuffle=False)

        train_loss, train_acc = training.train_model(cfg, model, train_loader, device, num_epochs=cfg.general.num_epochs, mask=mask)
        test_loss, test_acc = training.evaluate_model(cfg, model, test_loader, device, mask=mask)

        # Log performance
        task_performance["train_loss"].append(train_loss)
        task_performance["train_acc"].append(train_acc)
        task_performance["test_loss"].append(test_loss)
        task_performance["test_acc"].append(test_acc)

        if save and task_id % save_freq  == 0:
            torch.save(model.state_dict(), save_folder + f'snapshot_end_task{task_id}')

        print(f"Task {task_id + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Task {task_id + 1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    utils.pickle_obj(obj=task_performance, path=save_folder + 'task_performance')

def lop_experiment(cfg, dataset, device):
    if cfg.general.dataset == 'mnist':
        model = models.SimpleMLP()
    else:
        raise NotImplementedError
    
    lop(cfg, model, dataset, cfg.lop_experiment.num_tasks, cfg.lop_experiment.save_folder, True, cfg.lop_experiment.save_freq, device)

def winning_tickets_plasticity(cfg, dataset, device):
    if cfg.general.dataset == 'mnist':
        model = models.SimpleMLP()
    else:
        raise NotImplementedError

    task_ids = list(range(0, 50, 50))
    for task_id in task_ids:
        model.load_state_dict(torch.load(cfg.winning_tickets_plasticity.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
        with open(cfg.winning_tickets_plasticity.masks_dir + f'masks_task_{task_id}_target_percentage_{cfg.winning_tickets_plasticity.target_percentage}_pruning_rounds_{cfg.winning_tickets_plasticity.pruning_rounds}', 'rb') as file: 
            masks = pickle.load(file)
        lop(
            cfg, 
            model, 
            dataset, 
            cfg.winning_tickets_plasticity.num_tasks,
            cfg.winning_tickets_plasticity.save_folder,
            False,
            -1,
            device,
            mask=masks[0]
        )

def winning_tickets_masks(cfg, dataset, device):
    print(device)
    models_dir = cfg.winning_tickets_masks.models_dir

    task_ids = utils.get_unique_ids(models_dir)
    stats_dict  = {}

    for task_id in task_ids:
        stats_dict[task_id] = utils.winning_tickets_helper(cfg, dataset, task_id, device)

    utils.pickle_obj(obj=stats_dict, path=f'./results/winning_tickets_masks/pairwise_{cfg.winning_tickets_masks.pairwise}_target_percentage_{cfg.winning_tickets_masks.target_percentage}_pruning_rounds_{cfg.winning_tickets_masks.pruning_rounds}')



def parameter_plasticity(cfg,model,model_name,dataset,device):
    train = dataset[0]

    model = model.to(device)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.general.batch_size, shuffle=True)

    gradient = training.train_model_gradient(cfg, model, train_loader, device, num_epochs=cfg.general.num_epochs)

    gradients_stats = utils.aggregate_gradient_stats([gradient])

    utils.pickle_obj(obj=gradients_stats, path=f'./results/parameter_plasticity/gradients_stats_{model_name}')



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.load_config()
    utils.seed_everything(cfg.general.seed)

    if cfg.general.dataset == 'mnist':
        dataset = datasets.mnist.get_MNIST_dataset()
    else:
        raise NotImplementedError
    

    
    #lop_experiment(cfg=cfg, dataset=dataset, device=device)

    winning_tickets_masks(cfg=cfg, dataset=dataset,device=device)

    #winning_tickets_plasticity(cfg=cfg, dataset=dataset, device=device)
    

if __name__ == '__main__':
    main()