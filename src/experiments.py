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


def lop(cfg, dataset, device):
    if cfg.general.dataset == 'mnist':
        model = models.SimpleMLP()
        permutations = [np.random.permutation(28 * 28) for _ in range(cfg.lop.num_tasks)]

    train = dataset[0]
    test = dataset[1]

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.general.lr)
    criterion = torch.nn.CrossEntropyLoss()

    task_performance = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    save_folder = './results/lop/'

    for task_id, perm in enumerate(permutations):
        print(f"\nTask {task_id + 1}/{len(permutations)}")

        if task_id % cfg.lop.save_freq == 0:
            np.save(file=save_folder + f'permutation_task{task_id}', arr=perm)
            torch.save(model.state_dict(), f'./results/lop/snapshot_start_task{task_id}')

        if cfg.general.dataset == 'mnist':
            perm_train = datasets.mnist.PermutedMNIST(train, perm)
            perm_test = datasets.mnist.PermutedMNIST(test, perm)
        
        train_loader = torch.utils.data.DataLoader(perm_train, batch_size=cfg.general.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(perm_test, batch_size=cfg.general.batch_size, shuffle=False)

        train_loss, train_acc = training.train_model(model, train_loader, optimizer, criterion, device, num_epochs=cfg.general.num_epochs)
        test_loss, test_acc = training.evaluate_model(model, test_loader, criterion, device)

        # Log performance
        task_performance["train_loss"].append(train_loss)
        task_performance["train_acc"].append(train_acc)
        task_performance["test_loss"].append(test_loss)
        task_performance["test_acc"].append(test_acc)

        if task_id % cfg.lop.save_freq == 0:
            torch.save(model.state_dict(), f'./results/lop/snapshot_end_task{task_id}')

        print(f"Task {task_id + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Task {task_id + 1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    utils.pickle_obj(obj=task_performance, path='./results/lop/task_performance')


def winning_tickets_masks(cfg, dataset, device):
    models_dir = cfg.winning_tickets_masks.models_dir

    task_ids = utils.get_unique_ids(models_dir)
    stats_dict  = {}

    for task_id in task_ids:
        stats_dict[task_id] = utils.winning_tickets_helper(cfg, dataset, task_id, device)

    utils.pickle_obj(obj=stats_dict, path='./results/winning_tickets_masks/stats_dict')



def parameter_plasticity(cfg,model,dataset,device):
    train = dataset[0]

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.general.lr)
    criterion = torch.nn.CrossEntropyLoss()

    permutations = [np.random.permutation(28 * 28) for _ in range(cfg.lop.num_tasks)]

    gradients = []
    for task_id, perm in enumerate(permutations):
        print(f"\nTask {task_id + 1}/{len(permutations)}")


        if cfg.general.dataset == 'mnist':
            perm_train = datasets.mnist.PermutedMNIST(train, perm)
        
        train_loader = torch.utils.data.DataLoader(perm_train, batch_size=cfg.general.batch_size, shuffle=True)

        gradient = training.train_model_gradient(model, train_loader, optimizer, criterion, device, num_epochs=cfg.general.num_epochs)
        gradients.append(gradient)

    gradients_stats = utils.aggregate_gradient_stats(gradients)

    utils.pickle_obj(obj=gradients_stats, path='./results/parameter_plasticity/gradients_stats')



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.load_config()
    utils.seed_everything(cfg.general.seed)

    if cfg.general.dataset == 'mnist':
        dataset = datasets.mnist.get_MNIST_dataset()
    else:
        raise NotImplementedError

    #lop(cfg=cfg, dataset=dataset, device=device)

    winning_tickets_masks(cfg=cfg, dataset=dataset,device=device)
    

if __name__ == '__main__':
    main()