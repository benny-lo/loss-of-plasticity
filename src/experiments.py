import utils
import datasets.mnist
import numpy as np
import torch
import models
import training
import os
from overlap import compute_pairwise_overlap, compute_overlap
from actual_lottery_tickets import find_winning_ticket

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
            torch.save(model.state_dict, f'./results/lop/snapshot_start_task{task_id}')

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
            torch.save(model.state_dict, f'./results/lop/snapshot_end_task{task_id}')

        print(f"Task {task_id + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Task {task_id + 1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    utils.pickle_obj(obj=task_performance, path='./results/lop/task_performance')


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

def winning_tickets_masks(cfg):
    models_dir = cfg.winning_tickets_masks.models_dir
    num_tickets = cfg.winning_tickets_masks.num_tickets
    pairwise = cfg.winning_tickets_masks.pairwise
    target_percentage = cfg.winning_tickets_masks.target_percentage
    pruning_rounds = cfg.winning_tickets_masks.pruning_rounds
    num_epochs = cfg.general.num_epochs

    checkpoints = [f for f in os.listdir(models_dir) if f.endswith(".pth")]

    stats_dict  = {}

    for file_path in checkpoints:
        file_name = os.path.basename(file_path, num_tickets, pairwise)
        stats_dict[file_name] = winning_tickets_helper(file_name,num_tickets,pairwise,target_percentage,pruning_rounds,num_epochs)

    print(stats_dict)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.load_config()
    utils.seed_everything(cfg.general.seed)

    print(cfg.lop.save_freq)

    if cfg.general.dataset == 'mnist':
        dataset = datasets.mnist.get_MNIST_dataset()
    else:
        raise NotImplementedError

    lop(cfg=cfg, dataset=dataset, device=device)
    

if __name__ == '__main__':
    main()