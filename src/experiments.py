import numpy as np
import torch
import argparse

import models
import lottery_tickets
import training
import utils
import datasets.mnist

def lop(cfg, model, dataset, num_tasks, save_folder, save, save_freq, device, mask=None):
    if cfg.general.dataset == 'mnist':
        permutations = [np.random.permutation(28 * 28) for _ in range(num_tasks)]
    else:
        raise NotImplementedError

    if save:
        for task_id, perm in enumerate(permutations):
            np.save(save_folder + f'permutation_task{task_id}', arr=perm)

    train = dataset[0]
    test = dataset[1]

    task_performance = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for task_id, perm in enumerate(permutations):
        print(f"\nTask {task_id + 1}/{len(permutations)}")

        if save and task_id % save_freq == 0:
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

    utils.dump_pickle_obj(obj=task_performance, path=save_folder + 'task_performance')

def lop_experiment(cfg, dataset, device):
    utils.create_folder(cfg.lop_experiment.save_folder)

    if cfg.general.dataset == 'mnist':
        model = models.SimpleMLP()
    else:
        raise NotImplementedError
    
    model = model.to(device)
    lop(cfg, model, dataset, cfg.lop_experiment.num_tasks, cfg.lop_experiment.save_folder, True, cfg.lop_experiment.save_freq, device)

def winning_tickets_plasticity(cfg, dataset, device):
    utils.create_folder(cfg.winning_tickets_plasticity.save_folder)

    if cfg.general.dataset == 'mnist':
        model = models.SimpleMLP()
    else:
        raise NotImplementedError

    model = model.to(device)

    task_ids = cfg.winning_tickets_plasticity.tasks
    for task_id in task_ids:
        model.load_state_dict(torch.load(cfg.winning_tickets_plasticity.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
        masks = utils.load_pickle_obj(cfg.winning_tickets_plasticity.masks_dir + f'masks_task_{task_id}_target_percentage_{cfg.winning_tickets_plasticity.target_percentage}_pruning_rounds_{cfg.winning_tickets_plasticity.pruning_rounds}')

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
    task_ids = cfg.winning_tickets_masks.tasks
    stats_dict  = {}

    for task_id in task_ids:
        stats_dict[task_id] = lottery_tickets.winning_tickets_helper(cfg, dataset, task_id, device)

    utils.create_folder(cfg.winning_tickets_masks.save_folder)
    utils.dump_pickle_obj(obj=stats_dict, path=cfg.winning_tickets_masks.save_folder + f'./results/winning_tickets_masks/pairwise_{cfg.winning_tickets_masks.pairwise}_target_percentage_{cfg.winning_tickets_masks.target_percentage}_pruning_rounds_{cfg.winning_tickets_masks.pruning_rounds}')

def parameter_plasticity(cfg, dataset, device):
    task_ids = cfg.winning_tickets_plasticity.tasks

    for task_id in task_ids:
        print(f"\nTask {task_id + 1}")
        if cfg.general.dataset == 'mnist':
            model = models.SimpleMLP()
        else:
            raise NotImplementedError

        model = model.to(device)
        model.load_state_dict(torch.load(cfg.winning_tickets_masks.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
        model_name = f'snapshot_start_task{task_id}'
        
        train_dataset = dataset[0]        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.general.batch_size, shuffle=True)
        gradient = training.train_model_gradient(cfg, model, train_loader, device, num_epochs=cfg.general.num_epochs)  
        gradients_stats = {'avg_grad': gradient}

        utils.create_folder(cfg.parameter_plasticity.save_folder)
        utils.dump_pickle_obj(obj=gradients_stats, path=cfg.parameter_plasticity.save_folder + f'gradients_stats_{model_name}')

def winning_tickets_accuracy(cfg, dataset, device):
    task_performance = utils.load_pickle_obj(cfg.winning_tickets_accuracy.models_dir + 'task_performance')
    tickets_performance = {}
    
    task_ids = cfg.winning_tickets_accuracy.tasks

    for task_id in task_ids:
        tickets_performance[task_id] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        permutation = np.load(cfg.winning_tickets_accuracy.models_dir + f'permutation_task{task_id}.npy')
        if cfg.general.dataset == 'mnist':
            perm_train = datasets.mnist.PermutedMNIST(dataset[0], permutation)
            perm_test = datasets.mnist.PermutedMNIST(dataset[1], permutation)

            model = models.SimpleMLP()
            model = model.to(device)
        else:
            raise NotImplementedError

        train_loader = torch.utils.data.DataLoader(perm_train, batch_size=cfg.general.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(perm_test, batch_size=cfg.general.batch_size, shuffle=False)

        masks = utils.load_pickle_obj(cfg.winning_tickets_accuracy.masks_dir + f'masks_task_{task_id}_target_percentage_{cfg.winning_tickets_accuracy.target_percentage}_pruning_rounds_{cfg.winning_tickets_accuracy.pruning_rounds}')
        for mask in masks:
            model.load_state_dict(torch.load(cfg.winning_tickets_accuracy.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
            train_loss, train_acc = training.train_model(cfg, model, train_loader, device, num_epochs=cfg.general.num_epochs, mask=mask)
            test_loss, test_acc = training.evaluate_model(cfg, model, test_loader, device, mask=mask)

            tickets_performance[task_id]["train_loss"].append(train_loss)
            tickets_performance[task_id]["train_acc"].append(train_acc)
            tickets_performance[task_id]["test_loss"].append(test_loss)
            tickets_performance[task_id]["test_acc"].append(test_acc)

            with open(cfg.winning_tickets_accuracy.save_folder + 'output.txt', 'a') as file:
                print(f'Full model performance', file=file)
                print(f"Task {task_id + 1} - Train Loss: {task_performance['train_loss'][task_id]:.4f}, Train Acc: {task_performance['train_acc'][task_id]:.4f}", file=file)
                print(f"Task {task_id + 1} - Test Loss: {task_performance['test_loss'][task_id]:.4f}, Test Acc: {task_performance['test_acc'][task_id]:.4f}", file=file)

                print(f'Ticket performance', file=file)
                print(f"Task {task_id + 1} - Train Loss: {tickets_performance[task_id]['train_loss'][-1]:.4f}, Train Acc: {tickets_performance[task_id]['train_acc'][-1]:.4f}", file=file)
                print(f"Task {task_id + 1} - Test Loss: {tickets_performance[task_id]['test_loss'][-1]:.4f}, Test Acc: {tickets_performance[task_id]['test_acc'][-1]:.4f}", file=file)

                print('--------------------------------', file=file)

    utils.create_folder(cfg.winning_tickets_accuracy.save_folder)
    utils.dump_pickle_obj(tickets_performance, cfg.winning_tickets_accuracy.save_folder + 'tickets_performance')

def test_random_masks(cfg, dataset, device):
    task_performance = utils.load_pickle_obj(cfg.test_random_masks.models_dir + 'task_performance')
    random_tickets_performance = {}
    
    task_ids = cfg.test_random_masks.tasks

    for task_id in task_ids:
        random_tickets_performance[task_id] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        permutation = np.load(cfg.test_random_masks.models_dir + f'permutation_task{task_id}.npy')
        if cfg.general.dataset == 'mnist':
            perm_train = datasets.mnist.PermutedMNIST(dataset[0], permutation)
            perm_test = datasets.mnist.PermutedMNIST(dataset[1], permutation)

            model = models.SimpleMLP()
            model = model.to(device)
        else:
            raise NotImplementedError
        

        train_loader = torch.utils.data.DataLoader(perm_train, batch_size=cfg.general.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(perm_test, batch_size=cfg.general.batch_size, shuffle=False)

        for mask in range(cfg.test_random_masks.num_masks_per_task):
            def generate_random_mask(model, target_percentage):
                mask = []
                for name,param in model.named_parameters():
                    if 'weight' in name:
                        mask.append(torch.bernoulli(target_percentage * torch.ones_like(param)))
                return mask
            
            mask = generate_random_mask(model, cfg.test_random_masks.target_percentage)

            model.load_state_dict(torch.load(cfg.test_random_masks.models_dir + f'snapshot_start_task{task_id}', weights_only=True))
            train_loss, train_acc = training.train_model(cfg, model, train_loader, device, num_epochs=cfg.general.num_epochs, mask=mask)
            test_loss, test_acc = training.evaluate_model(cfg, model, test_loader, device, mask=mask)

            random_tickets_performance[task_id]["train_loss"].append(train_loss)
            random_tickets_performance[task_id]["train_acc"].append(train_acc)
            random_tickets_performance[task_id]["test_loss"].append(test_loss)
            random_tickets_performance[task_id]["test_acc"].append(test_acc)

            with open(cfg.test_random_masks.save_folder + 'output.txt', 'a') as file:
                print(f'Full model performance', file=file)
                print(f"Task {task_id + 1} - Train Loss: {task_performance['train_loss'][task_id]:.4f}, Train Acc: {task_performance['train_acc'][task_id]:.4f}", file=file)
                print(f"Task {task_id + 1} - Test Loss: {task_performance['test_loss'][task_id]:.4f}, Test Acc: {task_performance['test_acc'][task_id]:.4f}", file=file)

                print(f'Random ticket performance', file=file)
                print(f"Task {task_id + 1} - Train Loss: {random_tickets_performance[task_id]['train_loss'][-1]:.4f}, Train Acc: {random_tickets_performance[task_id]['train_acc'][-1]:.4f}", file=file)
                print(f"Task {task_id + 1} - Test Loss: {random_tickets_performance[task_id]['test_loss'][-1]:.4f}, Test Acc: {random_tickets_performance[task_id]['test_acc'][-1]:.4f}", file=file)

                print('--------------------------------', file=file)

    utils.create_folder(cfg.test_random_masks.save_folder)
    utils.dump_pickle_obj(random_tickets_performance, cfg.test_random_masks.save_folder + 'tickets_performance')

def overlap_parameters_tickets(cfg):
    stats_dict = {}
    masks_dir = cfg.overlap_parameters_tickets.masks_dir
    gradients_dir = cfg.overlap_parameters_tickets.gradients_dir

    task_ids = cfg.overlap_parameters_tickets.tasks
    for task_id in task_ids:
        target_percentage_values = cfg.overlap_paramters_tickets.target_percentages_values
        pruning_rounds_values = cfg.overlap_parameters_tickets.pruning_rounds_values

        for target_percentage, pruning_rounds in zip(target_percentage_values, pruning_rounds_values):
            print(f"running mask {task_id} with params {target_percentage} {pruning_rounds}")

            mask_file = open(masks_dir + f'masks_task_{task_id}_target_percentage_{target_percentage}_pruning_rounds_{pruning_rounds}','rb')
            masks = torch.load(mask_file)
            mask = lottery_tickets.mask_selection(cfg,masks)
            grad_file = open(gradients_dir + f'gradients_stats_snapshot_start_task{task_id}','rb')
            grad = utils.load_pickle_obj(grad_file)['avg_grad']

            if len(grad) > 2:
                grad = grad[:2]

            ones_percentage = utils.percentage_of_ones(mask)

            grad_mask = []

            for param_grad in grad:
                grad_mask.append(torch.ones_like(param_grad))

                threshold = torch.quantile(param_grad.abs().view(-1) ,1-ones_percentage)

                grad_mask[-1][param_grad.abs() < threshold] = 0

            ticket_parameters_overlap = lottery_tickets.compute_pairwise_overlap(mask,grad_mask)
            stats_dict[task_id]= ticket_parameters_overlap
            print(f"overlap between ticket and most important parameters : {ticket_parameters_overlap}")
            
            utils.create_folder(cfg.overlap_parameters_tickets.save_folder)
            utils.dump_pickle_obj(stats_dict,cfg.overlap_parameters_tickets.save_folder + f'overlap_parameters_tickets_{task_id}_target_percentage_{target_percentage}_pruning_rounds_{pruning_rounds}' )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = utils.load_config()
    utils.seed_everything(cfg.general.seed)

    if cfg.general.dataset == 'mnist':
        dataset = datasets.mnist.get_MNIST_dataset()
    else:
        raise NotImplementedError
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_name', 
        choices=['lop', 'winning_tickets_masks', 'winning_tickets_plasticity', 'winning_tickets_accuracy', 'test_random_masks'])

    args = parser.parse_args()

    if args.experiment_name == 'lop':
        lop_experiment(cfg=cfg, dataset=dataset, device=device)
    elif args.experiment_name == 'winning_tickets_masks':
        winning_tickets_masks(cfg=cfg, dataset=dataset, device=device)
        overlap_parameters_tickets(cfg=cfg)
    elif args.experiment_name == 'winning_tickets_plasticity':
        winning_tickets_plasticity(cfg=cfg, dataset=dataset, device=device)
    elif args.experiment_name == 'winning_tickets_accuracy':
        winning_tickets_accuracy(cfg=cfg, dataset=dataset, device=device)
    elif args.experiment_name == 'test_random_masks':
        test_random_masks(cfg=cfg, dataset=dataset, device=device)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()