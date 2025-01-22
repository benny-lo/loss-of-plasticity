import pickle
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device == 'cuda')

masks_dir = './results/winning_tickets_masks/masks/'
new_masks_dir = './results/winning_tickets_masks/new_masks/'


task_id_values = range(0,550,50)
target_percentage_values = [0.2,0.1,0.05]
pruning_rounds_values = [3,5,10]


for task_id in task_id_values:
    for idx in range(3):
        target_percentage = target_percentage_values[idx]
        pruning_rounds = pruning_rounds_values[idx]

        mask_file = open(masks_dir + f'masks_task_{task_id}_target_percentage_{target_percentage}_pruning_rounds_{pruning_rounds}','rb')
        masks = pickle.load(mask_file)

        new_masks = []

        for mask in masks:
            new_mask = []
            for idx in range(len(mask)):
                new_mask.append(mask[idx].cpu())
            new_masks.append(new_mask)


        torch.save(new_masks, new_masks_dir + f'masks_task_{task_id}_target_percentage_{target_percentage}_pruning_rounds_{pruning_rounds}')

        print(f"successfully reformatted mask {task_id} with params {target_percentage} {pruning_rounds}")