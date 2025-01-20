#!/bin/bash

#SBATCH --account=dl
#SBATCH --job-name=winning_tickets_masks_2       
#SBATCH --output=out2.txt    
#SBATCH --error=error2.txt     
#SBATCH --time=1-00:00:00               



conda init
conda activate lop_env

target_percentage_values=(0.2 0.1 0.05)  
pruning_rounds_values=(3 5 10)  

for i in "${!target_percentage_values[@]}"; do
  target_percentage="${target_percentage_values[$i]}"
  pruning_rounds="${pruning_rounds_values[$i]}"


  yq -yi ".winning_tickets_masks.target_percentage = $target_percentage" src/config.yaml
  yq -yi ".winning_tickets_masks.pruning_rounds = $pruning_rounds" src/config.yaml

  echo "Running experiment with target_percentage=$target_percentage, pruning_rounds=$pruning_rounds"

  python src/experiments.py 
  

done
