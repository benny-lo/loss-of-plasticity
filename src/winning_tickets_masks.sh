#!/bin/bash

#SBATCH --account=dl_jobs
#SBATCH --job-name=winning_tickets_masks_2       
#SBATCH --output=out2.txt    
#SBATCH --error=error2.txt     
#SBATCH --time=1-00:00:00              



conda init
conda activate lop_experiment_env

target_percentage_values=(0.5 0.6 0.7)  
pruning_rounds_values=(5 7 10)  

for i in "${!target_percentage_values[@]}"; do
  target_percentage="${target_percentage_values[$i]}"
  pruning_rounds="${pruning_rounds_values[$i]}"


  yq -yi ".winning_tickets_masks.target_percentage = $target_percentage" src/config.yaml
  yq -yi ".winning_tickets_masks.pruning_rounds = $pruning_rounds" src/config.yaml

  echo "Running experiment with target_percentage=$target_percentage, pruning_rounds=$pruning_rounds"

  python src/experiments.py 
  

done
