#!/bin/bash

#SBATCH --job-name=winning_tickets_masks        
#SBATCH --output=out.txt    
#SBATCH --error=error.txt     
#SBATCH --time=2-10:00:00               
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=4             
#SBATCH --mem=8G                      

conda activate lop_env

target_percentage_values=(0.2 0.1 0.05)  
pruning_rounds_values=(3 5 10)  

for i in "${!target_percentage_values[@]}"; do
  target_percentage="${target_percentage_values[$i]}"
  pruning_rounds="${pruning_rounds_values[$i]}"


  yq -yi ".winning_tickets_masks.target_percentage = $target_percentage" src/config.yaml
  yq -yi ".winning_tickets_masks.pruning_rounds = $pruning_rounds" src/config.yaml

  echo "Running experiment with target_percentage=$target_percentage, pruning_rounds=$pruning_rounds

  python src/experiment.py 

done
