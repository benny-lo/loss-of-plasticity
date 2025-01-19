#!/bin/bash

#SBATCH --account=dl_cpu
#SBATCH --job-name=test        
#SBATCH --output=test.txt    
#SBATCH --error=test_error.txt     
#SBATCH --time=00:20:00 


conda init
conda activate lop_env


yq -yi ".general.num_epochs = 1" src/config.yaml
yq -yi ".winning_tickets_masks.num_tickets = 1" src/config.yaml
yq -yi ".winning_tickets_masks.pruning_rounds = 1" src/config.yaml

python src/experiments.py 