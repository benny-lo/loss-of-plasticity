#!/bin/bash

#SBATCH --account=dl
#SBATCH --job-name=test        
#SBATCH --output=test.txt    
#SBATCH --error=test_error.txt     
#SBATCH --time=1-00:00:00 

python src/experiments.py --experiment_name "$1"