#!/bin/bash

#SBATCH --account=dl
#SBATCH --job-name=test        
#SBATCH --output=slurm_out.txt    
#SBATCH --error=slurm_error.txt     
#SBATCH --time=1-00:00:00 

python src/experiments.py $1