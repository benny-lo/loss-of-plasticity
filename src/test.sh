#!/bin/bash

#SBATCH --account=dl
#SBATCH --job-name=test        
#SBATCH --output=test.txt    
#SBATCH --error=test_error.txt     
#SBATCH --time=1-00:00:00 


conda init
conda activate lop_env


python src/experiments.py