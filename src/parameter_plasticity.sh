#!/bin/bash

#SBATCH --account=dl
#SBATCH --job-name=parameter_plasticity       
#SBATCH --output=plasticity_out.txt    
#SBATCH --error=plasticity_error.txt     
#SBATCH --time=2:00:00               

conda init
conda activate lop_env

python src/parameter_plasticity.py