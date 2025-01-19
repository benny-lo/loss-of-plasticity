#!/bin/bash

#SBATCH --account=dl_jobs
#SBATCH --job-name=parameter_plasticity       
#SBATCH --output=plasticity_out.txt    
#SBATCH --error=plasticity_error.txt     
#SBATCH --time=1-00:00:00               

conda init
conda activate lop_env

python src/parameter_plasticity.py