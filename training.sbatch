#!/bin/bash -l 
#SBATCH --time=06:00:00  
#SBATCH --job-name=SegmentationTraining
#SBATCH --gres=gpu:1 
#SBATCH --export=NONE 
#SBATCH --mail-type=ALL   	# (optional) Send a mail when the job ends or fails


unset SLURM_EXPORT_ENV

module unload python

# load modules
module load python/3.10-anaconda

# Activate your virtual environment if needed
source myenv/bin/activate

# Run your Python script
python training.py

# deactivate env
deactivate
# echo "model-trianing done"


module add amber-gpu/20p08-at20p12-gnu-cuda11.2 

### there is no need to fiddle around with CUDA_VISIBLE_DEVICES! 
pmemd.cuda -O -i mdin ...