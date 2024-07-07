#!/bin/bash -l
#SBATCH --job-name=Segmentation_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=yugashree.yc.chaudhari@fau.de
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --export=None

unset SLURM_EXPORT_ENV

echo "Your job is running"

module load cuda/11.2.0

# Activate conda environment
conda activate MasterThesis

CONFIG = config.yaml
# Run your training Python script
python3 training.py --config ${CONFIG}
