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

# Run your training Python script
python training.py
if [ $? -eq 0 ]; then
  echo "Training completed successfully."
else
  echo "Training failed."
  exit 1
fi

# Run Validation file
python Metric_original_image_spacing.py
if [ $? -eq 0 ]; then
  echo "Validation completed successfully."
else
  echo "Validation failed."
  exit 1
fi


### there is no need to fiddle around with CUDA_VISIBLE_DEVICES! 
pmemd.cuda -O -i mdin ...