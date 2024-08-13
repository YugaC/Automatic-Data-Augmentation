#!/bin/bash -l
#SBATCH --job-name=Segmentation_training
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --mail-user=yugashree.yc.chaudhari@fau.de
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --export=None

unset SLURM_EXPORT_ENV

echo "Your job is running"


module load cuda/11.2.0

module add python

# Activate conda environment
conda activate ThesisYugashree

conda info

CONFIG=config.yaml

# enter the working dir with config.yaml 
cd /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation

# Run your training Python script
python /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/training.py --config ${CONFIG}

python /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/Metric_original_image_spacing.py --config ${CONFIG}

python /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/best_model_output_viz.py --config ${CONFIG}

python /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/best_model_output.py --config ${CONFIG}

echo "Test images are now processing"

#python /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/inference_on_Testset.py --config ${CONFIG}

python /home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/inference_test_viz.py --config ${CONFIG}