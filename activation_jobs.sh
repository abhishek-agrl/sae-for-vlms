#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=sae_activation             # A nice readable name of your job, to see it in the queue
#SBATCH --partition=ampere
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=14           # Number of CPUs to request
#SBATCH --time=48:00:00
#SBATCH --output=logs/result_%j.out
#SBATCH --gres=gpu:1

module load mamba

mamba activate llava_env

# python3 run_extract_activations.py --dataset_type imagenet --output_file data/llava_imagenet_patch_validation_multiple_layers.h5 --batch_size 256 --total_images 50000
python3 run_extract_activations.py --dataset_type utkface --data_dir /home/abhishek.agrawal/utkface_split/train/ --output_file data/llava_utkface_patch_multiple_layers_train.h5 --batch_size 256
echo "Done"