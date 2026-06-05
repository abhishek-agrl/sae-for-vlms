#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=sae_array             # A nice readable name of your job, to see it in the queue
#SBATCH --partition=volta
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=4           # Number of CPUs to request
#SBATCH --time=48:00:00
#SBATCH --output=logs/result_%j.out
##SBATCH --array=0-2  # Launches 3 identical jobs indexed 0, 1, 2
#SBATCH --gres=gpu:1

# LAYERS=("layer_17" "layer_22" "layer_11")

module load mamba

# Activate your environment, you have to create it first
mamba activate llava_env

# Your job script goes below this line
# python3 topk_sae_ms_score.py
# python3 sae.py
# python3 utk_activations.py
# python3 utk_fine_tune.py
# python3 utk_topk_ms_score.py
# python3 utk_topk_ms_score_modified.py
# python3 utk_steering.py
# python3 activations_training_set.py
# python3 sae_training_set.py

# python3 run_steering_experiment.py --mode optuna --utk_image_dir /home/abhishek.agrawal/utkface_split/test/ --sae_paths batch_topk_sae_17=layer_17/BatchTopKSAE_patch_layer_17_1e5_24/trainer_0/ae.pt --sae_type batch_topk
python3 run_steering_experiment.py --mode standard --utk_image_dir /home/abhishek.agrawal/utkface_split/test/ --sae_paths batch_topk_sae_17=layer_17/BatchTopKSAE_patch_layer_17_1e5_24/trainer_0/ae.pt --sae_type batch_topk --interventions batch_topk_sae_17:39278:-40
echo "Done"