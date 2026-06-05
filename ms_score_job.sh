#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=sae_ms_score             # A nice readable name of your job, to see it in the queue
##SBATCH --partition=ampere
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=2           # Number of CPUs to request
#SBATCH --time=48:00:00
#SBATCH --output=logs/result_%j.out
#SBATCH --gres=gpu:1


# #SBATCH --array=0-2  # Launches 3 identical jobs indexed 0, 1, 2
# LAYERS=("layer_11" "layer_17" "layer_22")
# CURRENT_LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

#SBATCH --array=0-4  # Launches 3 identical jobs indexed 0, 1, 2, 3, 4
TAU=("0.1" "0.3" "0.5" "0.7" "0.9")
# TAU=("0.01" "0.1")
CURRENT_TAU=${TAU[$SLURM_ARRAY_TASK_ID]}

module load mamba

mamba activate llava_env

# python3 ms_score_validation_imagenet.py $CURRENT_LAYER
# echo "########################################################################"
# python3 ms_score_validation_utk.py $CURRENT_LAYER
# echo "########################################################################"
# python3 run_ms_score.py --metric ms_score --data_path data/llava_imagenet_patch_validation_multiple_layers.h5 --layer layer_17 --sae_path layer_17/BatchTopKSAE_patch_layer_17_1e5_24/trainer_0/ae.pt --dict_size 65536
python3 run_ms_score.py --metric signed_mi --data_path data/llava_utkface_patch_multiple_layers_train.h5 --layer layer_22 --sae_path layer_22/TopKSAE_patch_layer_22_1e5_24/trainer_0/ae.pt --dict_size 65536 --tau $CURRENT_TAU

echo "Done"