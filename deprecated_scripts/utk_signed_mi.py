import torch
from tqdm import tqdm
import h5py
import os
import sys
import math

# Use the correct dictionary_learning import from your working pipeline
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

# --- Configuration ---
act_size = 1024
expansion_size = 64
dict_size = 1024 * expansion_size
top_k = 20

# Activation Consistency Threshold (tau)
TAU = float(sys.argv[-1])
# TAU = 0.7

# layer_name = sys.argv[-1]
layer_name = "layer_22"

# data_path = "data/llava_utkface_multiple_layers_all.h5" 
data_path = "data/llava_utkface_patch_multiple_layers_train.h5" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# model_path = f"{layer_name}/BatchTopKSAE_patch_{layer_name}_1e5_24/trainer_0/ae.pt"
# model_path = f"{layer_name}/TopKSAE_patch_{layer_name}_1e5_old/trainer_0/ae.pt"
# model_path = f"{layer_name}/TopKSAE_{layer_name}_1e5/trainer_0/ae.pt"
model_path = f"{layer_name}/TopKSAE_patch_{layer_name}_1e5_24/trainer_0/ae.pt"
 
# --- 1. Load Model ---
print(f"Loading {model_path}")
# model = BatchTopKSAE(
#     activation_dim=act_size, 
#     dict_size=dict_size,
#     k=top_k
# )
model = AutoEncoderTopK(
    activation_dim=act_size, 
    dict_size=dict_size,
    k=top_k
)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# # --- 2. Load Data ---
# print(f"Loading {layer_name} activations and labels...")
# with h5py.File(data_path, 'r') as f:
#     activations = torch.from_numpy(f[layer_name][:]).float()
#     # Labels structure: [Age, Gender, Race]
#     labels = torch.from_numpy(f['labels'][:]).long() 

# N = activations.size(0)
# # UTKFace: 0 for Male, 1 for Female
# genders = labels[:, 1].to(device) 

# --- 2. Load Data ---
print(f"Loading {layer_name} activations and labels...")
with h5py.File(data_path, 'r') as f:
    activations = torch.from_numpy(f[layer_name][:]).float()
    # Labels structure: [Age, Gender, Race]
    labels = torch.from_numpy(f['labels'][:]).long() 

# --- CRITICAL FIX: Align Labels with Patch Activations ---
if activations.size(0) == labels.size(0) * 2:
    print(f"Detected 2 patches per image. Expanding {labels.size(0)} labels to {activations.size(0)}...")
    labels = torch.repeat_interleave(labels, repeats=2, dim=0)
elif activations.size(0) != labels.size(0):
    raise ValueError(f"Unexpected shape mismatch! Activations: {activations.size(0)}, Labels: {labels.size(0)}")

N = activations.size(0)
# UTKFace: 0 for Male, 1 for Female
genders = labels[:, 1].to(device)

# --- 3. Compute SAE features ---
print("Computing SAE features in batches...")
feature_acts_list = []
batch_size = 4096
with torch.no_grad():
    for i in tqdm(range(0, N, batch_size)):
        batch = activations[i:i+batch_size].to(device)

        # # Calculate the L2 norm of each embedding in the batch
        # norms = torch.norm(batch, p=2, dim=-1, keepdim=True)
        # # Scale the batch so its norm equals sqrt(act_size)
        # batch_normalized = batch * (math.sqrt(act_size) / (norms + 1e-8))

        latents = model.encode(batch)
        feature_acts_list.append(latents.cpu())

feature_acts = torch.cat(feature_acts_list, dim=0)
feature_acts_gpu = feature_acts.to(device)

print("\n--- Calculating Metrics ---")

# --- 4. Binarize Features for Probabilities & Consistency ---
# Indicator function I[a_ij]: 1 if active, 0 otherwise
ACT_THRESHOLD = 0.05 
F_active = (feature_acts_gpu > ACT_THRESHOLD).float()

Y_female = (genders == 1).float().unsqueeze(1)
Y_male = (genders == 0).float().unsqueeze(1)

# --- 5. Activation Consistency (AC) Filter ---
# Equation 7: freq = (1/N_c) * sum(I[a_ij])
freq_female = F_active[genders == 1].mean(dim=0)
freq_male = F_active[genders == 0].mean(dim=0)

count_female = F_active[genders == 1].sum(dim=0)
count_male = F_active[genders == 0].sum(dim=0)

# Equation 8: Consistent = 1 if freq >= tau else 0
consistent_female = freq_female >= TAU
consistent_male = freq_male >= TAU

# Remove features consistently active in BOTH classes to ensure class specificity
consistent_both = consistent_female & consistent_male
valid_female_mask = consistent_female & ~consistent_both
valid_male_mask = consistent_male & ~consistent_both

print(f"Features passing AC filter (tau={TAU}):")
print(f"  - Strictly Female Consistent: {valid_female_mask.sum().item()}")
print(f"  - Strictly Male Consistent:   {valid_male_mask.sum().item()}")

# --- 6. Calculate Signed Mutual Information (Signed-MI) ---
# Marginals
p_y1 = Y_female.mean() 
p_y0 = Y_male.mean()   
p_f1 = F_active.mean(dim=0)
p_f0 = 1.0 - p_f1

# Joint Probabilities
p_f1_y1 = (F_active * Y_female).mean(dim=0)
p_f1_y0 = (F_active * Y_male).mean(dim=0)
p_f0_y1 = ((1 - F_active) * Y_female).mean(dim=0)
p_f0_y0 = ((1 - F_active) * Y_male).mean(dim=0)

# Equation 5: MI Formula
def calc_mi_term(p_joint, p_marginal_f, p_marginal_y):
    mask = p_joint > 0
    term = torch.zeros_like(p_joint)
    denominator = p_marginal_f * p_marginal_y + 1e-9
    term[mask] = p_joint[mask] * torch.log2(p_joint[mask] / denominator[mask])
    return term

mi = (
    calc_mi_term(p_f1_y1, p_f1, p_y1) +
    calc_mi_term(p_f1_y0, p_f1, p_y0) +
    calc_mi_term(p_f0_y1, p_f0, p_y1) +
    calc_mi_term(p_f0_y0, p_f0, p_y0)
)

# Equation 6: Signed-MI
mean_act_female = feature_acts_gpu[genders == 1].mean(dim=0)
mean_act_male = feature_acts_gpu[genders == 0].mean(dim=0)

delta_a = mean_act_female - mean_act_male
signed_mi = mi * torch.sign(delta_a)

# --- 7. Apply AC Filter to Signed-MI and Extract Top Features ---
# We use negative/positive infinity to exclude invalid features from the top-k selection
filtered_mi_female = torch.where(valid_female_mask, signed_mi, torch.tensor(float('-inf'), device=device))
filtered_mi_male = torch.where(valid_male_mask, signed_mi, torch.tensor(float('inf'), device=device))

def print_top_mi_features(scores, means_target, means_other, counts_target, counts_other, gender_name, largest=True):
    vals, idxs = torch.topk(scores, 10, largest=largest)
    
    print(f"\n--- Top 10 Signed-MI {gender_name} Neurons (AC Filtered) ---")
    # Expanded formatting to fit the new count columns
    print(f"{'Index':<8} | {'Signed-MI':<10} | {'Avg Tgt Act':<12} | {'Tgt Fires':<10} | {'Avg Oth Act':<12} | {'Oth Fires':<10}")
    
    for i in range(10):
        idx = idxs[i].item()
        score = vals[i].item()
        
        # If the score is infinity, it means we ran out of features that passed the AC filter
        if float('inf') in [abs(score)]:
            print("  [No more features passed the AC filter]")
            break
            
        act_t = means_target[idx].item()
        act_o = means_other[idx].item()
        c_t = int(counts_target[idx].item())
        c_o = int(counts_other[idx].item())
        
        print(f"{idx:<8} | {score:<10.4f} | {act_t:<12.4f} | {c_t:<10d} | {act_o:<12.4f} | {c_o:<10d}")

# Largest positive values = Female associated
print_top_mi_features(
    filtered_mi_female, mean_act_female, mean_act_male, 
    count_female, count_male, "Female", largest=True
)

# Most negative values = Male associated
print_top_mi_features(
    filtered_mi_male, mean_act_male, mean_act_female, 
    count_male, count_female, "Male", largest=False
)