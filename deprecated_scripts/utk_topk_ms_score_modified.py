import torch
from sae_models import TopKSAE
from config import get_topk_config
from tqdm import tqdm
import h5py
import os
from PIL import Image
from torch.utils.data import Dataset
import glob

# --- Configuration ---
cfg = get_topk_config()
epochs = 1000
cfg['batch_size'] = 256
cfg['lr'] = 3e-4
cfg['act_size'] = 1024
cfg['dict_size'] = 1024 * 8
cfg["l1_coeff"] = 1e-4

cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# data_path = "llava_utkface_features_18_to_65.h5" 
data_path = "/home/abhishek.agrawal/vlm/llava_utkface_features_train_18_to_65.h5"
model_path = "/home/abhishek.agrawal/vlm/TopKSAE_training_set_layer_17.pth"
# Load Model
model = TopKSAE(cfg)
# model.load_state_dict(torch.load('TopKSAE.pth', map_location=cfg['device'], weights_only=True))
model.load_state_dict(torch.load(model_path, map_location=cfg['device'], weights_only=True))
model.eval()

# --- 1. Load Data ---
print("Loading activations and labels...")
with h5py.File(data_path, 'r') as f:
    activations = torch.from_numpy(f['activations'][:]).float()
    # Labels structure: [Age, Gender, Race]
    labels = torch.from_numpy(f['labels'][:]).long() 

N = activations.size(0)
genders = labels[:, 1].to(cfg['device']) # 0 for Male, 1 for Female

# Compute SAE features in batches
print("Computing SAE features...")
feature_acts_list = []
batch_size = 1000
with torch.no_grad():
    for i in tqdm(range(0, N, batch_size)):
        batch = activations[i:i+batch_size].to(cfg['device'])
        out = model(batch)
        feature_acts_list.append(out["feature_acts"].cpu())
feature_acts = torch.cat(feature_acts_list, dim=0)

# 3. Calculate Robust Gender Metrics
print("\nCalculating Robust Gender Scores...")
feature_acts_gpu = feature_acts.to(cfg['device'])

print("\nCalculating Signed Mutual Information (Signed-MI)...")

# 1. Binarize Features and Labels for Probability Calculation
# We consider a feature "active" if it overcomes a small noise threshold
ACT_THRESHOLD = 0.05 
F_active = (feature_acts_gpu > ACT_THRESHOLD).float()

# Labels: 1 for Female, 0 for Male
Y_female = (genders == 1).float().unsqueeze(1)
Y_male = (genders == 0).float().unsqueeze(1)

# N = F_active.shape

# --- 2. Calculate Marginals ---
# p(y): Probability of each class in the dataset
p_y1 = Y_female.mean() # p(Female)
p_y0 = Y_male.mean()   # p(Male)

# p(fi): Probability of each feature being active/inactive [Shape: dict_size]
p_f1 = F_active.mean(dim=0)
p_f0 = 1.0 - p_f1

# --- 3. Calculate Joint Probabilities p(fi, y) ---
# p(f=1, y=1): Feature Active AND Female
p_f1_y1 = (F_active * Y_female).mean(dim=0)
# p(f=1, y=0): Feature Active AND Male
p_f1_y0 = (F_active * Y_male).mean(dim=0)
# p(f=0, y=1): Feature Inactive AND Female
p_f0_y1 = ((1 - F_active) * Y_female).mean(dim=0)
# p(f=0, y=0): Feature Inactive AND Male
p_f0_y0 = ((1 - F_active) * Y_male).mean(dim=0)

# --- 4. Calculate Mutual Information ---
# Formula: Sum over f,y of p(f,y) * log( p(f,y) / (p(f)*p(y)) )
# We add 1e-9 to avoid log(0) errors
def calc_mi_term(p_joint, p_marginal_f, p_marginal_y):
    # If joint prob is 0, the limit of x*log(x) as x->0 is 0. 
    # We use a mask to enforce this and avoid NaN.
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

# --- 5. Calculate Signed-MI ---
# Per Formula (6) in the paper: MI * sign(delta mean act)
mean_act_female = feature_acts_gpu[genders == 1].mean(dim=0)
mean_act_male = feature_acts_gpu[genders == 0].mean(dim=0)

delta_a = mean_act_female - mean_act_male
signed_mi = mi * torch.sign(delta_a)

# --- 6. Extract Top Features ---
def print_top_mi_features(scores, means_target, means_other, gender_name, largest=True):
    vals, idxs = torch.topk(scores, 10, largest=largest)
    
    print(f"\n--- Top 10 Signed-MI {gender_name} Neurons ---")
    print(f"{'Index':<8} | {'Signed-MI':<10} | {'Avg Target Act':<15} | {'Avg Other Act':<15}")
    
    for i in range(10):
        idx = idxs[i].item()
        score = vals[i].item()
        act_t = means_target[idx].item()
        act_o = means_other[idx].item()
        
        print(f"{idx:<8} | {score:<10.4f} | {act_t:<15.4f} | {act_o:<15.4f}")

# Largest positive values = Female associated
print_top_mi_features(signed_mi, mean_act_female, mean_act_male, "Female", largest=True)

# Most negative values = Male associated
print_top_mi_features(signed_mi, mean_act_male, mean_act_female, "Male", largest=False)