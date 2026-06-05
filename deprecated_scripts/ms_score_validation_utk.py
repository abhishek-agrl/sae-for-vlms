import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import os
import sys
import glob
from PIL import Image

# Import the SAE architecture from the dictionary_learning library
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
import math
# --- Configuration ---
act_size = 1024
expansion_size = 64
dict_size = 1024 * expansion_size
top_k = 20

# Point this to your UTKFace HDF5 file
# data_path = "data/llava_utkface_multiple_layers_all.h5" 
# utkface_dir = "/home/abhishek.agrawal/utkface_aligned_cropped/crop_part1/"
data_path = "data/llava_utkface_patch_multiple_layers_train.h5" 
utkface_dir = "/home/abhishek.agrawal/utkface_split/train/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dynamic layer targeting
layer_name = sys.argv[-1]

# Define the path where trainSAE saved your weights (Using your trained ImageNet SAE)
# model_path = f"{layer_name}/BatchTopKSAE_{layer_name}_1e5/trainer_0/ae.pt"
# model_path = f"{layer_name}/TopKSAE_{layer_name}_1e5/trainer_0/ae.pt"
model_path = f"{layer_name}/TopKSAE_patch_{layer_name}_1e5_24/trainer_0/ae.pt"
# model_path = f"{layer_name}/BatchTopKSAE_patch_{layer_name}_1e5_24/trainer_0/ae.pt"

print(f"Loading TopK SAE for {layer_name}...")
# 1. Initialize AutoEncoder
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

# Load the weights safely
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- 1. Load Data ---
print(f"Loading {layer_name} activations and last_embeddings...")
with h5py.File(data_path, 'r') as f:
    activations = torch.from_numpy(f[layer_name][:]).float()
    
    # Load the last_embedding to use for our on-the-fly similarity matrix
    if 'last_embedding' not in f:
        raise KeyError("Could not find 'last_embedding' in the HDF5 file.")
    embeddings = torch.from_numpy(f['last_embedding'][:]).float().to(device)

    labels = torch.from_numpy(f['labels'][:]).long()

# --- CRITICAL FIX: Align Embeddings with Patch Activations ---
if activations.size(0) == embeddings.size(0) * 2:
    print(f"Detected 2 patches per image. Expanding {embeddings.shape} embeddings to {activations.shape}...")
    # This turns [A, B, C] into [A, A, B, B, C, C]
    embeddings = torch.repeat_interleave(embeddings, repeats=2, dim=0)
    labels = torch.repeat_interleave(labels, repeats=2, dim=0)
elif activations.size(0) != embeddings.size(0):
    raise ValueError(f"Unexpected shape mismatch! Activations: {activations.shape}, Embeddings: {embeddings.shape}")

N = activations.size(0)

# Compute SAE features in batches
print("Computing SAE features...")
feature_acts_list = []
batch_size = 4096 
with torch.no_grad():
    for i in tqdm(range(0, N, batch_size)):
        batch = activations[i:i+batch_size].to(device)
        # NEW: Apply L2 Norm scaling to match the training distribution
        # norms = torch.norm(batch, p=2, dim=-1, keepdim=True)
        # scale_factor = math.sqrt(act_size) / (norms + 1e-8)
        # batch_normalized = batch * scale_factor
        
        # Encode the normalized batch
        # latents = model.encode(batch_normalized)
        latents = model.encode(batch)
        feature_acts_list.append(latents.cpu())

feature_acts = torch.cat(feature_acts_list, dim=0) # [N, dict_size]

# Min-Max Normalization per neuron
print("Normalizing features...")
a_min = feature_acts.min(dim=0, keepdim=True).values
a_max = feature_acts.max(dim=0, keepdim=True).values
denom = a_max - a_min
denom[denom == 0] = 1.0
a_tilde = (feature_acts - a_min) / denom

# --- 2. On-the-Fly Similarity Matrix Setup ---
# L2-normalize embeddings so dot product equals cosine similarity
print("Normalizing embeddings for on-the-fly cosine similarity...")
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

num_neurons = a_tilde.size(1)
ms_scores = torch.zeros(num_neurons)

print("Computing Mono-Semanticity Scores (Weighted Average)...")
chunk_size = 512
# chunk_size_n = 5000
chunk_size_n = 10000

# Loop over neurons
for i in tqdm(range(0, num_neurons, chunk_size), desc="Neurons"):
    a_chunk = a_tilde[:, i:i+chunk_size].to(device) # [N, chunk_size]
    
    sum_all = torch.zeros(a_chunk.size(1), device=device)
    
    # Process similarity matrix in blocks ON THE FLY
    for j in range(0, N, chunk_size_n):
        E_chunk = embeddings[j:j+chunk_size_n]
        S_chunk = torch.matmul(E_chunk, embeddings.T) 
        S_a_chunk = torch.matmul(S_chunk, a_chunk)
        sum_all += torch.sum(a_chunk[j:j+chunk_size_n] * S_a_chunk, dim=0) 
    
    # Subtract diagonal
    sum_diag = torch.sum((a_chunk ** 2), dim=0) 
    numerator = sum_all - sum_diag
    
    # Calculate Denominator (Weight Sum)
    sum_a = torch.sum(a_chunk, dim=0) 
    sum_sq_a = torch.sum(a_chunk ** 2, dim=0) 
    weight_sum = (sum_a ** 2) - sum_sq_a
    
    # Calculate Final Score
    ms_k = torch.where(
        weight_sum != 0, 
        numerator / weight_sum, 
        torch.tensor(float('nan'), device=device)
    )
    
    ms_scores[i:i+chunk_size] = ms_k.cpu()

# --- 3. Custom Dataset for UTKFace Interpretation ---
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, min_age=18, max_age=65):
        all_image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.image_paths = []
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            try:
                age = int(filename.split('_')[0])
                if min_age <= age <= max_age:
                    self.image_paths.append(img_path)
            except (ValueError, IndexError):
                continue

# --- 4. Post-Processing & Interpretation ---
is_nan = torch.isnan(ms_scores)
nan_count = is_nan.sum().item()

valid_indices = ~is_nan
valid_ms_scores = ms_scores[valid_indices]
valid_indices_mapped = torch.nonzero(valid_indices).squeeze()

print(f"\nResults Overview:")
print(f"Total Features: {num_neurons}")
print(f"Dead/Inactive Features: {nan_count}")

race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

if len(valid_ms_scores) > 0:
    print(f"Mean Score: {valid_ms_scores.mean().item():.4f} +- {valid_ms_scores.std().item():.4f}")
    print(f"Max Score:  {valid_ms_scores.max().item():.4f}")
    
    k_top = min(10, len(valid_ms_scores))
    top_values, top_indices = torch.topk(valid_ms_scores, k_top)
    original_top_indices = valid_indices_mapped[top_indices]
    
    print("\nTop Most Monosemantic SAE Features and their Top Activating Images:")
    
    # Load dataset to map indices to actual file paths
    try:
        dataset = UTKFaceDataset(root_dir=utkface_dir, min_age=18, max_age=65)
    except Exception:
        dataset = None
    
    for idx, val in zip(original_top_indices, top_values):
        idx_item = idx.item()
        print(f"\nFeature {idx_item} - MS Score: {val.item():.4f}")
        
        feat_acts = feature_acts[:, idx_item]
        top_acts, top_img_indices = torch.topk(feat_acts, 5)
        
        for i in range(5):
            patch_idx = top_img_indices[i].item()
            act_val = top_acts[i].item()
            
            # Extract demographic info from the HDF5 labels array
            age, gender, race = labels[patch_idx].tolist()
            gender_str = "Female" if gender == 1 else "Male"
            race_str = race_map.get(race, "Unknown")

            # NEW: Map the patch index back to the original image index
            orig_img_idx = patch_idx // 2
            
            msg = f"  - Patch {patch_idx:5d} (Image {orig_img_idx}) | Act: {act_val:.4f} | Demographics: {age} yrs, {gender_str}, {race_str}"
            if dataset and orig_img_idx < len(dataset.image_paths):
                img_path = dataset.image_paths[orig_img_idx]
                msg += f" (Path: {img_path})"

            # msg = f"  - Image {img_idx:5d} | Act: {act_val:.4f} | Demographics: {age} yrs, {gender_str}, {race_str}"
            # if dataset and img_idx < len(dataset.image_paths):
            #     img_path = dataset.image_paths[img_idx]
            #     msg += f" (Path: {img_path})"
            
            print(msg)

output_filename = f'results/ms_scores_utkface_topk_{layer_name}.pth'
torch.save(ms_scores, output_filename)
print(f"\nSaved ms_scores to {output_filename}")