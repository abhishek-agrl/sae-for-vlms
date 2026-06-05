import torch
from sae_models import TopKSAE, VanillaSAE
from config import get_topk_config
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import os

# print(f"Is CUDA available: {torch.cuda.is_available()}")
# print(f"Device count: {torch.cuda.device_count()}")

cfg = get_topk_config()
    
epochs = 1000
cfg['batch_size'] = 256
cfg['lr'] = 3e-4
cfg['act_size']=1024
cfg['dict_size']=1024*8
cfg["l1_coeff"] = 1e-2
# Controls sparsity. Higher = more sparse.

data_path = "/home/abhishek.agrawal/vlm/llava_activations.h5"
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

model = TopKSAE(cfg)
model.load_state_dict(torch.load('TopKSAE.pth', map_location=cfg['device'], weights_only=True))
# model.load_state_dict(torch.load('TopKSAE_training_set_layer_17.pth', map_location=cfg['device'], weights_only=True))
model.eval()

# Load original activations
print("Loading activations...")
with h5py.File(data_path, 'r') as f:
    activations = torch.from_numpy(f['activations'][:]).float()

# Compute SAE features in batches
print("Computing SAE features...")
feature_acts_list = []
batch_size = 1000
with torch.no_grad():
    for i in tqdm(range(0, activations.size(0), batch_size)):
        batch = activations[i:i+batch_size].to(cfg['device'])
        out = model(batch)
        feature_acts_list.append(out["feature_acts"].cpu())
feature_acts = torch.cat(feature_acts_list, dim=0) # [N, dict_size]

# Min-Max Normalization per neuron
print("Normalizing features...")
a_min = feature_acts.min(dim=0, keepdim=True).values
a_max = feature_acts.max(dim=0, keepdim=True).values
denom = a_max - a_min
denom[denom == 0] = 1.0
a_tilde = (feature_acts - a_min) / denom
# Load similarity matrix to CPU to save GPU memory
print("Loading similarity matrix to CPU...")
similarity_matrix = torch.load('similarity_matrix', map_location='cpu')
N = a_tilde.shape[0]
num_neurons = a_tilde.shape[1]
ms_scores = torch.zeros(num_neurons)

s_diag = similarity_matrix.diag()

print(a_tilde.shape, similarity_matrix.shape, s_diag.shape)
print("Computing Mono-Semanticity Scores (Weighted Average)...")
chunk_size = 512
chunk_size_n = 5000

for i in tqdm(range(0, num_neurons, chunk_size), desc="Neurons"):
    a_chunk = a_tilde[:, i:i+chunk_size].to(cfg['device']) # [N, chunk_size]
    
    sum_all = torch.zeros(a_chunk.shape[1], device=cfg['device'])
    
    # 1. Calculate Numerator (Similarity Sum) - Processed in blocks
    for j in range(0, N, chunk_size_n):
        # Cast S_chunk to match a_chunk's float32 dtype right before math
        S_chunk = similarity_matrix[j:j+chunk_size_n, :].to(device=cfg['device'], dtype=a_chunk.dtype) 
        S_a_chunk = torch.matmul(S_chunk, a_chunk)
        # CRITICAL: dim=0 must be here to sum out the batch dimension
        sum_all += torch.sum(a_chunk[j:j+chunk_size_n] * S_a_chunk, dim=0) 
    
    # Subtract diagonal: sum_n (a^k_n)^2 * s_{nn}
    s_diag_chunk = s_diag.to(device=cfg['device'], dtype=a_chunk.dtype)
    
    # CRITICAL: dim=0 must be here to sum out the batch dimension
    sum_diag = torch.sum((a_chunk ** 2) * s_diag_chunk.unsqueeze(1), dim=0) 
    numerator = sum_all - sum_diag
    
    # 2. Calculate Denominator (Weight Sum) - Using the algebraic shortcut
    # CRITICAL: dim=0 must be on both of these
    sum_a = torch.sum(a_chunk, dim=0) 
    sum_sq_a = torch.sum(a_chunk ** 2, dim=0) 
    weight_sum = (sum_a ** 2) - sum_sq_a
    
    # 3. Calculate Final Score (handling division by zero for dead features)
    ms_k = torch.where(
        weight_sum != 0, 
        numerator / weight_sum, 
        torch.tensor(float('nan'), device=cfg['device'])
    )
    
    ms_scores[i:i+chunk_size] = ms_k.cpu()

# ... (Keep the post-processing and printouts from the previous code here)
# --- Post-Processing (Adapted from Script 1) ---

is_nan = torch.isnan(ms_scores)
nan_count = is_nan.sum().item()

# Filter out NaNs for accurate statistics
valid_indices = ~is_nan
valid_ms_scores = ms_scores[valid_indices]
valid_indices_mapped = torch.nonzero(valid_indices).squeeze()

print(f"\nResults Overview:")
print(f"Total Features: {num_neurons}")
print(f"Dead/Inactive Features: {nan_count}")

if len(valid_ms_scores) > 0:
    print(f"Mean Score: {valid_ms_scores.mean().item():.4f} +- {valid_ms_scores.std().item():.4f}")
    print(f"Max Score:  {valid_ms_scores.max().item():.4f}")
    
    # Get top 10 highest monosemantic features
    k_top = min(10, len(valid_ms_scores))
    top_values, top_indices = torch.topk(valid_ms_scores, k_top)
    original_top_indices = valid_indices_mapped[top_indices]
    
    # Try to load the dataset for image paths (Optional)
    imagenet_path = "/home/abhishek.agrawal/ImageNet_ILSVRC2012"
    dataset = None
    if os.path.exists(imagenet_path):
        try:
            dataset = datasets.ImageNet(root=imagenet_path, split='val')
        except Exception:
            dataset = None

    print("\nTop Most Monosemantic SAE Features and their Top Activating Images:")
    for idx, val in zip(original_top_indices, top_values):
        idx_item = idx.item()
        print(f"Feature {idx_item} - MS Score: {val.item():.4f}")
        
        # Get top 5 images that activate this feature
        feat_acts = feature_acts[:, idx_item]
        top_acts, top_img_indices = torch.topk(feat_acts, 5)
        
        for i in range(5):
            img_idx = top_img_indices[i].item()
            act_val = top_acts[i].item()
            msg = f"  - Image Index {img_idx}: Activation {act_val:.4f}"
            if dataset:
                try:
                    img_path, _ = dataset.samples[img_idx]
                    msg += f" (Path: {img_path})"
                except Exception:
                    pass
            print(msg)

torch.save(ms_scores, 'ms_scores_weighted.pth')
print("\nSaved ms_scores to ms_scores_weighted.pth")