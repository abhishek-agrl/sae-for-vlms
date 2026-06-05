import torch
from sae_models import TopKSAE, VanillaSAE
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

data_path = "/home/abhishek.agrawal/vlm/llava_utkface_features_test_18_to_65.h5" # <-- UPDATE to your UTKFace HDF5 file
utkface_dir = "/home/abhishek.agrawal/utkface_split/test"
model_path = "TopKSAE_layer_17_UTK_finetuned.pth"
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
model = TopKSAE(cfg)
model.load_state_dict(torch.load(model_path, map_location=cfg['device'], weights_only=True))
model.eval()

# --- 1. Load Data ---
print("Loading activations, embeddings, and labels...")
with h5py.File(data_path, 'r') as f:
    activations = torch.from_numpy(f['activations'][:]).float()
    embeddings = torch.from_numpy(f['embeddings'][:]).float().to(cfg['device'])
    labels = f['labels'][:] # Age, Gender, Race

N = activations.size(0)

# Compute SAE features in batches
print("Computing SAE features...")
feature_acts_list = []
batch_size = 1000
with torch.no_grad():
    for i in tqdm(range(0, N, batch_size)):
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

# --- 2. On-the-Fly Similarity Matrix Setup ---
# L2-normalize embeddings so dot product equals cosine similarity
print("Normalizing embeddings for cosine similarity...")
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

num_neurons = a_tilde.shape[1]
ms_scores = torch.zeros(num_neurons)

# Since we use normalized cosine similarity, the diagonal (vector against itself) is strictly 1.0
s_diag = torch.ones(N, device=cfg['device']) 

print("Computing Mono-Semanticity Scores (Weighted Average)...")
chunk_size = 512
chunk_size_n = 5000

# --- 2. Custom Dataset for UTKFace with Age Filtering ---
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, min_age=18, max_age=65):
        all_image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.image_paths = []
        self.labels = []
        
        # Filter images by age during initialization
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            try:
                parts = filename.split('_')
                age = int(parts[0])
                
                # Apply the age condition
                if min_age <= age <= max_age:
                    gender = int(parts[1])
                    race = int(parts[2])
                    self.image_paths.append(img_path)
                    self.labels.append((age, gender, race))
            except (ValueError, IndexError):
                # Safely skip any files with missing or malformed data
                continue
                
        print(f"Filtered Dataset: Found {len(self.image_paths)} images matching age {min_age}-{max_age}")

    def __len__(self):
        return len(self.image_paths)

    def get_path(self, idx):
        return self.image_paths[idx]
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx] # Use the cached labels
        
        return image, label

# Loop over neurons
for i in tqdm(range(0, num_neurons, chunk_size), desc="Neurons"):
    a_chunk = a_tilde[:, i:i+chunk_size].to(cfg['device']) # [N, chunk_size]
    
    sum_all = torch.zeros(a_chunk.shape[1], device=cfg['device'])
    
    # Process similarity matrix in blocks ON THE FLY
    for j in range(0, N, chunk_size_n):
        # 1. Dynamically compute the S_chunk
        # S_chunk = E_chunk dot E_transpose
        E_chunk = embeddings[j:j+chunk_size_n]
        S_chunk = torch.matmul(E_chunk, embeddings.T) # Shape: [chunk_size_n, N]
        
        # 2. Multiply with activations
        S_a_chunk = torch.matmul(S_chunk, a_chunk)
        
        # CRITICAL: dim=0 to sum out the batch dimension
        sum_all += torch.sum(a_chunk[j:j+chunk_size_n] * S_a_chunk, dim=0) 
    
    # Subtract diagonal: sum_n (a^k_n)^2 * s_{nn}
    # Since s_diag is 1.0 everywhere, we can simplify this
    sum_diag = torch.sum((a_chunk ** 2), dim=0) 
    numerator = sum_all - sum_diag
    
    # Calculate Denominator (Weight Sum) - Algebraic shortcut
    sum_a = torch.sum(a_chunk, dim=0) 
    sum_sq_a = torch.sum(a_chunk ** 2, dim=0) 
    weight_sum = (sum_a ** 2) - sum_sq_a
    
    # Calculate Final Score (handling division by zero for dead features)
    ms_k = torch.where(
        weight_sum != 0, 
        numerator / weight_sum, 
        torch.tensor(float('nan'), device=cfg['device'])
    )
    
    ms_scores[i:i+chunk_size] = ms_k.cpu()

# --- 3. Post-Processing & UTKFace Interpretation ---

is_nan = torch.isnan(ms_scores)
nan_count = is_nan.sum().item()

# Filter out NaNs for accurate statistics
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
    
    # Get top 10 highest monosemantic features
    k_top = min(10, len(valid_ms_scores))
    top_values, top_indices = torch.topk(valid_ms_scores, k_top)
    original_top_indices = valid_indices_mapped[top_indices]
    
    print("\nTop Most Monosemantic SAE Features and their Top Activating Images:")
    
    # # Helper mapping for UTKFace Race integers
    # race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
    
    # Try to load the dataset for image paths (Optional)
    
    
    dataset = UTKFaceDataset(root_dir=utkface_dir,min_age=18, max_age=65)
    
    for idx, val in zip(original_top_indices, top_values):
        idx_item = idx.item()
        print(f"\nFeature {idx_item} - MS Score: {val.item():.4f}")
        
        # Get top 5 images that activate this feature
        feat_acts = feature_acts[:, idx_item]
        top_acts, top_img_indices = torch.topk(feat_acts, 5)
        
        for i in range(5):
            img_idx = top_img_indices[i].item()
            act_val = top_acts[i].item()
            
            # Extract UTKFace labels
            age, gender, race = labels[img_idx]
            gender_str = "Female" if gender == 1 else "Male"
            race_str = race_map.get(race, "Unknown")

            msg = f"  - Image {img_idx:5d} | Act: {act_val:.4f} | Demographics: {age} yrs, {gender_str}, {race_str}"
            img_path = dataset.get_path(img_idx)
            msg += f" (Path: {img_path})"
            print(msg)
torch.save(ms_scores, 'ms_scores_utk.pth')
print("\nSaved ms_scores to ms_scores_utk.pth")

# --- Find Pure Gender Neurons (Filtered by Frequency) ---
print("\n--- Hunting for Pure Gender Neurons ---")

# 1. Define our strict thresholds
MIN_IMAGES = 500        # A feature must fire on at least 50 images to be valid
MIN_ACTIVATION = 0.3   # Ignore tiny noise activations below this threshold

# 2. Extract Gender Labels (Assuming UTKFace: 0 is Male, 1 is Female)
# Convert the HDF5 labels array to a tensor and slice ONLY the gender column (index 1)
labels_tensor = torch.tensor(labels, device=cfg['device'])
genders = labels_tensor[:, 1] if labels_tensor.dim() > 1 else labels_tensor

# Move activations to GPU for fast matrix math
feature_acts_gpu = feature_acts.to(cfg['device'])

# 3. Frequency Gating: Count how many images each neuron fires on
is_active = feature_acts_gpu > MIN_ACTIVATION
firing_counts = is_active.sum(dim=0) # Shape: [dict_size]

# Create a mask of valid "concept" features (fires enough times to matter)
valid_features_mask = firing_counts >= MIN_IMAGES

# 4. Calculate Concept Purity
# Now these masks will perfectly evaluate to shape [N, 1]
is_female_mask = (genders == 1).float().unsqueeze(1) 
is_male_mask = (genders == 0).float().unsqueeze(1)   

# Sum total activations for each feature [dict_size]
total_acts = feature_acts_gpu.sum(dim=0)

# Sum activations specifically for female and male images
# The shapes are now [N, 8192] * [N, 1], which broadcasts correctly!
female_acts_sum = (feature_acts_gpu * is_female_mask).sum(dim=0)

# Calculate Female Purity: 1.0 means it ONLY fires for females, 0.0 means ONLY males
# We add 1e-9 to prevent division by zero for completely dead features
purity_scores = female_acts_sum / (total_acts + 1e-9)

# 5. Mask out the invalid/ultra-sparse features CORRECTLY
# For Female (searching for max), set invalid features to 0.0
purity_scores_f = torch.where(
    valid_features_mask, 
    purity_scores, 
    torch.zeros_like(purity_scores)
)

# For Male (searching for min), set invalid features to 1.0
purity_scores_m = torch.where(
    valid_features_mask, 
    purity_scores, 
    torch.ones_like(purity_scores)
)

# --- 6. Extract and Print Results ---
k_top = 10

# Top Female Features (Purity closest to 1.0)
top_f_vals, top_f_idx = torch.topk(purity_scores_f, k_top, largest=True)

# Top Male Features (Purity closest to 0.0)
top_m_vals, top_m_idx = torch.topk(purity_scores_m, k_top, largest=False)

print(f"\nTOP {k_top} STRICTLY FEMALE NEURONS (Must fire on >= {MIN_IMAGES} images)")
for idx, purity in zip(top_f_idx, top_f_vals):
    f_count = firing_counts[idx].item()
    print(f"Feature {idx.item():5d} | Purity: {purity.item()*100:.1f}% Female | Fires on: {f_count} images")

print(f"\nTOP {k_top} STRICTLY MALE NEURONS (Must fire on >= {MIN_IMAGES} images)")
for idx, purity in zip(top_m_idx, top_m_vals):
    f_count = firing_counts[idx].item()
    # If purity is 0.0 for females, it is 1.0 (100%) for males
    male_purity = (1.0 - purity.item()) * 100 
    print(f"Feature {idx.item():5d} | Purity: {male_purity:.1f}% Male   | Fires on: {f_count} images")