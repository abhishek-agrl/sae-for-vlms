import torch

# Enable Tensor Cores for Ampere (A100)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high') # For Ampere GPUs speedup

import os
import sys
from torch.utils.data import DataLoader, Dataset
import h5py
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_l0_distribution(l0_counts, title, save_path):
    """Generates a histogram of active features per sample matching the reference style."""
    plt.figure(figsize=(10, 5))
    
    # Calculate statistics
    mean_val = np.mean(l0_counts)
    median_val = np.median(l0_counts)
    std_val = np.std(l0_counts)
    min_val = np.min(l0_counts)
    max_val = np.max(l0_counts)
    
    # Define bins ensuring each integer gets its own bar
    bins = np.arange(0, max(max_val + 2, 100)) - 0.5 
    
    # Plot histogram
    plt.hist(l0_counts, bins=bins, color='royalblue', alpha=0.8, edgecolor='darkblue', linewidth=0.5)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='forestgreen', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    
    # Styling
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Active Features')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='upper left')
    
    # Statistics text box
    stats_text = (f"Mean: {mean_val:.2f}\n"
                  f"Median: {median_val:.2f}\n"
                  f"Std Dev: {std_val:.2f}\n"
                  f"Min: {int(min_val)}\n"
                  f"Max: {int(max_val)}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='dimgrey')
    plt.text(0.97, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved distribution plot to {save_path}")

def evaluate_sae(ae, val_dataloader, device="cuda"):
    print("\nStarting Rigorous SAE Evaluation on Validation Set...")
    ae.eval()
    
    total_squared_error = 0.0
    total_variance = 0.0
    total_l0 = 0.0
    total_samples = 0
    feature_dim = None
    
    # Store L0 counts for every sample
    all_l0_counts = []

    with torch.no_grad():
        for x in val_dataloader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)

            if feature_dim is None:
                feature_dim = x.size(-1)
            
            # # --- CRITICAL FIX: Normalize input to match training distribution ---
            # norms = torch.norm(x, p=2, dim=-1, keepdim=True)
            # scale_factor = math.sqrt(feature_dim) / (norms + 1e-8)
            # x_normalized = x * scale_factor
            
            # # Forward pass in the normalized space
            # latents = ae.encode(x_normalized)
            # x_hat_normalized = ae.decode(latents)
            
            # # De-normalize the reconstruction back to the original scale
            # x_hat = x_hat_normalized / scale_factor

            # Forward pass
            latents = ae.encode(x)
            x_hat = ae.decode(latents)
            
            # 1. Accumulate Total Squared Error (for MSE)
            total_squared_error += torch.sum((x_hat - x) ** 2).item()
            
            # 2. Accumulate Total Variance (for R2)
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            total_variance += torch.sum((x - batch_mean) ** 2).item()
            
            # 3. Accumulate L0 Sparsity
            l0_per_sample = (latents > 0).float().sum(dim=-1)
            total_l0 += l0_per_sample.sum().item()
            
            # Save per-sample counts for the histogram
            all_l0_counts.append(l0_per_sample.cpu())
            
            total_samples += batch_size

    # Calculate final global metrics
    global_mse = total_squared_error / (total_samples * feature_dim)
    global_fve = 1.0 - (total_squared_error / total_variance)
    global_l0 = total_l0 / total_samples
    
    # Concatenate all batches into a single numpy array
    l0_array = torch.cat(all_l0_counts).numpy()
    
    print("-" * 30)
    print("Global Validation Metrics:")
    print(f"MSE:                {global_mse:.6f}")
    print(f"Explained Var (R2): {global_fve:.4f} ({global_fve*100:.2f}%)")
    print(f"L0 Sparsity:        {global_l0:.2f}")
    print("-" * 30)
    
    return global_mse, global_fve, global_l0, l0_array

class ActivationDataset(Dataset):
    def __init__(self, h5_path, layer_name):
        print(f"Loading {layer_name} from HDF5 into RAM... This will take a moment.")
        with h5py.File(h5_path, 'r') as f:
            if layer_name not in f:
                available_layers = list(f.keys())
                raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
            
            np_data = f[layer_name][:]
            
        self.data = torch.from_numpy(np_data).to(torch.bfloat16)
        print(f"Successfully loaded {self.data.shape} activations into memory!")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # Ensure sys.argv has the layer name, otherwise use a fallback for testing
    layer_to_train = sys.argv[-1] if len(sys.argv) > 1 else "layer_11"
    
    # --- Shared Configuration ---
    batch_size = 4096
    expansion_size = 64
    act_size = 1024
    dict_size = 1024 * expansion_size
    top_k_val = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_data_path = "data/llava_imagenet_patch_validation_multiple_layers.h5" 
    
    print("Setting up Validation Dataloader...")
    val_dataset = ActivationDataset(val_data_path, layer_to_train)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ==========================================
    # 1. Evaluate BatchTopKSAE
    # ==========================================
    print("\n" + "="*50)
    print("Evaluating BatchTopKSAE")
    print("="*50)
    
    batch_topk_dir = f'{layer_to_train}/BatchTopKSAE_patch_{layer_to_train}_1e5_24'
    
    trained_batch_ae = BatchTopKSAE(
        activation_dim=act_size, 
        dict_size=dict_size,
        k=top_k_val
    )
    
    batch_weights_path = os.path.join(batch_topk_dir, "trainer_0", "ae.pt") 
    if os.path.exists(batch_weights_path):
        trained_batch_ae.load_state_dict(torch.load(batch_weights_path, map_location=device, weights_only=True))
        trained_batch_ae = trained_batch_ae.to(device)
        
        _, _, _, batch_l0_array = evaluate_sae(trained_batch_ae, val_dataloader, device)
        
        # Plot and save histogram
        plot_l0_distribution(
            batch_l0_array, 
            title="Distribution of Active Features per Sample (BatchTopK)", 
            save_path=f"BatchTopK_L0_Distribution_{layer_to_train}.png"
        )
    else:
        print(f"Skipping BatchTopK: Weights not found at {batch_weights_path}")


    # ==========================================
    # 2. Evaluate Standard AutoEncoderTopK
    # ==========================================
    print("\n" + "="*50)
    print("Evaluating AutoEncoderTopK")
    print("="*50)

    topk_dir = f'{layer_to_train}/TopKSAE_patch_{layer_to_train}_1e5_24'

    trained_topk_ae = AutoEncoderTopK(
        activation_dim=act_size, 
        dict_size=dict_size,
        k=top_k_val
    )
    
    topk_weights_path = os.path.join(topk_dir, "trainer_0", "ae.pt") 
    if os.path.exists(topk_weights_path):
        trained_topk_ae.load_state_dict(torch.load(topk_weights_path, map_location=device, weights_only=True))
        trained_topk_ae = trained_topk_ae.to(device)
        
        _, _, _, topk_l0_array = evaluate_sae(trained_topk_ae, val_dataloader, device)
        
        # Plot and save histogram
        plot_l0_distribution(
            topk_l0_array, 
            title="Distribution of Active Features per Sample (Standard TopK)", 
            save_path=f"StandardTopK_L0_Distribution_{layer_to_train}.png"
        )
    else:
        print(f"Skipping Standard TopK: Weights not found at {topk_weights_path}")

    print("\nAll Training and Evaluation Finished!")