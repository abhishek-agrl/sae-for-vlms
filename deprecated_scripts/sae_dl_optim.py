import torch

# Enable Tensor Cores for Ampere (A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high') # For Ampere GPUs speedup

import torch.nn.functional as F
import os
import sys
from torch.utils.data import DataLoader, Dataset
import h5py
from math import sqrt
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.training import trainSAE

class ActivationDataset(Dataset):
    def __init__(self, h5_path, layer_name):
        print(f"Loading {layer_name} from HDF5 into RAM... This will take a moment.")
        with h5py.File(h5_path, 'r') as f:
            if layer_name not in f:
                available_layers = list(f.keys())
                raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
            
            # The [:] operator loads the entire dataset into a numpy array in RAM
            np_data = f[layer_name][:]
            
        # Convert to a float32 PyTorch tensor in CPU RAM immediately
        self.data = torch.from_numpy(np_data).to(torch.bfloat16)
        print(f"Successfully loaded {self.data.shape} activations into memory!")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Now this is just a lightning-fast memory lookup
        return self.data[idx]
    
def make_device_iterator(dataloader, target_device):
    """
    Infinitely yields batches from the dataloader, 
    moving them to the correct device for the trainer.
    """
    while True:
        for batch in dataloader:
            yield batch.to(target_device, non_blocking=True)

def evaluate_sae(ae, val_dataloader, device="cuda"):
    print("\nStarting Rigorous SAE Evaluation on Validation Set...")
    ae.eval()
    
    total_squared_error = 0.0
    total_variance = 0.0
    total_l0 = 0.0
    total_samples = 0
    feature_dim = None

    with torch.no_grad():
        for x in val_dataloader:
            x = x.to(device, non_blocking=True)
            batch_size = x.size(0)

            if feature_dim is None:
                feature_dim = x.size(-1)
            
            # Forward pass
            latents = ae.encode(x)
            x_hat = ae.decode(latents)
            
            # 1. Accumulate Total Squared Error (for MSE)
            # We use sum() instead of mean() so we can divide by the total dataset size later
            total_squared_error += torch.sum((x_hat - x) ** 2).item()
            
            # 2. Accumulate Total Variance (for R2)
            # Variance is the squared difference from the mean. 
            # With large batches (4096), the batch mean is a safe proxy for the global mean.
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            total_variance += torch.sum((x - batch_mean) ** 2).item()
            
            # 3. Accumulate L0 Sparsity
            total_l0 += (latents > 0).float().sum(dim=-1).sum().item()
            
            total_samples += batch_size

    # Calculate final global metrics
    # Divide by (total_samples * feature_dim) because MSE is averaged over features too
    global_mse = total_squared_error / (total_samples * feature_dim)
    
    # R2 = 1 - (Sum of Squared Errors / Total Sum of Squares)
    global_fve = 1.0 - (total_squared_error / total_variance)
    
    # L0 is average active features per image
    global_l0 = total_l0 / total_samples
    
    print("-" * 30)
    print("Global Validation Metrics:")
    print(f"MSE:                {global_mse:.6f}")
    print(f"Explained Var (R2): {global_fve:.4f} ({global_fve*100:.2f}%)")
    print(f"L0 Sparsity:        {global_l0:.2f}")
    print("-" * 30)
    
    return global_mse, global_fve, global_l0



if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 4096
    expansion_size = 64
    act_size = 1024
    dict_size = 1024 * expansion_size
    lr = 16/(125 * sqrt(dict_size))
    batch_top_k = 20
    # dictionary_learning uses total training steps rather than epochs
    total_steps = 110000
    
    data_path = "llava_patch_multiple_layers.h5"
    # data_path = "/home/abhishek.agrawal/vlm/data/llava_imagenet_patch_multiple_layers.h5"
    # data_path = "/home/abhishek.agrawal/vlm/data/llava_multiple_layers.h5"
    # data_path = "data/llava_validation_multiple_layers.h5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # layer_to_train = 'layer_17' # Available: 'layer_11', 'layer_17', 'layer_22', 'layer_23'
    layer_to_train = sys.argv[-1]

    save_dir = f'{layer_to_train}/BatchTopKSAE_patch_{layer_to_train}_1e5_24'
    # --- Data Loading ---
    dataset = ActivationDataset(data_path, layer_to_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": act_size,
        "dict_size": dict_size,
        "lr": lr,
        "device": device,
        "steps": total_steps,
        "layer": layer_to_train,
        "lm_name": "",
        
        # --- The New Pach et al. Hyperparameters ---
        "k": batch_top_k,
        "auxk_alpha": 0.03,
        "decay_start": 109999,
        "threshold_beta": 0.999,         # Moving average for the Batch-K threshold
        "threshold_start_step": 1000,    # When to start enforcing the strict Batch-K threshold
    }

    device_iterator = make_device_iterator(dataloader, device)

    print(f"Training Batch TopK SAE on {len(dataset)} activations of dim {act_size}...")
    print(f"Hidden Dimension: {dict_size}")
    print(f"Total training steps: {total_steps}")
    
    # --- Training ---
    # trainSAE handles the optimization, loss calculation, and logging internally
    ae = trainSAE(
        data=device_iterator, 
        # data=dataloader,
        trainer_configs=[trainer_cfg],
        steps=total_steps,
        save_dir=save_dir,
        autocast_dtype=torch.bfloat16,
        normalize_activations=True,
    )
    print(f"Training complete. Model saved to {save_dir}")

    print("Starting Validation...")
    # 1. Setup the Validation DataLoader
    # val_data_path = "data/llava_validation_multiple_layers.h5" 
    val_data_path = "data/llava_imagenet_patch_validation_multiple_layers.h5"
    val_dataset = ActivationDataset(val_data_path, layer_to_train)
    val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    # 2. Re-instantiate the empty SAE architecture
    # Note: Check the exact initialization arguments for AutoEncoderTopK in your library version.
    # They are usually similar to what you passed in trainer_cfg.
    trained_ae = BatchTopKSAE(
        activation_dim=act_size, 
        dict_size=dict_size,
        k=batch_top_k
    )
    
    # 3. Load the trained weights
    # The library usually saves the weights as 'ae.pt' or 'pytorch_model.bin' inside the save_dir.
    weights_path = os.path.join(save_dir,"trainer_0", "ae.pt") 
    
    if not os.path.exists(weights_path):
        # Fallback just in case the library named it something else (like step_100000.pt)
        available_files = os.listdir(save_dir)
        raise FileNotFoundError(f"Could not find ae.pt. Available files in {save_dir}: {available_files}")
        
    trained_ae.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    trained_ae = trained_ae.to(device)
    
    # 4. Run the rigorous evaluation
    evaluate_sae(trained_ae, val_dataloader, device)

    print("Training and Evaluation Finished...")

    # # --- Configuration ---
    batch_size = 4096
    expansion_size = 64
    act_size = 1024
    dict_size = 1024 * expansion_size
    lr = 16/(125 * sqrt(dict_size))
    top_k = 20
    # dictionary_learning uses total training steps rather than epochs
    total_steps = 110000
    
    data_path = "llava_patch_multiple_layers.h5"
    # data_path = "/home/abhishek.agrawal/vlm/data/llava_multiple_layers.h5"
    # data_path = "data/llava_validation_multiple_layers.h5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # layer_to_train = 'layer_17' # Available: 'layer_11', 'layer_17', 'layer_22', 'layer_23'
    layer_to_train = sys.argv[-1]

    save_dir = f'{layer_to_train}/TopKSAE_patch_{layer_to_train}_1e5_24'
    # --- Data Loading ---
    dataset = ActivationDataset(data_path, layer_to_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    # --- Trainer Config ---
    trainer_cfg = {
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": act_size,
        "dict_size": dict_size,
        "lr": lr,
        "device": device,
        "steps": total_steps,
        "k": top_k,
        "layer": "",
        "lm_name": "",
        "auxk_alpha": 0.03,
        "decay_start": 109999,
    }

    device_iterator = make_device_iterator(dataloader, device)

    print(f"Training TopK SAE on {len(dataset)} activations of dim {act_size}...")
    print(f"Hidden Dimension: {dict_size}")
    print(f"Total training steps: {total_steps}")
    
    # --- Training ---
    # trainSAE handles the optimization, loss calculation, and logging internally
    ae = trainSAE(
        data=device_iterator, 
        # data=dataloader,
        trainer_configs=[trainer_cfg],
        steps=total_steps,
        save_dir=save_dir,
        autocast_dtype=torch.bfloat16,
        normalize_activations=True,
    )
    print(f"Training complete. Model saved to {save_dir}")

    print("Starting Validation...")
    # 1. Setup the Validation DataLoader
    val_data_path = "data/llava_imagenet_patch_validation_multiple_layers.h5" 
    val_dataset = ActivationDataset(val_data_path, layer_to_train)
    val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    # 2. Re-instantiate the empty SAE architecture
    # Note: Check the exact initialization arguments for AutoEncoderTopK in your library version.
    # They are usually similar to what you passed in trainer_cfg.
    trained_ae = AutoEncoderTopK(
        activation_dim=act_size, 
        dict_size=dict_size,
        k=top_k
    )
    
    # 3. Load the trained weights
    # The library usually saves the weights as 'ae.pt' or 'pytorch_model.bin' inside the save_dir.
    weights_path = os.path.join(save_dir,"trainer_0", "ae.pt") 
    
    if not os.path.exists(weights_path):
        # Fallback just in case the library named it something else (like step_100000.pt)
        available_files = os.listdir(save_dir)
        raise FileNotFoundError(f"Could not find ae.pt. Available files in {save_dir}: {available_files}")
        
    trained_ae.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    trained_ae = trained_ae.to(device)
    
    # 4. Run the rigorous evaluation
    evaluate_sae(trained_ae, val_dataloader, device)

    print("Training and Evaluation Finished...")