import torch
torch.set_float32_matmul_precision('high') # For Ampere GPUs speedup

import sys
from torch.utils.data import DataLoader, Dataset
import h5py
from math import sqrt
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
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

if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 4096
    expansion_size = 64
    act_size = 1024
    dict_size = 1024 * expansion_size
    lr = 16/(125 * sqrt(dict_size))
    top_k = 20
    
    data_path = "/home/abhishek.agrawal/vlm/data/llava_multiple_layers.h5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # layer_to_train = 'layer_17' # Available: 'layer_11', 'layer_17', 'layer_22', 'layer_23'
    layer_to_train = sys.argv[-1]

    save_dir = f'{layer_to_train}/TopKSAE_training_set_{layer_to_train}'
    # --- Data Loading ---
    dataset = ActivationDataset(data_path, layer_to_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    
    # dictionary_learning uses total training steps rather than epochs
    total_steps = int(1e5)
    
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
        normalize_activations=True
    )
    print(f"Training complete. Model saved to {save_dir}")