import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
import os

from sae_models import TopKSAE
from config import get_topk_config


class UTKActivationDataset(Dataset):
    def __init__(self, h5_path, feature_key="activations"):
        """
        feature_key: By default, "activations" points to the layer 17 features 
        we extracted. You could also pass "embeddings" if you want to train on those.
        """
        self.h5_path = h5_path
        self.feature_key = feature_key
        self.h5_file = None 
        
        # Open temporarily just to verify the key and get the dataset length
        with h5py.File(self.h5_path, 'r') as f:
            if feature_key not in f:
                available_keys = list(f.keys())
                raise ValueError(f"Key '{feature_key}' not found. Available keys: {available_keys}")
            self.length = f[feature_key].shape[0]
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy loading to prevent multiprocessing crashes with HDF5
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        x = torch.from_numpy(self.h5_file[self.feature_key][idx]).float()
        return x


def finetune_sae(dataloader, model, optimizer, epochs, cfg, save_path):
    print(f"Fine-tuning SAE on {len(dataloader.dataset)} UTKFace activations...")
    
    model.train()
    
    for epoch in range(epochs):
        total_loss_epoch = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for input_activations in pbar:
            input_activations = input_activations.to(cfg['device'])
            
            model_out = model(input_activations)
            
            loss = model_out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            
            model.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
                        
            total_loss_epoch += loss.item()
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
            })
            
        # Optional: Save checkpoints every 50 epochs just in case
        if (epoch + 1) % 50 == 0:
            checkpoint_path = save_path.replace(".pth", f"_ep{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
        
    torch.save(model.state_dict(), save_path)
    print(f"Final fine-tuned model saved to {save_path}")
        

if __name__ == "__main__":
    cfg = get_topk_config()

    # --- Configuration for Fine-tuning ---
    epochs = 200 # Usually requires fewer epochs than training from scratch
    cfg['batch_size'] = 256
    
    # We lower the learning rate for fine-tuning to avoid catastrophic forgetting 
    # of the ImageNet representations.
    cfg['lr'] = 1e-5 
    
    cfg['act_size'] = 1024
    cfg['dict_size'] = 1024 * 8
    cfg["l1_coeff"] = 1e-4

    # Paths
    data_path = "llava_utkface_features_train_18_to_65.h5"
    pretrained_model_path = "/home/abhishek.agrawal/vlm/TopKSAE_training_set_layer_17.pth" # Update this to your saved model name
    finetuned_save_path = "TopKSAE_layer_17_UTK_finetuned.pth"
    
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Initialize Model and Load Weights ---
    model = TopKSAE(cfg)
    
    if os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained weights from {pretrained_model_path}...")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=cfg['device']))
    else:
        raise FileNotFoundError(f"Could not find pre-trained model at {pretrained_model_path}. Please check the path.")
        
    model.to(cfg['device'])

    # --- Setup Optimizer and Data ---
    optimizer = Adam(model.parameters(), lr=cfg['lr'])
    
    dataset = UTKActivationDataset(data_path, feature_key="activations")
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # --- Run Fine-tuning ---
    finetune_sae(
        dataloader=dataloader, 
        model=model, 
        optimizer=optimizer, 
        epochs=epochs, 
        cfg=cfg,
        save_path=finetuned_save_path
    )