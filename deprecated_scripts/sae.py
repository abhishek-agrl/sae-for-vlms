import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm

from sae_models import TopKSAE, VanillaSAE
from config import get_topk_config


class ActivationDataset(Dataset):
    def __init__(self, h5_path):
        self.data = h5py.File(h5_path, 'r')['activations']
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        return x


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, expansion_factor):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim)
        

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.decoder(h), h

def train_sae(dataloader, model, optimizer):
    print(f"Training SAE on {len(dataset)} activations of dim {cfg['act_size']}...")
    print(f"Hidden Dimension: {cfg['dict_size']}")
    
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
        
    torch.save(model.state_dict(), 'TopKSAE.pth')
        

if __name__ == "__main__":
    cfg = get_topk_config()
    

    epochs = 1000
    cfg['batch_size'] = 256
    cfg['lr'] = 5e-5
    cfg['act_size']=1024
    cfg['dict_size']=1024*8
    cfg["l1_coeff"] = 1e-4
    # Controls sparsity. Higher = more sparse.

    data_path = "/home/abhishek.agrawal/vlm/llava_activations.h5"
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    model = TopKSAE(cfg)
    # model = VanillaSAE(cfg)
    # model = SparseAutoencoder(input_dim, expansion_factor).to(device)
    optimizer = Adam(model.parameters(), lr=cfg['lr'])
    
    dataset = ActivationDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    
    
    train_sae(dataloader=dataloader, model=model, optimizer=optimizer)
