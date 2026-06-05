import os
import sys
import argparse
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.config import get_topk_config
from src.data import ActivationDataset
from src.models import TopKSAE, BatchTopKSAE, JumpReLUSAE, VanillaSAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Fine-tune Sparse Autoencoders (SAEs) on activations.")
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], required=True, help="Mode: 'train' from scratch or 'finetune' existing checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 activations file")
    parser.add_argument("--layer", type=str, required=True, help="Layer key to read from HDF5 (e.g. layer_17)")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save the resulting checkpoint or directory")
    
    # Model config
    parser.add_argument("--sae_type", type=str, choices=["topk", "batch_topk", "jumprelu", "vanilla"], default="topk", help="SAE model class type")
    parser.add_argument("--dict_size", type=int, default=1024 * 8, help="SAE dictionary/latent size (hidden dim)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K sparsity parameter")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto-calculated or type-specific)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 4096 for train, 256 for finetune)")
    parser.add_argument("--l1_coeff", type=float, default=1e-4, help="L1 regularization coefficient (where applicable)")
    
    # Train/Finetune execution config
    parser.add_argument("--steps", type=int, default=100000, help="Total steps to train (train mode)")
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs to train (finetune mode)")
    parser.add_argument("--pretrained_model_path", type=str, default="", help="Path to pretrained checkpoint (required for finetune)")
    
    return parser.parse_args()


def make_device_iterator(dataloader, target_device):
    """Infinitely yields batches from the dataloader, moving them to the correct device."""
    while True:
        for batch in dataloader:
            yield batch.to(target_device, non_blocking=True)


def train_from_scratch(args, device):
    """Trains an SAE from scratch, using the dictionary_learning library if requested."""
    try:
        from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
        from dictionary_learning.training import trainSAE
        use_lib = True
    except ImportError:
        print("Warning: dictionary_learning library not found. Falling back to training custom models directly.")
        use_lib = False
        
    batch_size = args.batch_size if args.batch_size is not None else 4096
    act_size = 1024
    lr = args.lr if args.lr is not None else 16 / (125 * math.sqrt(args.dict_size))
    
    print(f"Loading activations dataset from {args.data_path}...")
    # Convert to bfloat16 for speed/memory benefits during training from scratch
    dataset = ActivationDataset(args.data_path, args.layer, lazy=False, dtype=torch.bfloat16)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    if use_lib:
        print("Training using dictionary_learning library...")
        trainer_cfg = {
            "trainer": TopKTrainer,
            "dict_class": AutoEncoderTopK,
            "activation_dim": act_size,
            "dict_size": args.dict_size,
            "lr": lr,
            "device": device,
            "steps": args.steps,
            "k": args.top_k,
            "layer": "",
            "lm_name": "",
        }
        
        device_iterator = make_device_iterator(dataloader, device)
        
        # Create output dir if needed
        os.makedirs(args.save_path, exist_ok=True)
        
        trainSAE(
            data=device_iterator,
            trainer_configs=[trainer_cfg],
            steps=args.steps,
            save_dir=args.save_path,
            autocast_dtype=torch.bfloat16,
            normalize_activations=True
        )
        print(f"Training completed successfully. Model saved to {args.save_path}")
    else:
        # Custom training loop
        print("Training custom model directly...")
        cfg = get_topk_config()
        cfg.update({
            "device": device,
            "act_size": act_size,
            "dict_size": args.dict_size,
            "top_k": args.top_k,
            "l1_coeff": args.l1_coeff,
            "lr": lr,
            "dtype": torch.float32,
        })
        
        if args.sae_type == "topk":
            model = TopKSAE(cfg)
        elif args.sae_type == "batch_topk":
            model = BatchTopKSAE(cfg)
        elif args.sae_type == "vanilla":
            model = VanillaSAE(cfg)
        else:
            raise NotImplementedError(f"Direct training not fully implemented for custom '{args.sae_type}' type.")
            
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        
        model.train()
        step = 0
        pbar = tqdm(total=args.steps, desc="Training")
        
        while step < args.steps:
            for batch in dataloader:
                batch = batch.to(device).float()
                
                model_out = model(batch)
                loss = model_out["loss"]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                model.make_decoder_weights_and_grad_unit_norm()
                
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                if step >= args.steps:
                    break
                    
        pbar.close()
        # Save custom checkpoint
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Training completed. Model saved to {args.save_path}")


def finetune_checkpoint(args, device):
    """Fine-tunes an existing custom SAE model checkpoint."""
    if not args.pretrained_model_path:
        raise ValueError("--pretrained_model_path is required in finetune mode")
        
    batch_size = args.batch_size if args.batch_size is not None else 256
    lr = args.lr if args.lr is not None else 1e-5
    act_size = 1024
    
    cfg = get_topk_config()
    cfg.update({
        "device": device,
        "act_size": act_size,
        "dict_size": args.dict_size,
        "top_k": args.top_k,
        "l1_coeff": args.l1_coeff,
        "lr": lr,
        "dtype": torch.float32,
    })
    
    # Load model
    print(f"Initializing model and loading pretrained weights from {args.pretrained_model_path}...")
    if args.sae_type == "topk":
        model = TopKSAE(cfg)
    elif args.sae_type == "batch_topk":
        model = BatchTopKSAE(cfg)
    elif args.sae_type == "jumprelu":
        model = JumpReLUSAE(cfg)
    elif args.sae_type == "vanilla":
        model = VanillaSAE(cfg)
    else:
        raise ValueError(f"Unknown custom sae_type: {args.sae_type}")
        
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    model.to(device)
    
    # Setup optimizer and data
    optimizer = Adam(model.parameters(), lr=lr)
    
    print(f"Loading dataset from {args.data_path}...")
    dataset = ActivationDataset(args.data_path, args.layer, lazy=True, dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss_epoch = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = batch.to(device).float()
            
            model_out = model(batch)
            loss = model_out["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            model.make_decoder_weights_and_grad_unit_norm()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss_epoch += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # Save checkpoints periodically
        if (epoch + 1) % 50 == 0:
            checkpoint_path = args.save_path.replace(".pth", f"_ep{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Fine-tuning complete. Final model saved to {args.save_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.mode == "train":
        train_from_scratch(args, device)
    elif args.mode == "finetune":
        finetune_checkpoint(args, device)


if __name__ == "__main__":
    main()
