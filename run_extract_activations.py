import os
import argparse
import h5py
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from src.models import load_vlm
from src.data import UTKFaceDataset

# Speed up matrix multiplications on GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations from VLM vision tower and store in HDF5.")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="HF VLM Model ID")
    parser.add_argument("--dataset_type", type=str, choices=["utkface", "imagenet"], required=True, help="Dataset to extract from")
    parser.add_argument("--data_dir", type=str, default="", help="Path to local dataset directory (required for utkface)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output HDF5 file")
    parser.add_argument("--batch_size", type=str, default="256", help="Batch size for forward passes")
    parser.add_argument("--total_images", type=int, default=50000, help="Max images to process for ImageNet")
    parser.add_argument("--patches_per_image", type=int, default=2, help="Number of random patch tokens to capture per image (0 for CLS token)")
    parser.add_argument("--min_age", type=int, default=18, help="Min age for UTKFace filtering")
    parser.add_argument("--max_age", type=int, default=65, help="Max age for UTKFace filtering")
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = int(args.batch_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_vlm(args.model_id, device_map="auto")
    
    # 1. Prepare Dataset and Loader
    if args.dataset_type == "utkface":
        if not args.data_dir:
            raise ValueError("--data_dir must be specified for utkface dataset type")
        print(f"Loading local UTKFace dataset from {args.data_dir}...")
        dataset = UTKFaceDataset(args.data_dir, min_age=args.min_age, max_age=args.max_age)
        total_images = len(dataset)
        
        def collate_utk(batch):
            images = [item['image'] for item in batch]
            # Store labels: age, gender_idx, race
            labels = [[item['age'], item['gender_idx'], item['race']] for item in batch]
            return images, labels
            
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_utk
        )
    else:  # imagenet
        from datasets import load_dataset
        print("Streaming ImageNet-1K from Hugging Face...")
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
        total_images = args.total_images
        
        def collate_imagenet(batch):
            images = []
            labels = []
            for item in batch:
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                labels.append(item['label'])
            return images, labels
            
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_imagenet
        )

    # 2. Setup hook infrastructure
    layers_to_capture = {
        'layer_11': 11,
        'layer_17': 17,
        'layer_22': 22,
        'layer_23': 23
    }
    
    feature_dim = 1024
    embedding_feature_dims = 1024
    
    # Pre-allocate spaces
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    f = h5py.File(args.output_file, 'a')
    
    h5_datasets = {}
    
    # Determine the output length based on whether we save patches or CLS tokens
    mult = args.patches_per_image if args.patches_per_image > 0 else 1
    for name in layers_to_capture.keys():
        h5_datasets[name] = f.require_dataset(
            name, 
            (total_images * mult, feature_dim), 
            dtype='float16'
        )
        
    h5_datasets['last_embedding'] = f.require_dataset(
        "last_embedding", 
        (total_images, embedding_feature_dims), 
        dtype='float16'
    )
    
    if args.dataset_type == "utkface":
        # Labels are (age, gender_idx, race) -> [total_images, 3]
        h5_datasets['labels'] = f.require_dataset(
            "labels",
            (total_images, 3),
            dtype='int32'
        )
        
    temp_buffers = {name: [] for name in h5_datasets.keys()}
    hooks = []

    def create_activation_hook(layer_name):
        def hook(module, input, output):
            hidden_states = output[0]  # Shape: [batch_size, seq_len, hidden_dim]
            
            if args.patches_per_image > 0:
                # Strip off CLS token (index 0) and sample random patch tokens
                patch_tokens = hidden_states[:, 1:, :]
                batch_size, num_patches, hidden_dim = patch_tokens.shape
                
                sampled_patches = []
                for b in range(batch_size):
                    random_indices = torch.randperm(num_patches)[:args.patches_per_image]
                    sampled_patches.append(patch_tokens[b, random_indices, :])
                    
                sampled_activations = torch.cat(sampled_patches, dim=0).detach().cpu().numpy()
                temp_buffers[layer_name].append(sampled_activations)
            else:
                # Capture CLS token only
                cls_activation = hidden_states[:, 0, :].detach().cpu().numpy()
                temp_buffers[layer_name].append(cls_activation)
        return hook

    def last_embedding_hook(module, input, output):
        cls_embedding = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
        temp_buffers['last_embedding'].append(cls_embedding)

    # Register hooks on layers
    encoder_layers = model.model.vision_tower.vision_model.encoder.layers
    for name, idx in layers_to_capture.items():
        layer = encoder_layers[idx]
        hooks.append(layer.register_forward_hook(create_activation_hook(name)))
        
    hooks.append(model.model.vision_tower.register_forward_hook(last_embedding_hook))
    
    # 3. Execution loop
    model.eval()
    img_idx = 0
    patch_idx = 0
    
    # Enable JIT compilation for a small speedup
    model.model.vision_tower = torch.compile(model.model.vision_tower)
    
    print("Extracting activations...")
    with torch.no_grad():
        with tqdm(initial=0, total=total_images, desc="Images") as pbar:
            for images, labels in data_loader:
                prompt = "USER: <image>\nWhat is in the image? ASSISTANT:"
                texts = [prompt] * len(images)
                inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device=model.device, dtype=torch.float16)
                
                # Forward pass through the vision tower to trigger hooks
                _ = model.model.vision_tower(pixel_values)
                
                num_images_in_batch = len(images)
                num_patches_in_batch = num_images_in_batch * mult
                
                # Retrieve data from hook buffers and write to HDF5
                for name in layers_to_capture.keys():
                    batch_data = temp_buffers[name].pop(0)
                    h5_datasets[name][patch_idx : patch_idx + num_patches_in_batch] = batch_data
                    
                emb_data = temp_buffers['last_embedding'].pop(0)
                h5_datasets['last_embedding'][img_idx : img_idx + num_images_in_batch] = emb_data
                
                if args.dataset_type == "utkface":
                    h5_datasets['labels'][img_idx : img_idx + num_images_in_batch] = labels
                    
                img_idx += num_images_in_batch
                patch_idx += num_patches_in_batch
                
                pbar.update(num_images_in_batch)
                
                if img_idx % (256 * 4) == 0:
                    f.flush()
                    
                if img_idx >= total_images:
                    break
                    
    # Clean up
    for h in hooks:
        h.remove()
    f.close()
    
    print(f"Finished! Activations successfully saved to {args.output_file}")


if __name__ == "__main__":
    main()
