import torch

# Enable Tensor Cores for Ampere (A100/A6000)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from transformers import AutoProcessor, AutoModelForImageTextToText
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import h5py
import os
import glob
from PIL import Image
import numpy as np

# --- 1. Configuration ---
model_id = "llava-hf/llava-1.5-7b-hf"
output_file = "data/llava_utkface_patch_multiple_layers_train.h5"
utkface_dir = "/home/abhishek.agrawal/utkface_split/train/"
feature_dim = 1024
embedding_feature_dims = 1024
batch_size = 256  # Increased for optimized forward pass

layers_to_capture = {
    'layer_11': 11, 
    'layer_17': 17, 
    'layer_22': 22, 
    'layer_23': 23
}

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

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx] # Use the cached labels
        
        return image, label

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

# --- 3. Load Model and Prepare DataLoader ---
# use_fast=True speeds up image preprocessing on CPU
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)

# Initialize dataset with the specific age range
dataset = UTKFaceDataset(root_dir=utkface_dir, min_age=18, max_age=65)
total_images = len(dataset)

if total_images == 0:
    raise ValueError("No images found! Please check your dataset path and age range.")

data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=12,       
    pin_memory=True,     
    collate_fn=collate_fn
)

# --- 4. Setup Hooks and HDF5 Storage ---
f = h5py.File(output_file, 'a') # 'w' since we are creating it fresh for UTKFace

h5_datasets = {}
for name in layers_to_capture.keys():
    h5_datasets[name] = f.require_dataset(name, (total_images*2, feature_dim), dtype='float16')
    # h5_datasets[name] = f.require_dataset(name, (total_images, feature_dim), dtype='float16')

# Adding the 'last_embedding' and 'labels' datasets
h5_datasets['last_embedding'] = f.require_dataset("last_embedding", (total_images, embedding_feature_dims), dtype='float16')
h5_labels = f.require_dataset("labels", (total_images, 3), dtype='int32')

temp_buffers = {name: [] for name in h5_datasets.keys() if name != 'labels'}
hooks = []

# def create_activation_hook(layer_name):
#     """Factory function to create a hook for a specific layer name."""
#     def hook(module, input, output):
#         cls_activation = output[0][:, 0, :].detach().cpu().numpy()
#         temp_buffers[layer_name].append(cls_activation)
#     return hook

def create_activation_hook(layer_name):
    """Factory function to create a hook for a specific layer name."""
    def hook(module, input, output):
        # Transformer layer outputs a tuple; hidden states are at index 0
        # Shape: [batch_size, seq_len, hidden_dim]
        
        # 1. Strip off the CLS token (index 0) to leave only the patch tokens
        patch_tokens = output[0][:, 1:, :] 
        
        batch_size, num_patches, hidden_dim = patch_tokens.shape
        
        # 2. Randomly sample 2 patch tokens per image (Pach's --random_k 2)
        sampled_patches = []
        for b in range(batch_size):
            # Pick 2 random indices from the available patches (e.g., 576)
            random_indices = torch.randperm(num_patches)[:2]
            sampled_patches.append(patch_tokens[b, random_indices, :])
            
        # 3. Stack them into a flat batch of shape [batch_size * 2, hidden_dim]
        sampled_activations = torch.cat(sampled_patches, dim=0).detach().cpu().numpy()
        
        temp_buffers[layer_name].append(sampled_activations)
    return hook


def last_embedding_hook(module, input, output):
    """Hook specifically for the final vision tower output."""
    cls_embedding = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
    temp_buffers['last_embedding'].append(cls_embedding)

# Register the hooks
encoder_layers = model.model.vision_tower.vision_model.encoder.layers
for name, idx in layers_to_capture.items():
    layer = encoder_layers[idx]
    hooks.append(layer.register_forward_hook(create_activation_hook(name)))

hooks.append(model.model.vision_tower.register_forward_hook(last_embedding_hook))

# --- 5. Process Data ---
model.eval()
# current_idx = 0

# Compile the vision tower for massive speedups on Ampere GPUs
model.model.vision_tower = torch.compile(model.model.vision_tower)
img_idx = 0
patch_idx = 0
print(f"Starting forward passes for {total_images} images...")
with torch.no_grad():
    with tqdm(initial=0, total=total_images) as pbar:
        for images, labels in data_loader:
            prompt = "USER: <image>\nWhat is in the image? ASSISTANT:"
            texts = [prompt] * len(images)
            inputs = processor(
                text=texts, 
                images=images, 
                padding=True, 
                return_tensors="pt"
            )
            
            pixel_values = inputs["pixel_values"].to(device=model.device, dtype=torch.float16)
            
            # Forward pass ONLY through the vision tower
            _ = model.model.vision_tower(pixel_values)

            num_images_in_batch = len(images)
            num_patches_in_batch = num_images_in_batch * 2 # Because of patch_size 2

            # --- CRITICAL FIX: Save the demographic labels! ---
            batch_labels = np.array(labels, dtype=np.int32)
            h5_labels[img_idx : img_idx + num_images_in_batch] = batch_labels

            
            for name in h5_datasets.keys():
                batch_data = temp_buffers[name].pop(0) 
                
                if name == 'last_embedding':
                    # Save 1 CLS token per image
                    h5_datasets[name][img_idx : img_idx + num_images_in_batch] = batch_data
                else:
                    # Save 2 patch tokens per image (NO SLICING!)
                    h5_datasets[name][patch_idx : patch_idx + num_patches_in_batch] = batch_data
            # Advance the cursors separately
            img_idx += num_images_in_batch
            patch_idx += num_patches_in_batch
            
            pbar.update(num_images_in_batch)

            # Periodically flush to disk
            if img_idx % (256 * 10) == 0:
                f.flush()

            if img_idx >= total_images:
                break

# with torch.no_grad():
#     with tqdm(initial=0, total=total_images) as pbar:
#         for images, labels in data_loader:
#             prompt = "USER: <image>\nWhat is in the image? ASSISTANT:"
#             texts = [prompt] * len(images)
#             # We only need to process images, skipping the LLM entirely
#             inputs = processor(
#                 images=images, 
#                 padding=True, 
#                 return_tensors="pt",
#                 text=texts,
#             )
            
#             pixel_values = inputs["pixel_values"].to(device=model.device, dtype=torch.float16)
            
#             # Forward pass ONLY through the vision tower
#             _ = model.model.vision_tower(pixel_values)

#             # Retrieve data from buffers and save to HDF5
#             current_batch_size = len(images)
#             batch_labels = np.array(labels, dtype=np.int32)

#             for name in temp_buffers.keys():
#                 batch_data = temp_buffers[name].pop(0) 
#                 h5_datasets[name][current_idx : current_idx + current_batch_size] = batch_data[:current_batch_size]
            
#             # Save the labels
#             h5_labels[current_idx : current_idx + current_batch_size] = batch_labels
            
#             current_idx += current_batch_size
#             pbar.update(current_batch_size)

#             # Periodically flush to disk to prevent data loss
#             if current_idx % (batch_size * 10) == 0:
#                 f.flush()

#             if current_idx >= total_images:
#                 break

# --- 6. Cleanup ---
for h in hooks:
    h.remove()
f.close()

print(f"Finished! Activations for all specified layers and demographic labels saved to {output_file}")