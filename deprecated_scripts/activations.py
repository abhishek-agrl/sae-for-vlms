import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from transformers import AutoProcessor, AutoModelForImageTextToText
import torchvision.transforms as T
import torchvision.datasets as datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import os

# print(f"Is CUDA available: {torch.cuda.is_available()}")
# print(f"Device count: {torch.cuda.device_count()}")

model_id = "llava-hf/llava-1.5-7b-hf"
output_file = "data/llava_imagenet_patch_validation_multiple_layers.h5"
total_images = 50000
# output_file = "llava_patch_multiple_layers.h5"
# total_images = 1281167
feature_dim = 1024
embedding_feature_dims = 1024

# 1. Load the LLaVa model and processor
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map = "auto",
)

# 1.1 Check if the model is ok and what layer to target.
# print(model)

# We pre-allocate the space [50000, 1024]
f = h5py.File(output_file, 'a')
layers_to_capture = {
    'layer_11': 11, 
    'layer_17': 17, 
    'layer_22': 22, 
    'layer_23': 23
}

h5_datasets = {}
for name in layers_to_capture.keys():
    h5_datasets[name] = f.require_dataset(name, (total_images*2, feature_dim), dtype='float16') # Only for patch activations
    # h5_datasets[name] = f.require_dataset(name, (total_images, feature_dim), dtype='float16')

# Adding the 'Last' embedding layer
h5_datasets['last_embedding'] = f.require_dataset("last_embedding", (total_images, embedding_feature_dims), dtype='float16')

# 3. Define Hook Factory for Multiple Layers
temp_buffers = {name: [] for name in h5_datasets.keys()}
hooks = []

# def create_activation_hook(layer_name):
#     """Factory function to create a hook for a specific layer name."""
#     def hook(module, input, output):
#         # Transformer layer outputs a tuple; hidden states are at index 0
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

# 4. Register the hooks
encoder_layers = model.model.vision_tower.vision_model.encoder.layers
for name, idx in layers_to_capture.items():
    layer = encoder_layers[idx]
    hooks.append(layer.register_forward_hook(create_activation_hook(name)))

hooks.append(model.model.vision_tower.register_forward_hook(last_embedding_hook))


# Working on the validation dataset.
dataset = load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
# dataset = torch.utils.data.Subset(dataset, range(100))  # Test

# Create DataLoader for the subset
def collate_fn(batch):
    images = []
    labels = []
    for item in batch:
        img = item['image']
        # ImageNet contains some grayscale images which will crash the processor.
        # We must ensure they are converted to RGB.
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
        labels.append(item['label'])
    return images, labels


data_loader = DataLoader(
    dataset, 
    batch_size=256,
    num_workers=4,      # Use multiple CPU cores to load/resize images
    pin_memory=True,     # Speeds up transfer from CPU to GPU,
    collate_fn=collate_fn
)

# 5. Process the data and collect activations
model.eval()
# current_idx = 0
img_idx = 0
patch_idx = 0

model.model.vision_tower = torch.compile(model.model.vision_tower)

print("Starting forward passes...")
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

            # Retrieve data from buffers and save to HDF5
            # batch_size = len(images)

            # for name in h5_datasets.keys():
            #     # pop(0) fetches the array populated by the hook for this specific forward pass
            #     batch_data = temp_buffers[name].pop(0) 
            #     h5_datasets[name][current_idx : current_idx + batch_size] = batch_data[:batch_size]
            
            # current_idx += batch_size
            # pbar.update(batch_size)

            # # Periodically flush to disk to prevent data loss
            # if current_idx % (128 * 10) == 0:
            #     f.flush()

            # # Safeguard just in case the stream yields slightly more/less
            # if current_idx >= total_images:
            #     break

            num_images_in_batch = len(images)
            num_patches_in_batch = num_images_in_batch * 2 # Because of patch_size 2
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


# print("Starting forward passes...")
# with torch.no_grad():
#     # Use total_train_images for tqdm since data_loader has no length in streaming mode
#     with tqdm(initial=start_idx, total=total_train_images) as pbar:
#         for images, labels in data_loader:
#             prompt = "USER: <image>\nWhat is in the image? ASSISTANT:"
#             texts = [prompt] * len(images)
            
#             inputs = processor(
#                 text=texts, 
#                 images=images, 
#                 padding=True, 
#                 return_tensors="pt"
#             ).to(model.device).to(torch.float16)

#             # Forward pass
#             _ = model(**inputs)

#             # Retrieve data from buffers and save to HDF5
#             batch_size = len(images)
            
#             for name in h5_datasets.keys():
#                 # pop(0) fetches the array populated by the hook for this specific forward pass
#                 batch_data = temp_buffers[name].pop(0) 
#                 h5_datasets[name][current_idx : current_idx + batch_size] = batch_data
            
#             current_idx += batch_size
#             pbar.update(batch_size)

#             # Periodically flush to disk to prevent data loss
#             if current_idx % (12 * 50) == 0:
#                 f.flush()

#             # Safeguard just in case the stream yields slightly more/less
#             if current_idx >= total_train_images:
#                 break


# # 6. Save the activations to a file
# if cls_activations:
#     # Flatten and concatenate activations across the batch
#     # activations_tensor = torch.cat([a.unsqueeze(0) for a in cls_activations], dim=0)
#     activations_tensor = torch.cat(cls_activations, dim=0)
#     print(f"Shape of captured activations: {activations_tensor.shape}")

#     # Save activations as a .pth file
#     torch.save(activations_tensor, './llava_activations.pth')

# Cleanup
for h in hooks:
    h.remove()
f.close()

print(f"Finished! Activations for all specified layers saved to {output_file}")
