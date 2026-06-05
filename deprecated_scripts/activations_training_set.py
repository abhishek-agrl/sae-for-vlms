import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

model_id = "llava-hf/llava-1.5-7b-hf"
output_file = "llava_multiple_layers.h5"

# ImageNet-1K training set has exactly 1,281,167 images.
# We hardcode this to pre-allocate the HDF5 datasets correctly.
total_train_images = 1281167 
feature_dim = 1024

# --- RESUMPTION CONFIGURATION ---
start_idx = 0 
# --------------------------------

# 1. Load the LLaVa model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)

# 2. Setup HDF5 File and Pre-allocate Datasets for Multiple Layers
f = h5py.File(output_file, 'a')
layers_to_capture = {
    'layer_11': 11, 
    'layer_17': 17, 
    'layer_22': 22, 
    'layer_23': 23
}

# Create a separate HDF5 dataset for each layer
h5_datasets = {}
for name in layers_to_capture.keys():
    h5_datasets[name] = f.create_dataset(name, (total_train_images, feature_dim), dtype='float16')

# Adding the 'Last' embedding layer
h5_datasets['last_embedding'] = f.create_dataset("last_embedding", (total_train_images, feature_dim), dtype='float16')

# 3. Define Hook Factory for Multiple Layers
temp_buffers = {name: [] for name in h5_datasets.keys()}
hooks = []

def create_activation_hook(layer_name):
    """Factory function to create a hook for a specific layer name."""
    def hook(module, input, output):
        # Transformer layer outputs a tuple; hidden states are at index 0
        cls_activation = output[0][:, 0, :].detach().cpu().numpy()
        temp_buffers[layer_name].append(cls_activation)
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

hooks.append(model.vision_tower.register_forward_hook(last_embedding_hook))

# 5. Stream the Dataset
# streaming=True prevents downloading the full dataset to disk
# Stream and SKIP to the resume point
print(f"Fast-forwarding dataset to index {start_idx}. This may take a minute or two...")
dataset = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)
# dataset = dataset.skip(start_idx) # <--- Fast-forwards the stream

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

# When streaming, DataLoader still works for batching, but we can't use `shuffle=True` 
# in the standard way, and `len()` won't work automatically.
data_loader = DataLoader(
    dataset, 
    batch_size=12, 
    num_workers=2, # Keep workers low for streaming to avoid connection throttling
    collate_fn=collate_fn
)

# 6. Process the data
model.eval()
current_idx = start_idx
# current_idx = 0

print("Starting forward passes...")
with torch.no_grad():
    # Use total_train_images for tqdm since data_loader has no length in streaming mode
    with tqdm(initial=start_idx, total=total_train_images) as pbar:
        for images, labels in data_loader:
            prompt = "USER: <image>\nWhat is in the image? ASSISTANT:"
            texts = [prompt] * len(images)
            
            inputs = processor(
                text=texts, 
                images=images, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device).to(torch.float16)

            # Forward pass
            _ = model(**inputs)

            # Retrieve data from buffers and save to HDF5
            batch_size = len(images)
            
            for name in h5_datasets.keys():
                # pop(0) fetches the array populated by the hook for this specific forward pass
                batch_data = temp_buffers[name].pop(0) 
                h5_datasets[name][current_idx : current_idx + batch_size] = batch_data
            
            current_idx += batch_size
            pbar.update(batch_size)

            # Periodically flush to disk to prevent data loss
            if current_idx % (12 * 50) == 0:
                f.flush()

            # Safeguard just in case the stream yields slightly more/less
            if current_idx >= total_train_images:
                break

# 7. Cleanup
for h in hooks:
    h.remove()
f.close()

print(f"Finished! Activations for all specified layers saved to {output_file}")