import torch
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
output_file = "llava_utkface_features_test_18_to_65.h5"
# utkface_dir = "/home/abhishek.agrawal/utkface_aligned_cropped/crop_part1/"
utkface_dir = "/home/abhishek.agrawal/utkface_split/test/"
cls_feature_dim = 1024
embedding_feature_dims = 1024
batch_size = 12

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
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)

# Initialize dataset with the specific age range
dataset = UTKFaceDataset(root_dir=utkface_dir, min_age=18, max_age=65)
num_images = len(dataset)

# Fail safe if no images match the criteria or path is wrong
if num_images == 0:
    raise ValueError("No images found! Please check your dataset path and age range.")

data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=8,       
    pin_memory=True,     
    collate_fn=collate_fn
)

# --- 4. Setup Hooks and HDF5 Storage ---
captured_features = {'activations': None, 'embeddings': None}

def activation_hook(module, input, output):
    """Capture layer activations (e.g., from layer 17)."""
    cls_activation = output[0][:, 0, :].detach().cpu().numpy()
    captured_features['activations'] = cls_activation

def embedding_hook(module, input, output):
    """Capture the final embeddings from the vision tower."""
    cls_embedding = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
    captured_features['embeddings'] = cls_embedding

# target_layer = model.vision_tower.vision_model.encoder.layers
target_layer = model.model.vision_tower.vision_model.encoder.layers[17]
embedding_layer = model.vision_tower

act_hook = target_layer.register_forward_hook(activation_hook)
emb_hook = embedding_layer.register_forward_hook(embedding_hook)

# Pre-allocate HDF5 datasets using the new dynamically filtered length
f = h5py.File(output_file, 'w')
dset_acts = f.create_dataset("activations", (num_images, cls_feature_dim), dtype='float16')
dset_embs = f.create_dataset("embeddings", (num_images, embedding_feature_dims), dtype='float16')
dset_labels = f.create_dataset("labels", (num_images, 3), dtype='int32')

# --- 5. Process Data ---
model.eval()
current_idx = 0

print(f"Extracting features for {num_images} images...")
with torch.no_grad():
    for images, labels in tqdm(data_loader, total=len(data_loader)):
        prompt = "USER: <image>\nWhat is in the image? ASSISTANT:"
        
        # 1. DEBUG: If this prints <class 'int'>, your DataLoader is yielding labels/indices instead of images!
        # if not isinstance(images, Image.Image):
        #     print(f"\n[DEBUG] Expected PIL Image, but got: {type(images)}")
        #     print(f"[DEBUG] Value: {images}")
        #     break # Stop the loop so you can investigate
            
        # 2. Format specifically for LLaVA's batched processing
        # LLaVA expects a list of lists when processing a batch of texts
        # batched_images = [[img] for img in images]
        
        texts = [prompt] * len(images)
        
        inputs = processor(
            text=texts, 
            images=images, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device).to(torch.float16)

        # Forward pass (triggers the hooks)
        _ = model(**inputs)

        # Retrieve features
        batch_acts = captured_features['activations']
        batch_embs = captured_features['embeddings']
        batch_labels = np.array(labels, dtype=np.int32)
        
        num_in_batch = batch_embs.shape[0]
        
        # Save to HDF5
        dset_acts[current_idx : current_idx + num_in_batch] = batch_acts
        dset_embs[current_idx : current_idx + num_in_batch] = batch_embs
        dset_labels[current_idx : current_idx + num_in_batch] = batch_labels
        
        current_idx += num_in_batch

        if current_idx % (batch_size * 10) == 0:
            f.flush()

# --- 6. Cleanup ---
act_hook.remove()
emb_hook.remove()
f.close()

print(f"Finished! Activations, Embeddings, and Labels saved to {output_file}")