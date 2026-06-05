import os
import glob
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image

def parse_utk_metadata(filename):
    """
    Extracts age, gender, and race from UTKFace filenames.
    Format is typically: [age]_[gender]_[race]_[timestamp].jpg
    Gender: 0 (Male), 1 (Female)
    """
    try:
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])
        gender_str = "Male" if gender == 0 else "Female"
        race = int(parts[2]) if len(parts) > 2 else -1
        return age, gender_str, race
    except (IndexError, ValueError):
        return None, None, None


class UTKFaceDataset(Dataset):
    """
    Dataset to load UTKFace images and extract labels directly from their filenames.
    Filters images based on age range.
    """
    def __init__(self, root_dir, min_age=18, max_age=65, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Scan for JPG, PNG, and JPEG
        all_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            all_files.extend(glob.glob(os.path.join(root_dir, ext)))
        
        all_image_paths = sorted(all_files)
        self.image_paths = []
        self.labels = []  # List of tuples: (age, gender_str, gender_idx, race)
        
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            age, gender_str, race = parse_utk_metadata(filename)
            
            if age is not None and min_age <= age <= max_age:
                gender_idx = 1 if gender_str == "Female" else 0
                self.image_paths.append(img_path)
                self.labels.append((age, gender_str, gender_idx, race))
                
        print(f"UTKFace Dataset: Found {len(self.image_paths)} images matching age {min_age}-{max_age}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        age, gender_str, gender_idx, race = self.labels[idx]
        
        return {
            "image": image,
            "path": img_path,
            "filename": os.path.basename(img_path),
            "age": age,
            "gender": gender_str,
            "gender_idx": gender_idx,
            "race": race
        }


class ActivationDataset(Dataset):
    """
    A unified dataset for loading model activations from HDF5 files.
    Supports eager loading (into RAM) and lazy loading (on-the-fly).
    """
    def __init__(self, h5_path, dataset_key, lazy=False, dtype=torch.float32):
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self.lazy = lazy
        self.dtype = dtype
        self.h5_file = None
        
        # Verify dataset and check length
        with h5py.File(self.h5_path, 'r') as f:
            if dataset_key not in f:
                available_keys = list(f.keys())
                raise ValueError(f"Key '{dataset_key}' not found in HDF5. Available keys: {available_keys}")
            self.length = f[dataset_key].shape[0]
            
            if not self.lazy:
                print(f"Eagerly loading '{dataset_key}' activations from HDF5 into RAM...")
                # Load fully into CPU RAM
                np_data = f[dataset_key][:]
                self.data = torch.from_numpy(np_data).to(self.dtype)
                print(f"Successfully loaded {self.data.shape} activations.")
            else:
                print(f"Lazy loading initialized for '{dataset_key}' ({self.length} items).")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.lazy:
            return self.data[idx]
            
        # Lazy loading logic to avoid multi-processing issues
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        np_data = self.h5_file[self.dataset_key][idx]
        return torch.from_numpy(np_data).to(self.dtype)


def align_features_and_labels(activations, targets):
    """
    Aligns activations and targets (labels/embeddings) when there is a mismatch
    due to token extraction (e.g. 2 patches/tokens per image vs 1 target per image).
    """
    if activations.size(0) == targets.size(0) * 2:
        print(f"Mismatch detected: {activations.size(0)} activations vs {targets.size(0)} targets.")
        print("Repeating targets by 2x to align with token/patch activations...")
        if isinstance(targets, torch.Tensor):
            return torch.repeat_interleave(targets, repeats=2, dim=0)
        elif isinstance(targets, list):
            return [x for x in targets for _ in (0, 1)]
    elif activations.size(0) != targets.size(0):
        raise ValueError(
            f"Shape mismatch cannot be resolved: activations {activations.size(0)} vs targets {targets.size(0)}"
        )
    return targets
