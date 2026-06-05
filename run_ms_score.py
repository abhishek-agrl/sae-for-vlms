import os
import sys
import argparse
import h5py
import torch
import pandas as pd
from tqdm import tqdm

from src.config import get_topk_config
from src.models import load_sae
from src.data import align_features_and_labels, UTKFaceDataset
from src.metrics import compute_ms_scores, compute_signed_mutual_information


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAE features using Mono-Semanticity Score (MS-Score) or Signed Mutual Information (MI).")
    parser.add_argument("--metric", type=str, choices=["ms_score", "signed_mi"], required=True, help="Metric to calculate: ms_score or signed_mi")
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 activations file")
    parser.add_argument("--layer", type=str, required=True, help="Layer key in HDF5 file (e.g. layer_17)")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE weights checkpoint file")
    
    # SAE config override
    parser.add_argument("--sae_type", type=str, choices=["topk", "batch_topk", "vanilla", "jumprelu"], default="topk", help="SAE type")
    parser.add_argument("--dict_size", type=int, default=1024 * 64, help="SAE dict size")
    parser.add_argument("--top_k", type=int, default=20, help="SAE Top-K value")
    
    # Metric specific overrides
    parser.add_argument("--tau", type=float, default=0.7, help="Consistency threshold for Signed-MI")
    parser.add_argument("--act_threshold", type=float, default=0.05, help="Activation binarization threshold for Signed-MI")
    parser.add_argument("--top_n", type=int, default=10, help="Print top N scoring features")
    parser.add_argument("--utk_image_dir", type=str, default="", help="Optional local UTKFace images path to lookup top activating images")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup config
    cfg = get_topk_config()
    cfg['act_size'] = 1024
    cfg['dict_size'] = args.dict_size
    cfg['top_k'] = args.top_k
    cfg['device'] = device
    
    # 1. Load SAE Model
    model = load_sae(args.sae_path, device, cfg=cfg, sae_type=args.sae_type)
    
    # 2. Load Data from HDF5
    print(f"Loading {args.layer} activations from {args.data_path}...")
    with h5py.File(args.data_path, 'r') as f:
        activations = torch.from_numpy(f[args.layer][:]).float()
        
        # Load embeddings (supporting last_embedding or embeddings names)
        emb_key = 'last_embedding' if 'last_embedding' in f else 'embeddings'
        if emb_key in f:
            embeddings = torch.from_numpy(f[emb_key][:]).float()
        else:
            embeddings = None
            
        # Load labels if available
        if 'labels' in f:
            labels = torch.from_numpy(f['labels'][:]).long()
        else:
            labels = None
            
    N = activations.size(0)
    print(f"Loaded {N} activations of dim {activations.shape[1]}.")
    
    # 3. Compute SAE latents/features in batches
    print("Computing SAE feature activations in batches...")
    feature_acts_list = []
    batch_size = 4096
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="SAE Encode"):
            batch = activations[i:i+batch_size].to(device)
            # Support both library models (encode) and custom models (forward/encode helper)
            if hasattr(model, "encode"):
                latents = model.encode(batch)
            else:
                latents = model(batch)["feature_acts"]
            feature_acts_list.append(latents.cpu())
            
    feature_acts = torch.cat(feature_acts_list, dim=0)  # [N, dict_size]
    
    # 4. Perform computations
    if args.metric == "ms_score":
        if embeddings is None:
            raise KeyError("HDF5 file must contain 'last_embedding' or 'embeddings' to compute MS-Score.")
            
        # Align embeddings if there is a patch activation mismatch
        embeddings_aligned = align_features_and_labels(activations, embeddings)
        
        ms_scores = compute_ms_scores(
            feature_acts=feature_acts,
            embeddings=embeddings_aligned,
            device=device,
            chunk_size=512,
            chunk_size_n=5000
        )
        
        # Output summary
        is_nan = torch.isnan(ms_scores)
        valid_ms_scores = ms_scores[~is_nan]
        valid_indices = torch.nonzero(~is_nan).squeeze()
        
        print(f"\n--- MS-Score Results Overview ---")
        print(f"Total Features: {ms_scores.shape[0]}")
        print(f"Dead/Inactive Features: {is_nan.sum().item()}")
        
        if len(valid_ms_scores) > 0:
            print(f"Mean MS-Score: {valid_ms_scores.mean().item():.4f} +- {valid_ms_scores.std().item():.4f}")
            print(f"Max MS-Score:  {valid_ms_scores.max().item():.4f}")
            
            top_values, top_indices = torch.topk(valid_ms_scores, min(args.top_n, len(valid_ms_scores)))
            original_indices = valid_indices[top_indices]
            
            print(f"\nTop {args.top_n} Most Monosemantic SAE Features:")
            for idx, val in zip(original_indices, top_values):
                print(f"Feature {idx.item():<6} | MS-Score: {val.item():.4f}")
                
            # If UTKFace image dir is provided, print top activating details
            if args.utk_image_dir and labels is not None:
                race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
                print(f"\nLooking up top activating images in {args.utk_image_dir}...")
                
                # Check if labels need expansion
                labels_aligned = align_features_and_labels(activations, labels)
                
                for idx in original_indices:
                    idx_val = idx.item()
                    print(f"\nFeature {idx_val} details (top 5 images):")
                    
                    feat_acts = feature_acts[:, idx_val]
                    top_acts, top_img_indices = torch.topk(feat_acts, 5)
                    
                    for i in range(5):
                        img_idx = top_img_indices[i].item()
                        act_val = top_acts[i].item()
                        
                        age, gender, race = labels_aligned[img_idx]
                        gender_str = "Female" if gender == 1 else "Male"
                        race_str = race_map.get(race.item(), "Unknown")
                        
                        print(f"  Act: {act_val:<8.4f} | Age: {age:<2} | Gender: {gender_str:<6} | Race: {race_str}")
                        
    elif args.metric == "signed_mi":
        if labels is None:
            raise ValueError("HDF5 file must contain 'labels' dataset (containing Gender) to compute Signed-MI.")
            
        # Extract Gender column (typically column index 1 in labels)
        genders = labels[:, 1]
        
        # Align labels if patch mismatch
        genders_aligned = align_features_and_labels(activations, genders)
        
        print(f"Computing Signed Mutual Information (tau={args.tau}, act_threshold={args.act_threshold})...")
        mi_dict = compute_signed_mutual_information(
            feature_acts=feature_acts,
            genders=genders_aligned,
            tau=args.tau,
            act_threshold=args.act_threshold,
            device=device
        )
        
        signed_mi = mi_dict["signed_mi"]
        valid_female = mi_dict["valid_female_mask"]
        valid_male = mi_dict["valid_male_mask"]
        
        print(f"\n--- Signed-MI Results Overview ---")
        print(f"Features passing AC filter (tau={args.tau}):")
        print(f"  - Female Consistent: {valid_female.sum().item()}")
        print(f"  - Male Consistent:   {valid_male.sum().item()}")
        
        # Female associated features (largest positive Signed-MI)
        filtered_mi_female = torch.where(valid_female, signed_mi, torch.tensor(float('-inf')))
        # Male associated features (most negative Signed-MI)
        filtered_mi_male = torch.where(valid_male, signed_mi, torch.tensor(float('inf')))
        
        def print_top_mi(scores, means_tgt, means_oth, counts_tgt, counts_oth, title, largest=True):
            vals, idxs = torch.topk(scores, args.top_n, largest=largest)
            print(f"\nTop {args.top_n} Signed-MI {title} Features:")
            print(f"{'Index':<8} | {'Signed-MI':<10} | {'Avg Tgt Act':<12} | {'Tgt Fires':<10} | {'Avg Oth Act':<12} | {'Oth Fires':<10}")
            print("-" * 80)
            
            for i in range(args.top_n):
                idx = idxs[i].item()
                val = vals[i].item()
                if float('inf') in [abs(val)]:
                    print("  [No more features passed the filter]")
                    break
                    
                act_t = means_tgt[idx].item()
                act_o = means_oth[idx].item()
                c_t = int(counts_tgt[idx].item())
                c_o = int(counts_oth[idx].item())
                print(f"{idx:<8} | {val:<10.4f} | {act_t:<12.4f} | {c_t:<10d} | {act_o:<12.4f} | {c_o:<10d}")
                
        # Largest positive values = Female
        print_top_mi(
            filtered_mi_female, mi_dict["mean_act_female"], mi_dict["mean_act_male"],
            mi_dict["count_female"], mi_dict["count_male"], "Female", largest=True
        )
        
        # Most negative values = Male
        print_top_mi(
            filtered_mi_male, mi_dict["mean_act_male"], mi_dict["mean_act_female"],
            mi_dict["count_male"], mi_dict["count_female"], "Male", largest=False
        )


if __name__ == "__main__":
    main()
