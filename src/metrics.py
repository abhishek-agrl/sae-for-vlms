import torch
from tqdm import tqdm

def compute_signed_mutual_information(feature_acts, genders, tau=0.7, act_threshold=0.05, device="cpu"):
    """
    Computes the Signed Mutual Information (Signed-MI) and Activation Consistency (AC) filter
    between SAE feature activations and gender labels (0 for Male, 1 for Female).
    """
    feature_acts_dev = feature_acts.to(device)
    genders_dev = genders.to(device)
    
    # Binarize activations
    F_active = (feature_acts_dev > act_threshold).float()
    
    Y_female = (genders_dev == 1).float().unsqueeze(1)
    Y_male = (genders_dev == 0).float().unsqueeze(1)
    
    # Activation Consistency (AC) Filter
    freq_female = F_active[genders_dev == 1].mean(dim=0)
    freq_male = F_active[genders_dev == 0].mean(dim=0)
    
    count_female = F_active[genders_dev == 1].sum(dim=0)
    count_male = F_active[genders_dev == 0].sum(dim=0)
    
    consistent_female = freq_female >= tau
    consistent_male = freq_male >= tau
    
    consistent_both = consistent_female & consistent_male
    valid_female_mask = consistent_female & ~consistent_both
    valid_male_mask = consistent_male & ~consistent_both
    
    # Calculate Marginals
    p_y1 = Y_female.mean() 
    p_y0 = Y_male.mean()   
    p_f1 = F_active.mean(dim=0)
    p_f0 = 1.0 - p_f1
    
    # Calculate Joint Probabilities
    p_f1_y1 = (F_active * Y_female).mean(dim=0)
    p_f1_y0 = (F_active * Y_male).mean(dim=0)
    p_f0_y1 = ((1 - F_active) * Y_female).mean(dim=0)
    p_f0_y0 = ((1 - F_active) * Y_male).mean(dim=0)
    
    # Helper for MI term
    def calc_mi_term(p_joint, p_marginal_f, p_marginal_y):
        mask = p_joint > 0
        term = torch.zeros_like(p_joint)
        denominator = p_marginal_f * p_marginal_y + 1e-9
        term[mask] = p_joint[mask] * torch.log2(p_joint[mask] / denominator[mask])
        return term
        
    mi = (
        calc_mi_term(p_f1_y1, p_f1, p_y1) +
        calc_mi_term(p_f1_y0, p_f1, p_y0) +
        calc_mi_term(p_f0_y1, p_f0, p_y1) +
        calc_mi_term(p_f0_y0, p_f0, p_y0)
    )
    
    # Calculate sign using difference in mean activations
    mean_act_female = feature_acts_dev[genders_dev == 1].mean(dim=0)
    mean_act_male = feature_acts_dev[genders_dev == 0].mean(dim=0)
    
    delta_a = mean_act_female - mean_act_male
    signed_mi = mi * torch.sign(delta_a)
    
    return {
        "signed_mi": signed_mi.cpu(),
        "valid_female_mask": valid_female_mask.cpu(),
        "valid_male_mask": valid_male_mask.cpu(),
        "mean_act_female": mean_act_female.cpu(),
        "mean_act_male": mean_act_male.cpu(),
        "count_female": count_female.cpu(),
        "count_male": count_male.cpu()
    }


def compute_ms_scores(feature_acts, embeddings, device="cpu", chunk_size=512, chunk_size_n=5000):
    """
    Computes Mono-Semanticity Scores (MS-Scores) for SAE latent features
    using a batch-vectorized dot product implementation on the GPU.
    """
    N, num_neurons = feature_acts.shape
    
    # Min-Max Normalize activations per neuron
    print("Normalizing features...")
    a_min = feature_acts.min(dim=0, keepdim=True).values
    a_max = feature_acts.max(dim=0, keepdim=True).values
    denom = a_max - a_min
    denom[denom == 0] = 1.0
    a_tilde = (feature_acts - a_min) / denom
    
    # L2-normalize embeddings for cosine similarity
    print("Normalizing embeddings for cosine similarity...")
    embeddings_norm = torch.nn.functional.normalize(embeddings.to(device), p=2, dim=1)
    
    ms_scores = torch.zeros(num_neurons)
    
    # Loop over neurons in chunks to prevent memory overflow
    for i in tqdm(range(0, num_neurons, chunk_size), desc="Computing MS-Score"):
        a_chunk = a_tilde[:, i:i+chunk_size].to(device)  # [N, chunk_size]
        sum_all = torch.zeros(a_chunk.shape[1], device=device)
        
        # Process similarity matrix in blocks on-the-fly
        for j in range(0, N, chunk_size_n):
            E_chunk = embeddings_norm[j:j+chunk_size_n]
            S_chunk = torch.matmul(E_chunk, embeddings_norm.T)  # [chunk_size_n, N]
            S_a_chunk = torch.matmul(S_chunk, a_chunk)          # [chunk_size_n, chunk_size]
            sum_all += torch.sum(a_chunk[j:j+chunk_size_n] * S_a_chunk, dim=0)
            
        sum_diag = torch.sum((a_chunk ** 2), dim=0)
        numerator = sum_all - sum_diag
        
        sum_a = torch.sum(a_chunk, dim=0)
        sum_sq_a = torch.sum(a_chunk ** 2, dim=0)
        weight_sum = (sum_a ** 2) - sum_sq_a
        
        ms_k = torch.where(
            weight_sum != 0, 
            numerator / weight_sum, 
            torch.tensor(float('nan'), device=device)
        )
        ms_scores[i:i+chunk_size] = ms_k.cpu()
        
    return ms_scores
