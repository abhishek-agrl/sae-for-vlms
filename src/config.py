import torch

def get_default_cfg():
    default_cfg = {
        "seed": 17,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e9),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "layer": 8,
        "act_size": 3584,
        "dict_size": 16 * 3584,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_batch_size": 10,
        "num_batches_in_buffer": 2,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,

        # (Batch)TopKSAE specific
        "top_k": 16,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        # for jumprelu
        "bandwidth": 0.001,
    }
    return default_cfg

def get_topk_config():
    topk_config = get_default_cfg()
    topk_config['bandwidth'] = 0.001
    topk_config['input_unit_norm'] = True
    return topk_config
