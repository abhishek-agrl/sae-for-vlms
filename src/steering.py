import torch
import math

def sae_encode(sae, x):
    """Encodes input activations through the given SAE model (custom or library-based)."""
    if hasattr(sae, "encode"):
        return sae.encode(x), None
    else:
        # Custom SAE model encoder path
        x_prep, x_mean, x_std = sae.preprocess_input(x)
        x_cent = x_prep - sae.b_dec
        acts = torch.nn.functional.relu(x_cent @ sae.W_enc)
        
        # Apply Top-K if top_k configuration is present
        if hasattr(sae, "cfg") and sae.cfg.get("top_k") is not None:
            top_k_val = sae.cfg["top_k"]
            # Check if BatchTopKSAE
            if sae.cfg.get("sae_type") == "batch_topk":
                acts_flat = torch.topk(acts.flatten(), top_k_val * x.shape[0], dim=-1)
                acts = (
                    torch.zeros_like(acts.flatten())
                    .scatter(-1, acts_flat.indices, acts_flat.values)
                    .reshape(acts.shape)
                )
            else:
                # TopKSAE
                acts_topk = torch.topk(acts, top_k_val, dim=-1)
                acts = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        return acts, (x_mean, x_std)


def sae_decode(sae, acts, cache=None):
    """Decodes latent features back to activation space."""
    if hasattr(sae, "decode"):
        return sae.decode(acts)
    else:
        # Custom SAE model decoder path
        x_reconstruct = acts @ sae.W_dec + sae.b_dec
        if cache is not None:
            x_mean, x_std = cache
            x_reconstruct = sae.postprocess_output(x_reconstruct, x_mean, x_std)
        return x_reconstruct


class SAEHookSteerer:
    """
    A unified steering hook manager that can hook into multiple layers of a model
    and clamp specific SAE features to target values.
    Supports targeting the CLS token, patch tokens, or all tokens.
    """
    def __init__(self, sae_dict, intervention_dict, target_token="patches", device="cpu"):
        """
        sae_dict: dict mapping layer identifier (str) to sae_model (e.g., {"layer_17": sae})
        intervention_dict: dict mapping (layer_id, neuron_idx) -> clamping_strength (float)
        target_token: token indices to intervene on. Options: 'cls' (index 0), 'patches' (index 1:), or 'all'
        """
        self.sae_dict = sae_dict
        self.target_token = target_token
        self.device = device
        self.handles = []
        
        # Group interventions by layer
        self.layer_interventions = {}
        for (layer_id, neuron_idx), strength in intervention_dict.items():
            if layer_id not in self.sae_dict:
                raise ValueError(f"Missing SAE configuration for layer '{layer_id}' in sae_dict.")
            
            if layer_id not in self.layer_interventions:
                self.layer_interventions[layer_id] = {'neurons': [], 'strengths': []}
                
            self.layer_interventions[layer_id]['neurons'].append(neuron_idx)
            self.layer_interventions[layer_id]['strengths'].append(strength)
            
        # Pre-compute tensors for optimized vectorized clamping
        for layer_id in self.layer_interventions:
            self.layer_interventions[layer_id]['neurons'] = torch.tensor(
                self.layer_interventions[layer_id]['neurons'], dtype=torch.long, device=device
            )
            self.layer_interventions[layer_id]['strengths'] = torch.tensor(
                self.layer_interventions[layer_id]['strengths'], dtype=torch.float32, device=device
            )

    def _create_hook_for_layer(self, layer_id):
        sae = self.sae_dict[layer_id]
        neuron_indices = self.layer_interventions[layer_id]['neurons']
        clamping_values = self.layer_interventions[layer_id]['strengths']

        def hook_fn(module, input, output):
            with torch.no_grad():
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                steered_states = hidden_states.clone()
                
                # Determine target slice
                if self.target_token == "cls":
                    target_slice = steered_states[:, 0:1, :].to(torch.float32)
                elif self.target_token == "patches":
                    target_slice = steered_states[:, 1:, :].to(torch.float32)
                else:  # 'all'
                    target_slice = steered_states.to(torch.float32)
                
                # Run through SAE
                latents, cache = sae_encode(sae, target_slice)
                
                # Intervene/Clamp features
                latents[..., neuron_indices] = clamping_values.to(latents.dtype)
                
                # Reconstruct
                reconstructed = sae_decode(sae, latents, cache)
                
                # Re-inject steered slice
                if self.target_token == "cls":
                    steered_states[:, 0:1, :] = reconstructed.to(hidden_states.dtype)
                elif self.target_token == "patches":
                    steered_states[:, 1:, :] = reconstructed.to(hidden_states.dtype)
                else:
                    steered_states = reconstructed.to(hidden_states.dtype)
                    
            return (steered_states,) + output[1:] if is_tuple else steered_states
            
        return hook_fn

    def register_hooks(self, layers_container):
        """
        Attaches hooks to the vision encoder layers.
        layers_container: List/Sequential container of layer modules (e.g. model.vision_tower.vision_model.encoder.layers)
        """
        for layer_id in self.layer_interventions.keys():
            # Parse layer index from key names (e.g., 'layer_17' or 'topk_sae_17')
            parts = layer_id.split("_")
            try:
                layer_idx = int(parts[-1])
            except ValueError:
                raise ValueError(f"Could not parse layer index from layer ID: {layer_id}. Expected format ending in _[index].")
                
            target_layer = layers_container[layer_idx]
            hook = self._create_hook_for_layer(layer_id)
            handle = target_layer.register_forward_hook(hook)
            self.handles.append(handle)
            
            num_neurons = len(self.layer_interventions[layer_id]['neurons'])
            print(f"[+] Hook attached to layer {layer_id} (index {layer_idx}) -> Clamping {num_neurons} neurons.")

    def remove_hooks(self):
        """Removes all attached hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("[-] All steering hooks removed.")
