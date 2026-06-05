import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Import the exact SAE architecture from the dictionary_learning library
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

# ==========================================
# 1. Core Models & Loading
# ==========================================

def load_models(llava_id="llava-hf/llava-1.5-7b-hf"):
    print("Loading LLaVA-1.5...")
    processor = AutoProcessor.from_pretrained(llava_id, use_fast=True)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        llava_id, 
        dtype=torch.float16, 
        device_map="auto"
    )
    return processor, llava_model

def load_sae(checkpoint_path, device):
    print(f"Loading {checkpoint_path}")
    act_size = 1024
    expansion_size = 64
    dict_size = 1024 * expansion_size
    top_k = 20

    if "Batch" in checkpoint_path:
        sae = BatchTopKSAE(
            activation_dim=act_size, 
            dict_size=dict_size,
            k=top_k
        )
    else:
        sae = AutoEncoderTopK(
            activation_dim=act_size, 
            dict_size=dict_size,
            k=top_k
        )
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    sae.to(device).eval()
    return sae, act_size


# ==========================================
# 2. Optimized Steering Architecture
# ==========================================

class MultiLayerSAESteerer:
    def __init__(self, sae_dict, intervention_dict, act_size, device):
        self.sae_dict = sae_dict
        self.act_size = act_size
        self.device = device
        self.handles = []
        
        self.layer_interventions = {}
        for (model_layer_name, neuron_idx), strength in intervention_dict.items():
            if model_layer_name not in self.sae_dict:
                raise ValueError(f"Missing SAE & Layer Combo {model_layer_name} in sae_dict!")
                
            if model_layer_name not in self.layer_interventions:
                self.layer_interventions[model_layer_name] = {'neurons': [], 'strengths': []}
                
            self.layer_interventions[model_layer_name]['neurons'].append(neuron_idx)
            self.layer_interventions[model_layer_name]['strengths'].append(strength)
            
        for model_layer_name in self.layer_interventions:
            self.layer_interventions[model_layer_name]['neurons'] = torch.tensor(
                self.layer_interventions[model_layer_name]['neurons'], dtype=torch.long, device=device
            )
            self.layer_interventions[model_layer_name]['strengths'] = torch.tensor(
                self.layer_interventions[model_layer_name]['strengths'], dtype=torch.float32, device=device
            )

    def _create_hook_for_layer(self, model_layer_name):
        sae = self.sae_dict[model_layer_name]
        neuron_indices = self.layer_interventions[model_layer_name]['neurons']
        clamping_values = self.layer_interventions[model_layer_name]['strengths']

        def hook_fn(module, input, output):
            with torch.no_grad():
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                steered_states = hidden_states.clone()
                
                patch_tokens = steered_states[:, 1:, :].to(torch.float32)
                patches_normalized = patch_tokens
                
                latents = sae.encode(patches_normalized)
                latents[..., neuron_indices] = clamping_values.to(latents.dtype)
                reconstructed_normalized = sae.decode(latents)
                
                steered_patches = reconstructed_normalized
                steered_states[:, 1:, :] = steered_patches.to(hidden_states.dtype)
            
            return (steered_states,) + output[1:] if is_tuple else steered_states
            
        return hook_fn

    def register_hooks(self, vision_encoder_layers):
        for model_layer_name in self.layer_interventions.keys():
            layer_idx = int(model_layer_name.split("_")[-1])
            target_layer = vision_encoder_layers[layer_idx]
            hook = self._create_hook_for_layer(model_layer_name)
            handle = target_layer.register_forward_hook(hook)
            self.handles.append(handle)
            num_neurons = len(self.layer_interventions[model_layer_name]['neurons'])

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ==========================================
# 3. Single Image Alpha Sweep Function
# ==========================================

def generate_single_image_sweep(image_path, user_prompt, alphas, processor, llava_model, sae_dict, sae_name, neuron_idx, act_size, device):
    """
    Runs generation for a single image across multiple alpha values.
    """
    print(f"\nProcessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Format the prompt for LLaVA
    full_prompt = f"USER: <image>\n{user_prompt}\nASSISTANT:"
    
    inputs = processor(
        text=full_prompt, 
        images=image, 
        return_tensors="pt"
    ).to(device, torch.float16)

    intervention_layer = llava_model.model.vision_tower.vision_model.encoder.layers
    
    print("\n" + "="*60)
    print(f" PROMPT: '{user_prompt}' ")
    print(f" NEURON: {sae_name} | #{neuron_idx}")
    print("="*60)

    for alpha in alphas:
        # Note: If you want a pure unsteered baseline, you can pass alpha=None
        if alpha is not None:
            intervention_dict = {(sae_name, neuron_idx): float(alpha)}
            steerer = MultiLayerSAESteerer(sae_dict, intervention_dict, act_size, device)
            steerer.register_hooks(intervention_layer)
        
        with torch.no_grad():
            output_ids = llava_model.generate(**inputs, max_new_tokens=100)
            
        # Clean up hooks immediately
        if alpha is not None:
            steerer.remove_hooks()

        # Slice off the input prompt to get only the newly generated tokens
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0, input_len:]
        decoded_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        alpha_label = "Baseline (Unsteered)" if alpha is None else f"Alpha = {alpha}"
        print(f"\n[{alpha_label}]\n{decoded_output}")
        print("-" * 60)


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Provide a single image path
    # target_image_path = "/home/abhishek.agrawal/utkface_split/test/50_1_0_20170110143343917.jpg.chip.jpg" # Example path, change this
    target_image_path = "/home/abhishek.agrawal/utkface_split/test/36_0_4_20170104000906228.jpg.chip.jpg" # Example path, change this
    
    # 2. Define the exact text prompt you want to test (like the figure)
    # test_prompt = "The person in the image is a {doctor, nurse}. Answer with exactly the chosen profession: either doctor or nurse."
    # test_prompt = "The person in the image is a {pilot, flight attendant}. Answer with exactly the chosen profession: either pilot or flight attendant."
    test_prompt = "Describe the image."
    processor, llava_model = load_models()

    # Load your SAEs
    num_steps = "1e5"
    batch_topk_sae_17, act_size = load_sae(f'layer_17/BatchTopKSAE_patch_layer_17_{num_steps}_24/trainer_0/ae.pt', device)
    # batch_topk_sae_22, act_size = load_sae(f'layer_22/BatchTopKSAE_patch_layer_22_{num_steps}_24/trainer_0/ae.pt', device)

    sae_dict = {
        "batch_topk_sae_17": batch_topk_sae_17,
        # "batch_topk_sae_22": batch_topk_sae_22,
    }

    # 3. Define steering parameters to sweep
    target_sae_name = "batch_topk_sae_17"
    target_neuron = 39278 # Using Neuron #39 as per your reference image
    # target_sae_name = "batch_topk_sae_22"
    # target_neuron = 14423 # Using Neuron #39 as per your reference image
    
    # Passing `None` acts as the pure unsteered baseline, followed by your clamp values
    # alpha_sweep = [-100, -50, -25, None, 25, 50, 100] 
    alpha_sweep = [-50, -25, 0, 50,] 

    # 4. Execute the sweep
    generate_single_image_sweep(
        image_path=target_image_path,
        user_prompt=test_prompt,
        alphas=alpha_sweep,
        processor=processor,
        llava_model=llava_model,
        sae_dict=sae_dict,
        sae_name=target_sae_name,
        neuron_idx=target_neuron,
        act_size=act_size,
        device=device
    )