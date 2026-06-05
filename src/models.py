import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -------------------------------------------------------------
# VLM Loader
# -------------------------------------------------------------
def load_vlm(model_id="llava-hf/llava-1.5-7b-hf", device_map="auto", dtype=torch.float16):
    """Loads LLaVA vision-language processor and model."""
    print(f"Loading LLaVA model '{model_id}'...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map=device_map
    )
    return processor, model


# -------------------------------------------------------------
# Base SAE Class
# -------------------------------------------------------------
class BaseAutoencoder(nn.Module):
    """Base class for custom autoencoder models."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(cfg["device"])

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


# -------------------------------------------------------------
# BatchTopK SAE
# -------------------------------------------------------------
class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


# -------------------------------------------------------------
# TopK SAE
# -------------------------------------------------------------
class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def reconstruction(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        return x, sae_out

    def get_features(self, x, k=5):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        return (acts_topk.indices[-2].detach().cpu(), acts_topk.values[-2].detach().cpu())

    def intervene_prompt_features(self, x, intv_features):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        acts_topk[:, intv_features] = -78000
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        return sae_out

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + aux_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


# -------------------------------------------------------------
# Vanilla SAE
# -------------------------------------------------------------
class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        return output


# -------------------------------------------------------------
# JumpReLU Helper Modules and SAE
# -------------------------------------------------------------
class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super().__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"], device=cfg["device"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)
        if use_pre_enc_bias:
            x = x - self.b_dec
        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)
        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec
        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss
        loss = l2_loss + l1_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
        }
        return output


# -------------------------------------------------------------
# Unified SAE Loader (supports library-based and custom SAEs)
# -------------------------------------------------------------
def load_sae(checkpoint_path, device, cfg=None, sae_type="topk"):
    """
    Unified function to load an SAE checkpoint. Proactively supports loading
    both our custom architectures and library-based architectures from dictionary_learning.
    """
    print(f"Loading SAE checkpoint from {checkpoint_path}...")
    
    # If using dictionary_learning library models
    is_dictionary_learning = False
    
    # Try importing dictionary_learning modules
    try:
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        from dictionary_learning.trainers.batch_top_k import BatchTopKSAE as LibBatchTopKSAE
        is_dictionary_learning = True
    except ImportError:
        pass

    # Basic defaults if config not provided
    if cfg is None:
        from src.config import get_topk_config
        cfg = get_topk_config()
        cfg['act_size'] = 1024
        cfg['dict_size'] = 1024 * 64
        cfg['top_k'] = 20
        cfg['device'] = device

    # Determine architecture
    filename = os.path.basename(checkpoint_path)
    
    if is_dictionary_learning and ("trainer" in checkpoint_path or "dictionary_learning" in checkpoint_path or "ae.pt" in filename):
        # We assume the library models are used
        act_size = cfg.get('act_size', 1024)
        dict_size = cfg.get('dict_size', 1024 * 64)
        top_k = cfg.get('top_k', 20)
        
        if "Batch" in checkpoint_path or "Batch" in sae_type:
            print("Instantiating dictionary_learning LibBatchTopKSAE...")
            sae = LibBatchTopKSAE(activation_dim=act_size, dict_size=dict_size, k=top_k)
        else:
            print("Instantiating dictionary_learning AutoEncoderTopK...")
            sae = AutoEncoderTopK(activation_dim=act_size, dict_size=dict_size, k=top_k)
    else:
        # Custom models
        if sae_type == "topk":
            sae = TopKSAE(cfg)
        elif sae_type == "batch_topk":
            sae = BatchTopKSAE(cfg)
        elif sae_type == "jumprelu":
            sae = JumpReLUSAE(cfg)
        elif sae_type == "vanilla":
            sae = VanillaSAE(cfg)
        else:
            raise ValueError(f"Unknown custom sae_type: {sae_type}")

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sae.load_state_dict(state_dict)
    sae.to(device).eval()
    
    return sae
