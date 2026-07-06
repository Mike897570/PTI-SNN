import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import re
import math
import time
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, surrogate, functional
from sklearn.metrics import classification_report, confusion_matrix
from timm.models.layers import trunc_normal_

# ======================== 配置 ======================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "D:/Code_reproduction/SNN-LSTM/Data/SEED_IV/eeg_feature_smooth"
time_steps = 64
num_classes = 4
batch_size = 32
total_epochs = 50
initial_lr = 5e-4


# ======================== 自适应膜电位损失 ======================== #
class AdaptiveMembraneLoss(nn.Module):
    def __init__(self, alpha_base=0.015, beta_base=0.03, decay_rate=0.98,
                 init_sparsity=0.1, adapt_rate=0.1, threshold_ratio=0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha_base = alpha_base
        self.beta_base = beta_base
        self.decay_rate = decay_rate
        self.target_sparsity = init_sparsity
        self.adapt_rate = adapt_rate
        self.threshold_ratio = threshold_ratio
        self.epoch = 0
        self.recorded_metrics = {}

    def update_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, inputs, targets, layered_membranes_with_thresholds):
        if self.epoch < 10:
            decay_factor = 1.0
        else:
            decay_factor = self.decay_rate ** min((self.epoch - 10) / 20, 1.0)

        alpha = self.alpha_base * decay_factor
        beta = self.beta_base * decay_factor

        ce_loss = self.ce(inputs, targets)

        mem_reg = 0.0
        count = 0
        total_membranes = 0

        for layer_type, membranes in layered_membranes_with_thresholds.items():
            for mem, spike, threshold in membranes:
                total_membranes += 1
                if mem is not None and isinstance(mem, torch.Tensor) and mem.numel() > 0:
                    threshold_value = threshold if threshold is not None else 0.3
                    near_threshold_mask = (mem > self.threshold_ratio * threshold_value) & (mem < threshold_value)

                    if near_threshold_mask.any():
                        near_threshold_values = mem[near_threshold_mask]
                        near_threshold_penalty = torch.mean(
                            torch.abs(near_threshold_values) *
                            (near_threshold_values - self.threshold_ratio * threshold_value)
                        )
                        mem_reg += near_threshold_penalty
                        count += 1

        mem_reg = mem_reg / count if count else torch.tensor(0.0, device=inputs.device)

        sparse_terms = []
        layer_weights = []

        for layer_type, membranes in layered_membranes_with_thresholds.items():
            for _, spike, threshold in membranes:
                if spike is not None and isinstance(spike, torch.Tensor):
                    spike_ratio = (spike > 0).float().mean()
                    sparse_terms.append(spike_ratio)

                    if 'attention' in layer_type:
                        layer_weights.append(1.2)
                    elif 'conv' in layer_type:
                        layer_weights.append(0.8)
                    else:
                        layer_weights.append(1.0)

        dynamic_target = self.target_sparsity * (0.8 + 0.2 * max(self.epoch / 50, 1.0))

        if sparse_terms:
            sparse_tensor = torch.stack(sparse_terms)
            weights_tensor = torch.tensor(layer_weights, device=sparse_tensor.device)
            avg_sparsity = (sparse_tensor * weights_tensor).sum() / weights_tensor.sum()
            sparse_reg = torch.abs(avg_sparsity - dynamic_target)
        else:
            avg_sparsity = torch.tensor(0.0, device=inputs.device)
            sparse_reg = torch.tensor(0.0, device=inputs.device)

        total_loss = ce_loss + alpha * mem_reg + beta * sparse_reg

        self.recorded_metrics = {
            'ce_loss': ce_loss.item(),
            'mem_reg': mem_reg.item() if count > 0 else 0.0,
            'sparse_reg': sparse_reg.item(),
            'avg_sparsity': avg_sparsity.item(),
            'active_neurons_ratio': count / max(total_membranes, 1),
            'dynamic_target': dynamic_target,
            'alpha': alpha,
            'beta': beta
        }

        return total_loss, ce_loss, mem_reg, sparse_reg


# ======================== Spikformer 模块 ======================== #
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(step_mode='m', tau=2.0, v_threshold=0.3, detach_reset=False, backend='torch')
        self.drop1 = nn.Dropout(drop_rate)

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(step_mode='m', tau=2.0, v_threshold=0.3, detach_reset=False, backend='torch')
        self.drop2 = nn.Dropout(drop_rate)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, N, C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)
        x = self.drop1(x)

        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        x = self.drop2(x)

        return x


class SpikeRateGuidedSpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, tau=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.tau = tau

        self.spike_estimator = neuron.LIFNode(
            tau=2.0,
            v_threshold=0.3,
            detach_reset=True,
            step_mode='m'
        )

        self.spike_rate_estimator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            neuron.LIFNode(tau=1.5, v_threshold=0.3, step_mode='m'),
            nn.Linear(dim, num_heads),
            nn.Tanh()
        )

        self.modulation_strength = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.8)
        self.modulation_clamp_min = 0.1
        self.modulation_clamp_max = 2.0

    def compute_actual_spike_rate_v2(self, x):
        spikes = self.spike_estimator(x)

        temporal_mean = spikes.mean(dim=0)
        spatial_mean = temporal_mean.mean(dim=1)

        spatial_std = temporal_mean.std(dim=1, unbiased=False)
        spatial_cv = spatial_std / (spatial_mean + 1e-8)

        spike_features = torch.cat([
            spatial_mean,
            spatial_cv,
        ], dim=-1)

        return spike_features, spikes

    def forward(self, x, spatial_attention_scores):
        T, B, N, C = x.shape
        H = self.num_heads

        spike_rate_features, actual_spikes = self.compute_actual_spike_rate_v2(x)

        rate_bias = self.spike_rate_estimator(spike_rate_features)
        rate_bias = rate_bias.view(B, H, 1, 1)

        modulation_strength = torch.clamp(
            self.modulation_strength,
            self.modulation_clamp_min,
            self.modulation_clamp_max
        )

        modulated_scores = spatial_attention_scores * (1 + modulation_strength * rate_bias)

        self.last_modulation_magnitude = (modulation_strength * rate_bias).abs().mean().item()
        self.last_actual_spikes = actual_spikes.detach()
        self.last_spike_rate = spike_rate_features[:, :C].mean().item()

        return modulated_scores


class MembranePotentialModulatedTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads, tau=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.tau = tau

        self.membrane_feature_extractor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            neuron.LIFNode(tau=1.5, v_threshold=0.3, step_mode='m'),
            nn.Linear(dim, num_heads),
            nn.Tanh()
        )

        self.temporal_decay = nn.Parameter(torch.linspace(1.0, 0.95, num_heads).view(1, num_heads, 1, 1))
        self.membrane_modulation = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.6)

    def compute_membrane_features(self, membrane_potentials):
        if membrane_potentials is None:
            return None

        temporal_mean = membrane_potentials.mean(dim=0)
        mean_feature = temporal_mean.mean(dim=1)

        temporal_std = membrane_potentials.std(dim=0)
        temporal_cv = temporal_std.mean(dim=1) / (torch.abs(temporal_mean).mean(dim=1) + 1e-8)

        membrane_features = torch.cat([
            mean_feature,
            temporal_cv,
        ], dim=-1)

        return membrane_features

    def forward(self, x, temporal_attention_scores, membrane_potentials):
        B, N, T, C = x.shape
        H = self.num_heads

        membrane_features = self.compute_membrane_features(membrane_potentials)

        if membrane_features is not None:
            if membrane_features.dim() == 1:
                membrane_features = membrane_features.unsqueeze(0).expand(B, -1)
            elif membrane_features.dim() == 2 and membrane_features.size(0) == B:
                pass
            else:
                membrane_features = membrane_features.mean(dim=0, keepdim=True).expand(B, -1)
        else:
            x_temporal_mean = x.mean(dim=[1, 2])
            x_temporal_std = x.std(dim=[1, 2], unbiased=False)
            x_temporal_cv = x_temporal_std / (x_temporal_mean.abs() + 1e-8)

            membrane_features = torch.cat([
                x_temporal_mean,
                x_temporal_cv,
            ], dim=-1)

        modulation_factors = self.membrane_feature_extractor(membrane_features)

        if modulation_factors.size(0) != B or modulation_factors.size(1) != H:
            modulation_factors = modulation_factors.view(B, -1)
            if modulation_factors.size(1) > H:
                modulation_factors = modulation_factors[:, :H]
            elif modulation_factors.size(1) < H:
                padding = torch.zeros(B, H - modulation_factors.size(1),
                                      device=modulation_factors.device)
                modulation_factors = torch.cat([modulation_factors, padding], dim=1)

        modulation_factors = modulation_factors.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        membrane_modulation = self.membrane_modulation.unsqueeze(0)

        temporal_factors = self.temporal_decay ** torch.arange(T, device=x.device).float().view(1, 1, 1, T)
        temporal_factors = temporal_factors.unsqueeze(0).expand(B, 1, H, 1, T)

        modulated_scores = temporal_attention_scores * (1 + membrane_modulation * modulation_factors)
        modulated_scores = modulated_scores * temporal_factors

        self.last_modulation_magnitude = (membrane_modulation * modulation_factors).abs().mean().item()
        self.last_membrane_features = {
            'mean_level': membrane_features[:, :C].mean().item(),
            'temporal_stability': membrane_features[:, C:].mean().item()
        }

        return modulated_scores


class SpikeEventConsistencyFusion(nn.Module):
    def __init__(self, dim, tau=2.0, drop_rate=0.2):
        super().__init__()
        self.dim = dim

        self.consistency_gate = nn.Sequential(
            nn.Linear(2, dim // 4),
            neuron.LIFNode(tau=1.5, v_threshold=0.3, step_mode='m'),
            nn.Linear(dim // 4, 2),
            nn.Sigmoid()
        )

        self.out_linear = nn.Linear(dim, dim)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_lif = neuron.LIFNode(tau=tau, v_threshold=0.3, step_mode='m')
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, spatial_x, temporal_x):
        T, B, N, C = spatial_x.shape

        spatial_rate = (spatial_x > 0).float().mean(dim=(0, 2, 3))
        temporal_rate = (temporal_x > 0).float().mean(dim=(0, 2, 3))

        consistency_feat = torch.stack(
            [spatial_rate, temporal_rate], dim=-1
        )

        gate = self.consistency_gate(consistency_feat)
        g_s = gate[:, 0].view(1, B, 1, 1)
        g_t = gate[:, 1].view(1, B, 1, 1)

        fused = g_s * spatial_x + g_t * temporal_x

        fused_flat = self.out_linear(fused.flatten(0, 1))
        fused_bn = self.out_bn(fused_flat.transpose(-1, -2)).transpose(-1, -2)
        fused_out = fused_bn.reshape(T, B, N, C)
        fused_out = self.out_lif(fused_out)

        return self.dropout(fused_out)


class SpatioTemporalSSA(nn.Module):
    def __init__(self, dim, num_heads=4, drop_rate=0.2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(dim // num_heads)
        self.dropout = nn.Dropout(drop_rate)

        self.spatial_q_dwconv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
        self.spatial_q_pwconv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.spatial_q_conv_bn = nn.BatchNorm1d(dim)
        self.spatial_q_conv_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=False, backend='torch')

        self.spatial_k_linear = nn.Linear(dim, dim)
        self.spatial_k_ln = nn.LayerNorm(dim)
        self.spatial_k_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=False, backend='torch')

        self.spatial_v_linear = nn.Linear(dim, dim)
        self.spatial_v_bn = nn.BatchNorm1d(dim)
        self.spatial_v_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=False, backend='torch')

        self.temporal_q_linear = nn.Linear(dim, dim)
        self.temporal_q_bn = nn.BatchNorm1d(dim)
        self.temporal_q_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=False, backend='torch')

        self.temporal_k_linear = nn.Linear(dim, dim)
        self.temporal_k_bn = nn.BatchNorm1d(dim)
        self.temporal_k_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=False, backend='torch')

        self.temporal_v_linear = nn.Linear(dim, dim)
        self.temporal_v_bn = nn.BatchNorm1d(dim)
        self.temporal_v_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=False, backend='torch')

        self.attn_lif = neuron.LIFNode(step_mode='m', tau=2.0, v_threshold=0.5, detach_reset=False, backend='torch')

        self.spike_rate_spatial_attention = SpikeRateGuidedSpatialAttention(dim, num_heads)
        self.membrane_temporal_attention = MembranePotentialModulatedTemporalAttention(dim, num_heads)

        self.innovative_spatial_weight = nn.Parameter(torch.tensor(4.0))
        self.innovative_temporal_weight = nn.Parameter(torch.tensor(1.0))

        self.weight_clamp_min = -1e6
        self.weight_clamp_max = 1e6

        self.fusion = SpikeEventConsistencyFusion(dim, drop_rate=drop_rate)

    def forward(self, x, epoch=None):
        T, B, N, C = x.shape

        modulation_info = {
            'spatial': 0.0,
            'temporal': 0.0
        }

        x_flat = x.flatten(0, 1)
        x_conv = x_flat.permute(0, 2, 1).contiguous()
        q_conv = self.spatial_q_dwconv(x_conv)
        q_conv = self.spatial_q_pwconv(q_conv)
        q_conv = self.spatial_q_conv_bn(q_conv)
        q_conv = q_conv.permute(0, 2, 1).contiguous()
        spatial_q = q_conv.reshape(T, B, N, C).contiguous()
        spatial_q = self.spatial_q_conv_lif(spatial_q)
        spatial_q = spatial_q.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        spatial_k = self.spatial_k_linear(x_flat)
        spatial_k = self.spatial_k_ln(spatial_k)
        spatial_k = spatial_k.reshape(T, B, N, C).contiguous()
        spatial_k = self.spatial_k_lif(spatial_k)
        spatial_k = spatial_k.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        spatial_v = self.spatial_v_linear(x_flat)
        spatial_v = self.spatial_v_bn(spatial_v.transpose(-1, -2)).transpose(-1, -2)
        spatial_v = spatial_v.reshape(T, B, N, C).contiguous()
        spatial_v = self.spatial_v_lif(spatial_v)
        spatial_v = spatial_v.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        spatial_attn = (spatial_q @ spatial_k.transpose(-2, -1)) * self.scale

        innovative_spatial_attn = self.spike_rate_spatial_attention(x, spatial_attn)
        innovative_spatial_w = torch.clamp(
            self.innovative_spatial_weight,
            self.weight_clamp_min,
            self.weight_clamp_max
        )
        spatial_attn_combined = spatial_attn * (1 + innovative_spatial_w * innovative_spatial_attn)

        spatial_x = (spatial_attn_combined @ spatial_v).transpose(2, 3).reshape(T, B, N, C).contiguous()

        x_time = x.permute(1, 2, 0, 3).contiguous()
        x_time_flat = x_time.flatten(0, 1)

        temporal_q = self.temporal_q_linear(x_time_flat)
        temporal_q = self.temporal_q_bn(temporal_q.transpose(-1, -2)).transpose(-1, -2)
        temporal_q = temporal_q.reshape(B, N, T, C)
        temporal_q = self.temporal_q_lif(temporal_q)
        temporal_q = temporal_q.reshape(B, N, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        temporal_k = self.temporal_k_linear(x_time_flat)
        temporal_k = self.temporal_k_bn(temporal_k.transpose(-1, -2)).transpose(-1, -2)
        temporal_k = temporal_k.reshape(B, N, T, C)
        temporal_k = self.temporal_k_lif(temporal_k)
        temporal_k = temporal_k.reshape(B, N, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        temporal_v = self.temporal_v_linear(x_time_flat)
        temporal_v = self.temporal_v_bn(temporal_v.transpose(-1, -2)).transpose(-1, -2)
        temporal_v = temporal_v.reshape(B, N, T, C)
        temporal_v = self.temporal_v_lif(temporal_v)
        temporal_v = temporal_v.reshape(B, N, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        temporal_attn = (temporal_q @ temporal_k.transpose(-2, -1)) * self.scale

        membrane_info = None
        try:
            temporal_neurons = [self.temporal_q_lif, self.temporal_k_lif, self.temporal_v_lif]
            membrane_tensors = []
            for neuron_module in temporal_neurons:
                if hasattr(neuron_module, 'v') and neuron_module.v is not None:
                    v_tensor = neuron_module.v.detach()
                    if isinstance(v_tensor, torch.Tensor) and v_tensor.numel() > 0:
                        if v_tensor.dim() == 4 and v_tensor.shape[0] == T:
                            membrane_tensors.append(v_tensor)
                        elif v_tensor.dim() == 3:
                            expanded_v = v_tensor.unsqueeze(0).expand(T, -1, -1, -1)
                            membrane_tensors.append(expanded_v)

            if membrane_tensors:
                membrane_info = torch.stack(membrane_tensors).mean(dim=0)
            else:
                membrane_info = torch.zeros(T, B, N, C, device=x.device)
        except Exception:
            membrane_info = torch.zeros(T, B, N, C, device=x.device)

        innovative_temporal_attn = self.membrane_temporal_attention(x_time, temporal_attn, membrane_info)
        innovative_temporal_w = torch.clamp(
            self.innovative_temporal_weight,
            self.weight_clamp_min,
            self.weight_clamp_max
        )
        temporal_attn_combined = temporal_attn * (1 + innovative_temporal_w * innovative_temporal_attn)

        temporal_x = (temporal_attn_combined @ temporal_v).transpose(2, 3).reshape(B, N, T, C).contiguous()
        temporal_x = temporal_x.permute(2, 0, 1, 3).contiguous()

        fused_x = self.fusion(spatial_x, temporal_x)

        x = self.attn_lif(fused_x)
        x = self.dropout(x)

        spatial_mod = torch.abs(innovative_spatial_attn - spatial_attn).mean().item() / (
                torch.abs(spatial_attn).mean().item() + 1e-8)
        temporal_mod = torch.abs(innovative_temporal_attn - temporal_attn).mean().item() / (
                torch.abs(temporal_attn).mean().item() + 1e-8)
        modulation_info['spatial'] = spatial_mod
        modulation_info['temporal'] = temporal_mod

        return x, modulation_info


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., drop_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatioTemporalSSA(
            dim,
            num_heads=num_heads,
            drop_rate=drop_rate
        )
        self.dropout1 = nn.Dropout(drop_rate)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(self, x, epoch=None):
        identity = x
        x = self.norm1(x)
        x, modulation_info = self.attn(x, epoch=epoch)
        x = self.dropout1(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = identity + x

        return x, modulation_info


class SpikeTransformer(nn.Module):
    def __init__(self, seq_len, in_features, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.embed = nn.Linear(in_features, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        trunc_normal_(self.temporal_pos_embed, std=.02)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=2.,
                    drop_rate=0.2
                )
            )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            neuron.LIFNode(tau=2.0, v_threshold=0.3),
            nn.Linear(64, 32),
            neuron.LIFNode(tau=2.0, v_threshold=0.3),
            nn.Linear(32, num_classes)
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x, epoch=None):
        x = self.embed(x) + self.pos_embed
        x = x.permute(1, 0, 2).unsqueeze(2)
        temporal_pos = self.temporal_pos_embed.permute(1, 0, 2).unsqueeze(1)
        x = x + temporal_pos

        modulation_info = {
            'spatial': [],
            'temporal': []
        }

        for blk in self.blocks:
            x, block_modulation = blk(x, epoch=epoch)
            if block_modulation is not None:
                modulation_info['spatial'].append(block_modulation.get('spatial', 0))
                modulation_info['temporal'].append(block_modulation.get('temporal', 0))

        x = x.mean(0)
        x = self.norm(x)
        x = x.mean(1)
        x = self.head(x)

        avg_modulation = {
            'spatial': np.mean(modulation_info['spatial']) if modulation_info['spatial'] else 0.0,
            'temporal': np.mean(modulation_info['temporal']) if modulation_info['temporal'] else 0.0
        }

        return x, avg_modulation


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels should be divisible by 2 for two branches"
        mid = out_channels // 2

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, (3, 3), padding=(1, 1)),
            neuron.LIFNode(tau=1.5, v_threshold=0.5, surrogate_function=surrogate.ATan())
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, (5, 3), padding=(2, 1)),
            neuron.LIFNode(tau=1.5, v_threshold=0.5, surrogate_function=surrogate.ATan())
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            neuron.LIFNode(tau=1.5, v_threshold=0.2, surrogate_function=surrogate.ATan())
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fusion(x)

        fusion_neuron = self.fusion[1]
        membrane = fusion_neuron.v if hasattr(fusion_neuron, 'v') else None

        spike = x
        x = nn.AdaptiveAvgPool2d((62, 5))(x)
        x = x.view(x.size(0), -1)

        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(4960, 310).to(x.device)

        x = self.projection(x)

        return x, membrane, spike


class FrequencyAwareEventDrivenEncoder(nn.Module):
    def __init__(self,
                 num_channels=62,
                 num_bands=5,
                 tau=2.0,
                 v_threshold=0.3):
        super().__init__()

        self.num_channels = num_channels
        self.num_bands = num_bands

        self.band_weight = nn.Parameter(
            torch.tensor([0.6, 0.8, 1.0, 1.2, 1.4])
        )

        self.encoder_neuron = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode='m'
        )

    def forward(self, x):
        B, T, F = x.shape
        assert F == self.num_channels * self.num_bands

        x = x.view(B, T, self.num_channels, self.num_bands)
        x = x * self.band_weight.view(1, 1, 1, self.num_bands)

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - mean) / std

        x = x.permute(1, 0, 2, 3).contiguous()

        spike = self.encoder_neuron(x)

        spike = spike.permute(1, 0, 2, 3).contiguous()
        spike = spike.view(B, T, -1)

        return spike


class PureSpatioTemporalSNN(nn.Module):
    def __init__(self, time_steps, num_classes, num_electrodes=62):
        super().__init__()
        self.spatial = SpatialEncoder()
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes
        self.encoder = FrequencyAwareEventDrivenEncoder()

        self.temporal = SpikeTransformer(
            seq_len=time_steps,
            in_features=310,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes
        )
        functional.set_step_mode(self, step_mode='m')

        self._register_key_modules()

    def _register_key_modules(self):
        self.key_membrane_modules = []

        spatial_modules = []
        for branch in [self.spatial.branch1, self.spatial.branch2, self.spatial.fusion]:
            if len(branch) > 1 and isinstance(branch[1], neuron.LIFNode):
                spatial_modules.append(branch[1])

        self.key_membrane_modules.extend(spatial_modules)

        def register_temporal_modules(module, prefix=""):
            for name, child in module.named_children():
                if isinstance(child, neuron.LIFNode):
                    self.key_membrane_modules.append(child)
                else:
                    register_temporal_modules(child, f"{prefix}.{name}")

        register_temporal_modules(self.temporal)

    def forward(self, x, epoch=None):
        x = self.encoder(x)
        B = x.size(0)
        spatial_feats = []
        all_spks = []
        current_membranes = {'conv': [], 'attention': [], 'mlp': [], 'other': []}

        for t in range(self.time_steps):
            xt = x[:, t, :].view(B, 1, self.num_electrodes, 5)
            with torch.set_grad_enabled(t == self.time_steps - 1):
                feat, membrane, spk = self.spatial(xt)
                spatial_feats.append(feat)
                all_spks.append(spk)
            if membrane is not None and spk is not None and isinstance(
                    membrane, torch.Tensor) and isinstance(spk, torch.Tensor):
                threshold = getattr(self.spatial.fusion[1], 'v_threshold', 0.2)
                if t < self.time_steps - 1:
                    current_membranes['conv'].append((membrane.detach(), spk.detach(), threshold))
                else:
                    current_membranes['conv'].append((membrane, spk, threshold))

        spatial_feats = torch.stack(spatial_feats, dim=1).to(x.device)
        logits, modulation_info = self.temporal(spatial_feats, epoch=epoch)
        avg_spk = torch.stack(all_spks).mean() if all_spks else torch.tensor(0.0, device=x.device)
        layered_membranes = self.selective_collect_membranes(current_membranes)

        return logits, layered_membranes, avg_spk, modulation_info

    def selective_collect_membranes(self, spatial_membranes):
        membranes = {
            'conv': spatial_membranes.get('conv', []),
            'attention': [],
            'mlp': [],
            'other': []
        }

        for module in self.key_membrane_modules:
            if hasattr(module, 'v') and module.v is not None:
                v = module.v
                if isinstance(v, torch.Tensor):
                    s = getattr(module, 's', torch.zeros_like(v))
                    threshold = getattr(module, 'v_threshold', 0.5)
                else:
                    continue

                module_name = str(module)
                if any(keyword in module_name for keyword in ['attn', 'attention']):
                    membranes['attention'].append((v, s, threshold))
                elif any(keyword in module_name for keyword in ['mlp', 'fc']):
                    membranes['mlp'].append((v, s, threshold))
                else:
                    membranes['other'].append((v, s, threshold))

        return membranes


def augment_eeg_data(x, noise_level=0.02, time_warp=0.1):
    B, T, C = x.shape
    device = x.device

    noise = torch.randn_like(x) * noise_level

    if random.random() < 0.1:
        warp_factor = 1.0 + random.uniform(-time_warp, time_warp)
        new_T = int(T * warp_factor)
        x_resampled = F.interpolate(x.permute(0, 2, 1), size=new_T, mode='linear', align_corners=False)
        x_resampled = F.interpolate(x_resampled, size=T, mode='linear', align_corners=False).permute(0, 2, 1)
    else:
        x_resampled = x

    if random.random() < 0.1:
        mask = torch.ones(B, 1, C, device=device)
        dropout_mask = (torch.rand(C, device=device) > 0.1).float()
        mask = mask * dropout_mask.view(1, 1, C)
        x_resampled = x_resampled * mask

    if random.random() < 0.1:
        scale = 1.0 + random.uniform(-0.1, 0.1)
        x_resampled = x_resampled * scale

    if random.random() < 0.1:
        num_bands = 5
        band_mask = torch.ones(1, 1, C, device=device)
        band_to_mask = random.randint(0, num_bands - 1)
        band_size = C // num_bands
        if band_size > 0:
            start_idx = band_to_mask * band_size
            end_idx = min((band_to_mask + 1) * band_size, C)
            band_mask[:, :, start_idx:end_idx] = 0.0
            x_resampled = x_resampled * band_mask

    return x_resampled + noise


def load_seed_features(path):
    all_features, all_labels, subject_ids = [], [], []

    session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    session_labels = [session1_label, session2_label, session3_label]

    mat_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.mat') and 'label' not in f.lower():
                mat_files.append(os.path.join(root, f))

    mat_files = sorted(mat_files)

    if not mat_files:
        raise ValueError(f"在路径 '{path}' 下没有找到任何 .mat 文件！请检查数据集路径。")

    print(f"找到 {len(mat_files)} 个 .mat 文件，正在加载...")

    subject_session_count = {}

    for file_path in mat_files:
        data = sio.loadmat(file_path)
        filename = os.path.basename(file_path)

        subject_id_match = re.findall(r'\d+', filename)
        subject_id = int(subject_id_match[0]) if subject_id_match else 0

        if subject_id not in subject_session_count:
            subject_session_count[subject_id] = 0

        session_idx = subject_session_count[subject_id] % 3
        subject_session_count[subject_id] += 1

        for i in range(24):
            key = f'de_movingAve{i + 1}'
            if key not in data:
                alt_key = f'de_LDS{i + 1}'
                if alt_key in data:
                    key = alt_key
                else:
                    continue

            arr = data[key]

            if len(arr.shape) != 3 or arr.shape[0] != 62 or arr.shape[2] != 5:
                if arr.shape[1] == 62 and arr.shape[2] == 5:
                    arr = arr.transpose(1, 0, 2)
                else:
                    continue

            T = arr.shape[1]
            feat = arr.transpose(1, 0, 2).reshape(T, -1)

            if T < time_steps:
                feat = np.vstack([feat, np.zeros((time_steps - T, 310))])
            else:
                feat = feat[:time_steps]

            all_features.append(feat)
            all_labels.append(session_labels[session_idx][i])
            subject_ids.append(subject_id)

    if len(all_features) == 0:
        raise ValueError("数据加载失败，未提取到任何有效样本！")

    X = np.array(all_features)
    y = np.array(all_labels)
    sids = np.array(subject_ids)

    return (X - X.min()) / (X.max() - X.min() + 1e-8), y, sids


class AdaptiveGradientClipper:
    def __init__(self, clip_value=1.0, mode='fixed'):
        self.clip_value = clip_value
        self.mode = mode

    def clip(self, model):
        if self.mode == 'fixed':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        else:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            clip_coef = self.clip_value / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)


def compute_all_layer_spike_rate(layered_membranes):
    total_spikes = 0
    total_elements = 0

    for layer_type, membranes in layered_membranes.items():
        for _, spike, _ in membranes:
            if spike is not None and isinstance(spike, torch.Tensor):
                total_spikes += (spike > 0).sum().item()
                total_elements += spike.numel()

    if total_elements > 0:
        return total_spikes / total_elements
    else:
        return 0.0


class ModulationMonitor:
    def __init__(self):
        self.spatial_modulations = []
        self.temporal_modulations = []

    def update(self, modulations):
        if modulations is not None:
            self.spatial_modulations.append(modulations.get('spatial', 0))
            self.temporal_modulations.append(modulations.get('temporal', 0))

    def get_avg(self):
        spatial_avg = np.mean(self.spatial_modulations) if self.spatial_modulations else 0.0
        temporal_avg = np.mean(self.temporal_modulations) if self.temporal_modulations else 0.0
        return spatial_avg, temporal_avg

    def reset(self):
        self.spatial_modulations = []
        self.temporal_modulations = []


def train_loso_fold(target_subject, features, labels, sids, device, total_epochs=150):
    train_idx = np.where(sids != target_subject)[0]
    test_idx = np.where(sids == target_subject)[0]

    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = PureSpatioTemporalSNN(time_steps, num_classes, num_electrodes=62).to(device)

    base_lr = initial_lr
    innov_lr = initial_lr * 0.8

    innov_params = [p for n, p in model.named_parameters() if 'innovative' in n]
    other_params = [p for n, p in model.named_parameters() if 'innovative' not in n]

    optimizer = optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': innov_params, 'lr': innov_lr}
    ], weight_decay=2e-4, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=4e-4
    )

    loss_fn = AdaptiveMembraneLoss(
        alpha_base=0.015,
        beta_base=0.03,
        decay_rate=0.98,
        init_sparsity=0.1,
        threshold_ratio=0.75
    )
    grad_clipper = AdaptiveGradientClipper(clip_value=5.0, mode='adaptive')

    best_test_acc_so_far = 0.0

    for epoch in range(1, total_epochs + 1):
        loss_fn.update_epoch(epoch)
        model.train()
        functional.reset_net(model)
        total, correct = 0, 0

        for xb, yb in train_loader:
            functional.reset_net(model)
            pred, layered_membranes, spk, modulations = model(xb, epoch=epoch)
            loss, ce, mem, sparse = loss_fn(pred, yb, layered_membranes)

            optimizer.zero_grad()
            loss.backward()
            grad_clipper.clip(model)
            optimizer.step()

            total += yb.size(0)
            correct += (pred.argmax(1) == yb).int().sum().item()

        scheduler.step()

        model.eval()
        functional.reset_net(model)
        test_total, test_correct = 0, 0

        with torch.no_grad():
            for xb, yb in test_loader:
                functional.reset_net(model)
                pred, layered_membranes, spk, _ = model(xb, epoch=epoch)
                test_total += yb.size(0)
                test_correct += (pred.argmax(1) == yb).int().sum().item()

        test_acc = test_correct / test_total
        if test_acc > best_test_acc_so_far:
            best_test_acc_so_far = test_acc
            best_model_path = f"results_loso_snn/checkpoints/best_model_subj_{target_subject}.pth"
            torch.save(model.state_dict(), best_model_path)

        if epoch % 50 == 0 or epoch == total_epochs:
            print(f"  [Subj {target_subject}] Epoch {epoch:03d}/{total_epochs} | Train Acc: {correct / total:.4f} | Test Acc: {test_acc:.4f}")

    final_model_path = f"results_loso_snn/checkpoints/final_model_subj_{target_subject}.pth"
    torch.save(model.state_dict(), final_model_path)

    model.eval()
    functional.reset_net(model)
    test_preds, test_targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            functional.reset_net(model)
            out, layered_membranes, _, _ = model(xb)
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_targets.extend(yb.cpu().numpy())

    final_test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    print(f"--> 被试 {target_subject} 最终测试准确率: {final_test_acc:.4f} | 最佳准确率: {best_test_acc_so_far:.4f}")

    return {
        'subject': target_subject,
        'final_test_acc': final_test_acc,
        'test_preds': test_preds,
        'test_targets': test_targets
    }


def main():
    print(f"\n{'=' * 50}")
    print("开始留一被试交叉验证 (LOSO Cross-Validation)")
    print(f"{'=' * 50}")

    os.makedirs("results_loso_snn/checkpoints", exist_ok=True)

    global_seed = 35
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    features, labels, sids = load_seed_features(data_path)
    unique_subjects = np.unique(sids)
    all_results = []

    for i, target_subject in enumerate(unique_subjects):
        result = train_loso_fold(target_subject, features, labels, sids, device, total_epochs)
        all_results.append(result)

    per_subj_reports = []
    final_accs = []

    for result in all_results:
        preds = result['test_preds']
        targets = result['test_targets']
        report = classification_report(targets, preds, target_names=['Neutral', 'Sad', 'Fear', 'Happy'],
                                       output_dict=True, zero_division=0)
        per_subj_reports.append(report)
        final_accs.append(result['final_test_acc'])

    avg_accuracy = np.mean([r['accuracy'] for r in per_subj_reports])
    avg_precision = np.mean([r['weighted avg']['precision'] for r in per_subj_reports])
    avg_recall = np.mean([r['weighted avg']['recall'] for r in per_subj_reports])
    avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in per_subj_reports])

    print("\n" + "=" * 50)
    print(f"平均准确率 (LOSO): {avg_accuracy:.4f} ± {np.std(final_accs):.4f}")
    print(f"平均加权 F1-score: {avg_f1:.4f}")
    print("=" * 50)

    with open("results_loso_snn/avg_classification_report.txt", "w") as f:
        f.write(f"Average Accuracy (LOSO): {avg_accuracy:.4f} ± {np.std(final_accs):.4f}\n")
        f.write(f"Average Weighted Precision: {avg_precision:.4f}\n")
        f.write(f"Average Weighted Recall: {avg_recall:.4f}\n")
        f.write(f"Average Weighted F1-score: {avg_f1:.4f}\n")

    with open("results_loso_snn/summary_results.txt", "w") as f:
        f.write("留一被试交叉验证（LOSO）结果汇总\n")
        f.write("=" * 50 + "\n\n")
        for result in all_results:
            f.write(f"被试 {result['subject']}: {result['final_test_acc']:.4f}\n")
        f.write(f"\n跨被试平均最终测试准确率: {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}\n")


if __name__ == "__main__":
    main()