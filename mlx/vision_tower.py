"""
dots.mocr vision tower ported to MLX.

Mirrors the PyTorch DotsVisionTransformer architecture:
  DotsPatchEmbed -> 42x DotsVisionBlock -> RMSNorm -> PatchMerger
Each block: RMSNorm -> VisionAttention (with RoPE) -> RMSNorm -> SwiGLU FFN
"""
import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    embed_dim: int = 1536
    hidden_size: int = 1536
    intermediate_size: int = 4224
    num_hidden_layers: int = 42
    num_attention_heads: int = 12
    num_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    rms_norm_eps: float = 1e-5
    use_bias: bool = False
    is_causal: bool = False
    post_norm: bool = True


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs):
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # freqs: (seq, head_dim//2) -> expand for heads and tile for full dim
    # mx.tile replicates the whole slice [a,b]->[a,b,a,b], matching PyTorch .repeat()
    cos = mx.expand_dims(cos, axis=1)  # (seq, 1, head_dim//2)
    sin = mx.expand_dims(sin, axis=1)
    cos = mx.tile(cos, (1, 1, 2))  # (seq, 1, head_dim)
    sin = mx.tile(sin, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)  # (1, seq, 1, head_dim)
    sin = mx.expand_dims(sin, axis=0)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, seqlen: int):
        seq = mx.arange(seqlen, dtype=mx.float32)
        freqs = mx.outer(seq, self.inv_freq)
        return freqs


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb):
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (seq, heads, head_dim)

        # Apply rotary embeddings
        q = apply_rotary_pos_emb_vision(
            mx.expand_dims(q, axis=0), rotary_pos_emb
        ).squeeze(0)
        k = apply_rotary_pos_emb_vision(
            mx.expand_dims(k, axis=0), rotary_pos_emb
        ).squeeze(0)

        # Process each sub-sequence independently (no O(N^2) mask needed)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        start = 0
        outputs = []
        for slen in seqlens:
            slen = int(slen)
            q_i = q[start:start + slen]  # (slen, heads, head_dim)
            k_i = k[start:start + slen]
            v_i = v[start:start + slen]
            # MLX SDPA expects (batch, heads, seq, head_dim)
            q_i = mx.expand_dims(q_i.transpose(1, 0, 2), axis=0)
            k_i = mx.expand_dims(k_i.transpose(1, 0, 2), axis=0)
            v_i = mx.expand_dims(v_i.transpose(1, 0, 2), axis=0)
            scale = math.sqrt(1.0 / self.head_dim)
            out = mx.fast.scaled_dot_product_attention(q_i, k_i, v_i, scale=scale)
            # (1, heads, slen, head_dim) -> (slen, heads, head_dim)
            out = out.squeeze(0).transpose(1, 0, 2)
            outputs.append(out)
            start += slen

        attn_output = mx.concatenate(outputs, axis=0)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def __call__(self, x):
        return self.fc2(nn.silu(self.fc1(x)) * self.fc3(x))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attn = VisionAttention(
            config.embed_dim, config.num_attention_heads, bias=config.use_bias
        )
        self.norm1 = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config.embed_dim, config.intermediate_size, bias=config.use_bias)
        self.norm2 = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings using a Conv2d-equivalent linear projection."""
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        # Conv2d(3, 1536, 14x14, stride=14) — stored as weight (1536, 3, 14, 14) + bias
        self.proj_weight = mx.zeros((config.embed_dim, config.num_channels, config.patch_size, config.patch_size))
        self.proj_bias = mx.zeros((config.embed_dim,))
        self.norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, x):
        # x: (num_patches, num_channels * temporal_patch_size * patch_size * patch_size)
        # Reshape to (num_patches, channels, patch_h, patch_w), take temporal slice 0
        x = x.reshape(-1, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = x[:, :, 0]  # (num_patches, C, H, W)
        # Apply conv2d as matrix multiply: flatten spatial dims, multiply by flattened kernel
        # weight: (embed_dim, C, H, W) -> (embed_dim, C*H*W)
        w = self.proj_weight.reshape(self.embed_dim, -1)  # (embed_dim, C*H*W)
        x_flat = x.reshape(x.shape[0], -1)  # (num_patches, C*H*W)
        x = x_flat @ w.T + self.proj_bias  # (num_patches, embed_dim)
        x = self.norm(x)
        return x


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp_2 = nn.Linear(self.hidden_size, dim)

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        x = nn.gelu(self.mlp_0(x))
        x = self.mlp_2(x)
        return x


class DotsVisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(config)

        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [VisionBlock(config) for _ in range(config.num_hidden_layers)]

        if config.post_norm:
            self.post_trunk_norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
        )

    def get_pos_ids_by_grid(self, grid_thw):
        """Compute 2D position IDs for rotary embeddings based on image grid."""
        pos_ids = []
        for i in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
            hpos = mx.broadcast_to(mx.arange(h).reshape(-1, 1), (h, w))
            hpos = hpos.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).transpose(0, 2, 1, 3).reshape(-1)

            wpos = mx.broadcast_to(mx.arange(w).reshape(1, -1), (h, w))
            wpos = wpos.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).transpose(0, 2, 1, 3).reshape(-1)

            hw_ids = mx.stack([hpos, wpos], axis=-1)
            if t > 1:
                hw_ids = mx.tile(hw_ids, (t, 1))
            pos_ids.append(hw_ids)
        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = int(grid_thw[:, 1:].max())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        return rotary_pos_emb

    def __call__(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Compute cu_seqlens: each spatial sequence (h*w) is repeated t times per image
        spatial_seqs = grid_thw[:, 1] * grid_thw[:, 2]  # h * w per image
        temporal = grid_thw[:, 0]  # t per image
        # repeat_interleave: repeat each spatial seq count by its temporal count
        repeated = mx.concatenate([mx.broadcast_to(s, (int(t),)) for s, t in zip(spatial_seqs, temporal)])
        cu_seqlens = mx.concatenate([mx.array([0]), mx.cumsum(repeated)])

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states


def load_vision_weights(model: DotsVisionTower, weights_dir: str):
    """Load PyTorch vision tower weights into the MLX model."""
    from safetensors import safe_open
    import json
    import os
    import numpy as np

    index_path = os.path.join(weights_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Collect vision tower weights from all shards
    vt_keys = {k: v for k, v in index["weight_map"].items() if k.startswith("vision_tower.")}
    shard_files = sorted(set(vt_keys.values()))

    raw_weights = {}
    for shard in shard_files:
        shard_path = os.path.join(weights_dir, shard)
        # Use PyTorch framework to handle bfloat16, then convert to numpy float32
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("vision_tower."):
                    raw_weights[key] = f.get_tensor(key).float().numpy()

    # Map PyTorch keys to MLX model keys
    mapped = {}
    for pt_key, np_val in raw_weights.items():
        mlx_key = pt_key.replace("vision_tower.", "")

        # patch_embed.patchifier.proj -> patch_embed.proj_weight/proj_bias
        mlx_key = mlx_key.replace("patch_embed.patchifier.proj.weight", "patch_embed.proj_weight")
        mlx_key = mlx_key.replace("patch_embed.patchifier.proj.bias", "patch_embed.proj_bias")
        mlx_key = mlx_key.replace("patch_embed.patchifier.norm.", "patch_embed.norm.")

        # merger.mlp.0 -> merger.mlp_0, merger.mlp.2 -> merger.mlp_2
        mlx_key = mlx_key.replace("merger.mlp.0.", "merger.mlp_0.")
        mlx_key = mlx_key.replace("merger.mlp.2.", "merger.mlp_2.")

        mapped[mlx_key] = mx.array(np_val)

    # Load into model (strict=False to skip computed buffers like inv_freq)
    model.load_weights(list(mapped.items()), strict=False)

    print(f"Loaded {len(mapped)} vision tower weights")
    return model
