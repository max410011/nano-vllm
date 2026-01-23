"""
xKV Cache Manager for nano-vllm.

This module manages the KV cache compression process during inference:
- Stores temporary pre-RoPE KV cache during prefill
- Triggers compression at the end of each layer group
- Returns compressed KV for attention computation
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from nanovllm.xkv.config import xKVConfig, LayerGroup
from nanovllm.xkv.compressor import fake_svd, fake_minicache_merge

if TYPE_CHECKING:
    from nanovllm.layers.attention import Attention


class xKVCacheManager:
    """
    Manages xKV compression process.

    During prefill:
    1. Stores pre-RoPE K, V for each layer in a group
    2. At the last layer of each group, triggers compression
    3. Returns compressed K, V for attention computation
    """

    def __init__(self, config: xKVConfig, num_kv_heads: int):
        """
        Initialize xKV cache manager.

        Args:
            config: xKV configuration
            num_kv_heads: Number of KV heads per layer (after TP split)
        """
        self.config = config
        self.num_kv_heads = num_kv_heads

        # Temporary storage for pre-RoPE KV during prefill
        # Dict[layer_idx, (k_pre_rope, v)]
        self.temp_kv_cache: Dict[int, Tuple[Tensor, Tensor]] = {}

        # Storage for compressed KV after compression
        # Dict[layer_idx, (k_compressed, v_compressed)]
        self.compressed_kv_cache: Dict[int, Tuple[Tensor, Tensor]] = {}

    @property
    def enabled(self) -> bool:
        """Check if xKV compression is enabled."""
        return len(self.config.layer_groups) > 0

    def is_layer_in_group(self, layer_idx: int) -> bool:
        """Check if this layer belongs to any compression group."""
        return self.config.get_group_for_layer(layer_idx) is not None

    def should_store_temp(self, layer_idx: int) -> bool:
        """Check if we should store temp KV cache for this layer."""
        return self.is_layer_in_group(layer_idx)

    def should_compress(self, layer_idx: int) -> bool:
        """Check if this is the last layer in its group (triggers compression)."""
        group = self.config.get_group_for_layer(layer_idx)
        if group is None:
            return False
        return layer_idx == group.layers[-1]

    def store_temp_kv(self, layer_idx: int, k: Tensor, v: Tensor):
        """
        Store pre-RoPE K, V for later compression.

        Args:
            layer_idx: Current layer index
            k: Key tensor (pre-RoPE), shape (num_tokens, num_kv_heads, head_dim)
            v: Value tensor, shape (num_tokens, num_kv_heads, head_dim)
        """
        # Clone k because it will be modified in-place by RoPE
        self.temp_kv_cache[layer_idx] = (k.clone(), v.clone())

    def get_group_layers(self, layer_idx: int) -> Optional[List[int]]:
        """Get all layer indices in the same group as layer_idx."""
        group = self.config.get_group_for_layer(layer_idx)
        if group is None:
            return None
        return group.layers

    @torch.no_grad()
    def compress_group(self, last_layer_idx: int) -> Dict[int, Tuple[Tensor, Tensor]]:
        """
        Compress KV cache for a layer group.

        Args:
            last_layer_idx: The last layer index in the group

        Returns:
            Dict[layer_idx, (compressed_k, compressed_v)]
        """
        group = self.config.get_group_for_layer(last_layer_idx)
        if group is None:
            return {}

        layer_indices = group.layers

        # Collect KV from temp cache
        keys = []
        values = []
        for idx in layer_indices:
            if idx not in self.temp_kv_cache:
                raise RuntimeError(f"Layer {idx} KV cache not found in temp storage")
            k, v = self.temp_kv_cache[idx]
            keys.append(k)
            values.append(v)

        # Compress based on method
        if self.config.layer_merge_impl == "svd":
            compressed_keys, compressed_values = self._compress_svd(keys, values, group)
        elif self.config.layer_merge_impl == "slerp":
            compressed_keys, compressed_values = self._compress_slerp(keys, values, group)
        else:
            raise NotImplementedError(f"Unknown method: {self.config.layer_merge_impl}")

        # Build result dict
        result = {}
        for i, layer_idx in enumerate(layer_indices):
            result[layer_idx] = (compressed_keys[i], compressed_values[i])

        # Clear temp cache for this group
        for idx in layer_indices:
            del self.temp_kv_cache[idx]

        return result

    def _compress_svd(
        self,
        keys: List[Tensor],
        values: List[Tensor],
        group: LayerGroup,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply SVD compression to grouped KV cache."""
        # Reshape from (N, num_heads, head_dim) to (1, num_heads, N, head_dim) for SVD
        keys_4d = [k.unsqueeze(0).transpose(1, 2) for k in keys]
        values_4d = [v.unsqueeze(0).transpose(1, 2) for v in values]

        # Concatenate along heads dimension
        combined_key = torch.cat(keys_4d, dim=1)
        combined_value = torch.cat(values_4d, dim=1)

        split_sizes = [self.num_kv_heads for _ in keys]

        # Apply SVD compression
        if self.config.merge_key and group.rank_k is not None:
            combined_key = fake_svd(combined_key.float(), rank=group.rank_k).to(combined_key.dtype)

        if self.config.merge_value and group.rank_v is not None:
            combined_value = fake_svd(combined_value.float(), rank=group.rank_v).to(combined_value.dtype)

        # Split back to per-layer tensors
        key_layers_4d = torch.split(combined_key, split_sizes, dim=1)
        value_layers_4d = torch.split(combined_value, split_sizes, dim=1)

        # Reshape back to (N, num_heads, head_dim)
        compressed_keys = [k.squeeze(0).transpose(0, 1) for k in key_layers_4d]
        compressed_values = [v.squeeze(0).transpose(0, 1) for v in value_layers_4d]

        return compressed_keys, compressed_values

    def _compress_slerp(
        self,
        keys: List[Tensor],
        values: List[Tensor],
        group: LayerGroup,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply SLERP compression to grouped KV cache (only for group_size=2)."""
        assert len(keys) == 2 and len(values) == 2, "SLERP only supports group size 2"

        slerp_t = group.slerp_t if group.slerp_t is not None else 0.5
        slerp_gamma = group.slerp_gamma if group.slerp_gamma is not None else 0.05

        compressed_keys = keys
        compressed_values = values

        if self.config.merge_key:
            k0_flat = keys[0].reshape(-1, keys[0].shape[-1])
            k1_flat = keys[1].reshape(-1, keys[1].shape[-1])
            k0_hat, k1_hat = fake_minicache_merge(k0_flat, k1_flat, t=slerp_t, gamma=slerp_gamma)
            compressed_keys = [k0_hat.reshape(keys[0].shape), k1_hat.reshape(keys[1].shape)]

        if self.config.merge_value:
            v0_flat = values[0].reshape(-1, values[0].shape[-1])
            v1_flat = values[1].reshape(-1, values[1].shape[-1])
            v0_hat, v1_hat = fake_minicache_merge(v0_flat, v1_flat, t=slerp_t, gamma=slerp_gamma)
            compressed_values = [v0_hat.reshape(values[0].shape), v1_hat.reshape(values[1].shape)]

        return compressed_keys, compressed_values

    def store_compressed_kv(self, layer_idx: int, k: Tensor, v: Tensor):
        """Store compressed KV for later use."""
        self.compressed_kv_cache[layer_idx] = (k, v)

    def get_compressed_kv(self, layer_idx: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Get compressed KV for a layer (if available)."""
        return self.compressed_kv_cache.get(layer_idx, None)

    def clear_temp_cache(self):
        """Clear all temporary KV cache (call after prefill)."""
        self.temp_kv_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_compressed_cache(self):
        """Clear all compressed KV cache."""
        self.compressed_kv_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_all(self):
        """Clear all caches."""
        self.temp_kv_cache.clear()
        self.compressed_kv_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

