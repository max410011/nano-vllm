"""
xKV - Cross-layer KV-Cache Compression for nano-vllm.

This module implements SVD-based and SLERP-based compression for KV cache,
enabling memory-efficient inference with minimal accuracy loss.
"""

from nanovllm.xkv.config import xKVConfig, LayerGroup, generate_consecutive_xKV_config
from nanovllm.xkv.compressor import fake_svd, fake_minicache_merge
from nanovllm.xkv.cache_manager import xKVCacheManager

__all__ = [
    "xKVConfig",
    "LayerGroup",
    "generate_consecutive_xKV_config",
    "fake_svd",
    "fake_minicache_merge",
    "xKVCacheManager",
]

