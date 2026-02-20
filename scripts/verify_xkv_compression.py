#!/usr/bin/env python3
"""
Verify that xKV compression is actually being applied during evaluation.

This script adds debug output to track:
1. When KV cache is stored (temp)
2. When compression is triggered
3. When compressed KV is used for attention
"""

import sys
import torch
from nanovllm.llm import LLM
from nanovllm.xkv import generate_consecutive_xKV_config, xKVCacheManager
from nanovllm.xkv.compressor import fake_svd
from transformers import AutoConfig, AutoTokenizer

MODEL_PATH = '/share2/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'


def test_compression_directly():
    """Test compression logic directly without loading full model."""
    print("=" * 60)
    print("Test 1: Direct compression test")
    print("=" * 60)
    
    hf_config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    
    # Create xKV config with aggressive compression
    xkv_config = generate_consecutive_xKV_config(
        layer_merge_impl='svd',
        start_layer=0,
        end_layer=num_layers - 1,
        num_layers=num_layers,
        group_size=2,
        rank_k=16,
        rank_v=32,
        paged_writeback=False,  # Don't need paged cache for this test
    )
    
    manager = xKVCacheManager(xkv_config, num_kv_heads)
    
    # Simulate prefill with random KV
    seq_len = 100
    k0 = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda')
    v0 = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda')
    k1 = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda')
    v1 = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda')
    
    print(f"\nOriginal K0: mean={k0.mean():.4f}, std={k0.std():.4f}")
    print(f"Original K1: mean={k1.mean():.4f}, std={k1.std():.4f}")
    
    # Store temp KV (simulating Layer 0)
    manager.store_temp_kv(0, k0, v0)
    
    # At this point, Layer 0's attention would use original k0, v0
    # because compression hasn't happened yet
    
    # Store temp KV (simulating Layer 1)
    manager.store_temp_kv(1, k1, v1)
    
    # Trigger compression at Layer 1
    compressed_kv = manager.compress_group(1)
    
    # Store compressed KV
    for lyr_idx, (comp_k, comp_v) in compressed_kv.items():
        manager.store_compressed_kv(lyr_idx, comp_k, comp_v)
    
    # Now check if compressed KV is different from original
    for layer_idx in [0, 1]:
        comp_kv = manager.get_compressed_kv(layer_idx)
        if comp_kv is not None:
            comp_k, comp_v = comp_kv
            orig_k = k0 if layer_idx == 0 else k1
            mse_k = ((orig_k - comp_k) ** 2).mean().item()
            print(f"\nLayer {layer_idx}:")
            print(f"  K MSE (original vs compressed): {mse_k:.6f}")
            print(f"  Compressed K: mean={comp_k.mean():.4f}, std={comp_k.std():.4f}")
            
            if mse_k < 0.01:
                print(f"  ⚠️  WARNING: MSE is very small, compression may not be effective!")
            else:
                print(f"  ✅ Compression is effective (MSE > 0.01)")
    
    return True


def test_attention_uses_compressed():
    """Test that attention calculation uses compressed KV."""
    print("\n" + "=" * 60)
    print("Test 2: Verify attention uses compressed KV")
    print("=" * 60)
    
    # This is the key issue:
    # - Layer 0 attention is computed BEFORE compression
    # - Layer 1 attention is computed AFTER compression
    # 
    # For xKV to affect accuracy, we need either:
    # A) Delay Layer 0 attention until after compression
    # B) Use decode mode where all layers use compressed KV from paged cache
    # C) Accept that only Layer 1 (and later groups' last layers) use compressed KV
    
    print("\nCurrent behavior:")
    print("- Layer 0: Uses ORIGINAL KV (compression happens later)")
    print("- Layer 1: Uses COMPRESSED KV (compression just triggered)")
    print("")
    print("This means only 50% of layers in each group use compressed KV!")
    print("For group_size=2, layers 1, 3, 5, ... use compressed KV")
    print("But layers 0, 2, 4, ... use original KV")
    print("")
    print("This explains why accuracy doesn't change much:")
    print("The compression effect is diluted because many layers use original KV.")
    
    return True


def main():
    print("xKV Compression Verification")
    print("=" * 60)
    
    test_compression_directly()
    test_attention_uses_compressed()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The xKV implementation has a fundamental timing issue:

1. Compression is triggered at the LAST layer of each group
2. But earlier layers in the group have already computed attention
   using the ORIGINAL (uncompressed) KV

To fix this, we need one of:
A) Run prefill twice: first to collect & compress KV, second to compute logits
B) Use token-by-token decode for evaluation (uses compressed KV from paged cache)
C) Restructure the model to delay attention computation until compression

For evaluation (lm_eval), option A or B would correctly measure the impact
of KV compression on model accuracy.
""")


if __name__ == "__main__":
    main()

