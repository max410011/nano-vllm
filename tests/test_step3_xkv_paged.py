"""
Step 3: xKV + Paged Attention Tests

Tests for paged writeback functionality in xKV compression.
"""

import pytest
import torch

from nanovllm.xkv.config import xKVConfig, LayerGroup, generate_consecutive_xKV_config
from nanovllm.xkv.cache_manager import xKVCacheManager


class TestPagedWritebackConfig:
    """Test paged_writeback configuration."""

    def test_paged_writeback_default_false(self):
        """Test that paged_writeback defaults to False."""
        config = xKVConfig()
        assert config.paged_writeback is False

    def test_paged_writeback_can_be_enabled(self):
        """Test that paged_writeback can be set to True."""
        config = xKVConfig(paged_writeback=True)
        assert config.paged_writeback is True

    def test_generate_config_with_paged_writeback(self):
        """Test generate_consecutive_xKV_config with paged_writeback."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            rank_k=64,
            rank_v=128,
            paged_writeback=True,
        )
        assert config.paged_writeback is True


class TestAttentionModuleRegistration:
    """Test attention module registration for paged writeback."""

    def test_register_attention_stores_module(self):
        """Test that register_attention stores the module correctly."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            paged_writeback=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=8)

        # Create a mock attention module
        class MockAttention:
            def __init__(self):
                self.k_cache = torch.zeros(100, 8, 64)
                self.v_cache = torch.zeros(100, 8, 64)

            def write_kvcache(self, k, v, slot_mapping):
                pass

        attn = MockAttention()
        manager.register_attention(0, attn)

        assert 0 in manager.attention_modules
        assert manager.attention_modules[0] is attn

    def test_register_multiple_attentions(self):
        """Test registering multiple attention modules."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            paged_writeback=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=8)

        class MockAttention:
            pass

        for i in range(4):
            attn = MockAttention()
            manager.register_attention(i, attn)

        assert len(manager.attention_modules) == 4
        for i in range(4):
            assert i in manager.attention_modules


class TestWritebackCompressedToPagedCache:
    """Test writeback_compressed_to_paged_cache functionality."""

    def test_writeback_with_no_compressed_kv_does_nothing(self):
        """Test that writeback does nothing if no compressed KV exists."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            paged_writeback=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=8)

        # No compressed KV stored, should not raise
        slot_mapping = torch.zeros(10, dtype=torch.long)
        manager.writeback_compressed_to_paged_cache(1, slot_mapping)

    def test_writeback_raises_if_attention_not_registered(self):
        """Test that writeback raises error if attention is not registered."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            paged_writeback=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=8)

        # Store some compressed KV
        k = torch.randn(10, 8, 64)
        v = torch.randn(10, 8, 64)
        manager.store_compressed_kv(0, k, v)

        slot_mapping = torch.zeros(10, dtype=torch.long)

        with pytest.raises(RuntimeError, match="Attention module for layer 0 not registered"):
            manager.writeback_compressed_to_paged_cache(1, slot_mapping)

    def test_writeback_calls_write_kvcache(self):
        """Test that writeback calls write_kvcache on attention modules."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            paged_writeback=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=8)

        # Track write_kvcache calls
        call_count = [0]
        written_data = []

        class MockAttention:
            def write_kvcache(self, k, v, slot_mapping):
                call_count[0] += 1
                written_data.append((k.clone(), v.clone(), slot_mapping.clone()))

        # Register mock attentions
        for i in range(4):
            manager.register_attention(i, MockAttention())

        # Store compressed KV for layer 0 and 1 (group 0)
        k0 = torch.randn(10, 8, 64)
        v0 = torch.randn(10, 8, 64)
        k1 = torch.randn(10, 8, 64)
        v1 = torch.randn(10, 8, 64)
        manager.store_compressed_kv(0, k0, v0)
        manager.store_compressed_kv(1, k1, v1)

        slot_mapping = torch.arange(10, dtype=torch.long)

        # Writeback group 0 (layers 0 and 1, triggered at layer 1)
        manager.writeback_compressed_to_paged_cache(1, slot_mapping)

        # Should have called write_kvcache twice (once for each layer)
        assert call_count[0] == 2

    def test_writeback_clears_compressed_cache(self):
        """Test that writeback clears compressed cache for the group."""
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            num_layers=4,
            group_size=2,
            paged_writeback=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=8)

        class MockAttention:
            def write_kvcache(self, k, v, slot_mapping):
                pass

        for i in range(4):
            manager.register_attention(i, MockAttention())

        # Store compressed KV for all layers
        for i in range(4):
            k = torch.randn(10, 8, 64)
            v = torch.randn(10, 8, 64)
            manager.store_compressed_kv(i, k, v)

        slot_mapping = torch.arange(10, dtype=torch.long)

        # Writeback group 0 (layers 0 and 1)
        manager.writeback_compressed_to_paged_cache(1, slot_mapping)

        # Layers 0 and 1 should be cleared
        assert manager.get_compressed_kv(0) is None
        assert manager.get_compressed_kv(1) is None
        # Layers 2 and 3 should still exist
        assert manager.get_compressed_kv(2) is not None
        assert manager.get_compressed_kv(3) is not None


class TestAttentionWriteKVCache:
    """Test Attention.write_kvcache method."""

    def test_write_kvcache_method_exists(self):
        """Test that Attention class has write_kvcache method."""
        from nanovllm.layers.attention import Attention

        attn = Attention(
            num_heads=8,
            head_dim=64,
            scale=0.125,
            num_kv_heads=8,
        )
        assert hasattr(attn, "write_kvcache")
        assert callable(attn.write_kvcache)

    def test_write_kvcache_with_empty_cache_is_noop(self):
        """Test that write_kvcache does nothing with empty cache."""
        from nanovllm.layers.attention import Attention

        attn = Attention(
            num_heads=8,
            head_dim=64,
            scale=0.125,
            num_kv_heads=8,
        )

        # With default empty cache, should not raise
        k = torch.randn(10, 8, 64)
        v = torch.randn(10, 8, 64)
        slot_mapping = torch.arange(10, dtype=torch.long)

        attn.write_kvcache(k, v, slot_mapping)

