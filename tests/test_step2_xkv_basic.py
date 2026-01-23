"""
Step 2: xKV Basic Integration Tests

Tests for xKV compression functionality:
1. SVD compression accuracy
2. SLERP compression accuracy  
3. xKVConfig validation
4. xKVCacheManager operations
5. Integration with model (without full inference)
"""

import pytest
import torch
import torch.nn.functional as F


class TestSVDCompression:
    """Tests for fake_svd compression function."""

    def test_svd_output_shape(self):
        """Test SVD compression preserves shape."""
        from nanovllm.xkv.compressor import fake_svd
        
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        tensor = torch.randn(1, 8, 100, 64, dtype=torch.float32)
        rank = 32
        
        result = fake_svd(tensor, rank)
        
        assert result.shape == tensor.shape, f"Expected {tensor.shape}, got {result.shape}"

    def test_svd_compression_effect(self):
        """Test SVD compression reduces information (lower rank approximation)."""
        from nanovllm.xkv.compressor import fake_svd
        
        tensor = torch.randn(1, 4, 50, 64, dtype=torch.float32)
        rank = 16
        
        result = fake_svd(tensor, rank)
        
        # Reconstruction should be close but not identical
        diff = (tensor - result).abs().mean()
        assert diff > 0, "SVD compression should change the tensor"
        assert diff < tensor.abs().mean(), "SVD should not completely destroy the tensor"

    def test_svd_low_rank_more_compression(self):
        """Test lower rank means more compression (higher error)."""
        from nanovllm.xkv.compressor import fake_svd
        
        tensor = torch.randn(1, 4, 50, 64, dtype=torch.float32)
        
        result_high_rank = fake_svd(tensor, rank=48)
        result_low_rank = fake_svd(tensor, rank=8)
        
        error_high = (tensor - result_high_rank).abs().mean()
        error_low = (tensor - result_low_rank).abs().mean()
        
        assert error_low > error_high, "Lower rank should have higher reconstruction error"


class TestSLERPCompression:
    """Tests for SLERP merge function."""

    def test_slerp_output_shape(self):
        """Test SLERP merge preserves shape."""
        from nanovllm.xkv.compressor import fake_minicache_merge
        
        X1 = torch.randn(100, 64, dtype=torch.float32)
        X2 = torch.randn(100, 64, dtype=torch.float32)
        
        E1, E2 = fake_minicache_merge(X1, X2, t=0.5, gamma=0.05)
        
        assert E1.shape == X1.shape, f"Expected {X1.shape}, got {E1.shape}"
        assert E2.shape == X2.shape, f"Expected {X2.shape}, got {E2.shape}"

    def test_slerp_t_parameter_effect(self):
        """Test SLERP t parameter affects interpolation."""
        from nanovllm.xkv.compressor import slerp_merge_rows_batch

        # Create orthogonal vectors to ensure SLERP is applied (not bypassed)
        X1 = torch.randn(50, 64, dtype=torch.float32)
        X2 = torch.randn(50, 64, dtype=torch.float32)

        # Use slerp_merge_rows_batch directly to check interpolation effect
        E_t0, _, _, _ = slerp_merge_rows_batch(X1, X2, t=0.1, gamma=0.0)  # gamma=0 forces diverge
        E_t1, _, _, _ = slerp_merge_rows_batch(X1, X2, t=0.9, gamma=0.0)

        # Different t values should give different results
        assert not torch.allclose(E_t0, E_t1, atol=1e-3), "Different t should give different results"


class TestXKVConfig:
    """Tests for xKV configuration."""

    def test_config_creation(self):
        """Test basic xKVConfig creation."""
        from nanovllm.xkv.config import xKVConfig, LayerGroup
        
        config = xKVConfig(
            num_layers=28,
            layer_merge_impl="svd",
            rank_k=256,
            rank_v=768,
            layer_groups=[
                LayerGroup(layers=[0, 1]),
                LayerGroup(layers=[2, 3]),
            ],
        )
        
        assert config.layer_merge_impl == "svd"
        assert config.rank_k == 256
        assert config.rank_v == 768
        assert len(config.layer_groups) == 2

    def test_consecutive_config_generation(self):
        """Test generate_consecutive_xKV_config helper."""
        from nanovllm.xkv.config import generate_consecutive_xKV_config
        
        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=27,
            group_size=2,
            rank_k=256,
            rank_v=768,
        )
        
        assert len(config.layer_groups) == 14  # 28 layers / 2 = 14 groups
        assert config.layer_groups[0].layers == [0, 1]
        assert config.layer_groups[1].layers == [2, 3]

    def test_layer_map_building(self):
        """Test layer-to-group mapping."""
        from nanovllm.xkv.config import xKVConfig, LayerGroup
        
        config = xKVConfig(
            layer_merge_impl="svd",
            rank_k=256,
            layer_groups=[
                LayerGroup(layers=[0, 1, 2]),
                LayerGroup(layers=[3, 4, 5]),
            ],
        )
        
        group0 = config.get_group_for_layer(0)
        group1 = config.get_group_for_layer(1)
        group3 = config.get_group_for_layer(3)
        group_none = config.get_group_for_layer(10)
        
        assert group0 is group1  # Same group
        assert group0 is not group3  # Different groups
        assert group_none is None  # Layer not in any group

    def test_invalid_layer_merge_impl(self):
        """Test invalid layer_merge_impl raises error."""
        from nanovllm.xkv.config import xKVConfig

        with pytest.raises(ValueError, match="Invalid layer_merge_impl"):
            xKVConfig(layer_merge_impl="invalid")


class TestXKVCacheManager:
    """Tests for xKVCacheManager operations."""

    def test_cache_manager_creation(self):
        """Test xKVCacheManager creation."""
        from nanovllm.xkv import xKVCacheManager, generate_consecutive_xKV_config

        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=5,
            group_size=2,
            rank_k=32,
            rank_v=64,
        )

        manager = xKVCacheManager(config, num_kv_heads=4)

        assert manager.enabled
        assert manager.num_kv_heads == 4

    def test_should_store_temp(self):
        """Test should_store_temp logic."""
        from nanovllm.xkv import xKVCacheManager, generate_consecutive_xKV_config

        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=5,
            group_size=2,
            rank_k=32,
        )

        manager = xKVCacheManager(config, num_kv_heads=4)

        # Layers 0-5 are in groups, layer 10 is not
        assert manager.should_store_temp(0)
        assert manager.should_store_temp(5)
        assert not manager.should_store_temp(10)

    def test_should_compress(self):
        """Test should_compress logic (last layer in group)."""
        from nanovllm.xkv import xKVCacheManager, generate_consecutive_xKV_config

        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=5,
            group_size=2,
            rank_k=32,
        )

        manager = xKVCacheManager(config, num_kv_heads=4)

        # Groups: [0,1], [2,3], [4,5]
        # Last layers: 1, 3, 5
        assert not manager.should_compress(0)  # First in group
        assert manager.should_compress(1)      # Last in group [0,1]
        assert not manager.should_compress(2)  # First in group
        assert manager.should_compress(3)      # Last in group [2,3]

    def test_store_and_retrieve_temp_kv(self):
        """Test storing and retrieving temporary KV cache."""
        from nanovllm.xkv import xKVCacheManager, generate_consecutive_xKV_config

        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=3,
            group_size=2,
            rank_k=32,
        )

        manager = xKVCacheManager(config, num_kv_heads=4)

        # Store KV for layer 0
        k = torch.randn(10, 4, 64)
        v = torch.randn(10, 4, 64)
        manager.store_temp_kv(0, k, v)

        assert 0 in manager.temp_kv_cache
        stored_k, stored_v = manager.temp_kv_cache[0]
        assert stored_k.shape == k.shape
        assert stored_v.shape == v.shape

    def test_svd_compression_flow(self):
        """Test full SVD compression flow."""
        from nanovllm.xkv import xKVCacheManager, generate_consecutive_xKV_config

        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=1,
            group_size=2,
            rank_k=16,
            rank_v=32,
        )

        manager = xKVCacheManager(config, num_kv_heads=4)

        # Simulate prefill: store KV for both layers in group
        k0 = torch.randn(10, 4, 64)
        v0 = torch.randn(10, 4, 64)
        k1 = torch.randn(10, 4, 64)
        v1 = torch.randn(10, 4, 64)

        manager.store_temp_kv(0, k0, v0)
        manager.store_temp_kv(1, k1, v1)

        # Trigger compression at last layer
        compressed = manager.compress_group(1)

        assert 0 in compressed
        assert 1 in compressed

        # Check compressed shapes
        comp_k0, comp_v0 = compressed[0]
        comp_k1, comp_v1 = compressed[1]

        assert comp_k0.shape == k0.shape
        assert comp_v0.shape == v0.shape

    def test_clear_all(self):
        """Test clearing all caches."""
        from nanovllm.xkv import xKVCacheManager, generate_consecutive_xKV_config

        config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=1,
            group_size=2,
            rank_k=16,
        )

        manager = xKVCacheManager(config, num_kv_heads=4)

        # Store some data
        manager.store_temp_kv(0, torch.randn(10, 4, 64), torch.randn(10, 4, 64))
        manager.store_compressed_kv(0, torch.randn(10, 4, 64), torch.randn(10, 4, 64))

        assert len(manager.temp_kv_cache) > 0
        assert len(manager.compressed_kv_cache) > 0

        manager.clear_all()

        assert len(manager.temp_kv_cache) == 0
        assert len(manager.compressed_kv_cache) == 0

