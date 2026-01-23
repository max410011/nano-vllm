"""
Step 4: xKV Sparse Integration Tests

Tests for ShadowKV-style sparse attention integration:
1. Sparse selector functionality
2. Chunk-based landmark computation
3. Outlier detection
4. Integration with xKV compression
"""

import pytest
import torch

from nanovllm.xkv import (
    xKVConfig,
    generate_consecutive_xKV_config,
    xKVCacheManager,
    SparseSelector,
)


class TestSparseSelector:
    """Tests for SparseSelector class."""

    def test_sparse_selector_init(self):
        """Test SparseSelector initialization."""
        selector = SparseSelector(
            chunk_size=8,
            sparse_budget=256,
            num_outliers=48,
        )
        assert selector.chunk_size == 8
        assert selector.sparse_budget == 256
        assert selector.num_outliers == 48
        assert selector.landmarks is None
        assert selector.outlier_k is None

    def test_compute_landmarks_and_outliers(self):
        """Test landmark and outlier computation."""
        selector = SparseSelector(chunk_size=8, sparse_budget=16, num_outliers=4)
        
        # Create test KV cache: 128 tokens, 4 heads, 64 dim
        num_tokens = 128
        num_kv_heads = 4
        head_dim = 64
        
        k = torch.randn(num_tokens, num_kv_heads, head_dim)
        v = torch.randn(num_tokens, num_kv_heads, head_dim)
        
        selector.compute_landmarks_and_outliers(k, v)
        
        # Check landmarks are computed
        assert selector.landmarks is not None
        num_chunks = num_tokens // selector.chunk_size
        assert selector.landmarks.shape == (num_chunks, num_kv_heads, head_dim)
        
        # Check outliers are detected
        assert selector.outlier_indices is not None
        assert len(selector.outlier_indices) == 4
        assert selector.outlier_k is not None
        assert selector.outlier_k.shape[0] == 4 * selector.chunk_size

    def test_compute_landmarks_with_padding(self):
        """Test landmark computation with padding for non-divisible lengths."""
        selector = SparseSelector(chunk_size=8, sparse_budget=8, num_outliers=2)
        
        # 100 tokens (not divisible by 8)
        num_tokens = 100
        num_kv_heads = 2
        head_dim = 32
        
        k = torch.randn(num_tokens, num_kv_heads, head_dim)
        v = torch.randn(num_tokens, num_kv_heads, head_dim)
        
        selector.compute_landmarks_and_outliers(k, v)
        
        # Should pad to 104 tokens (13 chunks)
        expected_chunks = (num_tokens + selector.chunk_size - 1) // selector.chunk_size
        assert selector.num_chunks == expected_chunks
        assert selector.landmarks.shape[0] == expected_chunks

    def test_select_sparse_chunks(self):
        """Test sparse chunk selection."""
        selector = SparseSelector(chunk_size=8, sparse_budget=8, num_outliers=4)
        
        num_tokens = 128
        num_kv_heads = 4
        head_dim = 64
        num_heads = 16  # GQA: 4 heads per kv head
        
        k = torch.randn(num_tokens, num_kv_heads, head_dim)
        v = torch.randn(num_tokens, num_kv_heads, head_dim)
        
        selector.compute_landmarks_and_outliers(k, v)
        
        # Query for decode
        q = torch.randn(1, num_heads, head_dim)
        scale = 1.0 / (head_dim ** 0.5)
        
        selected = selector.select_sparse_chunks(q, scale)
        
        # Should select sparse_budget chunks (excluding outliers)
        assert selected.shape[0] <= selector.sparse_budget
        # All indices should be valid
        assert (selected >= 0).all()
        assert (selected < selector.num_chunks).all()

    def test_get_outlier_kv(self):
        """Test getting outlier KV cache."""
        selector = SparseSelector(chunk_size=8, sparse_budget=8, num_outliers=4)
        
        k = torch.randn(64, 4, 32)
        v = torch.randn(64, 4, 32)
        
        selector.compute_landmarks_and_outliers(k, v)
        
        outlier_k, outlier_v = selector.get_outlier_kv()
        
        assert outlier_k is not None
        assert outlier_v is not None
        assert outlier_k.shape == outlier_v.shape
        assert selector.get_num_outlier_tokens() == 4 * 8  # num_outliers * chunk_size

    def test_clear(self):
        """Test clearing sparse selector state."""
        selector = SparseSelector(chunk_size=8, sparse_budget=8, num_outliers=4)
        
        k = torch.randn(64, 4, 32)
        v = torch.randn(64, 4, 32)
        
        selector.compute_landmarks_and_outliers(k, v)
        assert selector.landmarks is not None
        
        selector.clear()
        
        assert selector.landmarks is None
        assert selector.outlier_k is None
        assert selector.num_chunks == 0


class TestXKVConfigSparse:
    """Tests for xKVConfig sparse parameters."""

    def test_config_sparse_defaults(self):
        """Test sparse parameter defaults."""
        config = xKVConfig()
        assert config.enable_sparse is False
        assert config.chunk_size == 8
        assert config.sparse_budget == 256
        assert config.num_outliers == 48

    def test_config_sparse_enabled(self):
        """Test config with sparse enabled."""
        config = generate_consecutive_xKV_config(
            num_layers=8,
            end_layer=7,
            enable_sparse=True,
            chunk_size=16,
            sparse_budget=128,
            num_outliers=24,
        )
        assert config.enable_sparse is True
        assert config.chunk_size == 16
        assert config.sparse_budget == 128
        assert config.num_outliers == 24


class TestCacheManagerSparse:
    """Tests for xKVCacheManager sparse integration."""

    def test_manager_sparse_selectors_created(self):
        """Test that sparse selectors are created when enabled."""
        config = generate_consecutive_xKV_config(
            num_layers=4,
            end_layer=3,
            group_size=2,
            enable_sparse=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=4)

        # Should have sparse selectors for all layers in groups
        assert len(manager.sparse_selectors) == 4
        for i in range(4):
            assert i in manager.sparse_selectors

    def test_manager_sparse_not_created_when_disabled(self):
        """Test that sparse selectors are not created when disabled."""
        config = generate_consecutive_xKV_config(
            num_layers=4,
            end_layer=3,
            group_size=2,
            enable_sparse=False,
        )
        manager = xKVCacheManager(config, num_kv_heads=4)

        assert len(manager.sparse_selectors) == 0

    def test_compute_sparse_landmarks(self):
        """Test sparse landmark computation through cache manager."""
        config = generate_consecutive_xKV_config(
            num_layers=4,
            end_layer=3,
            group_size=2,
            enable_sparse=True,
            chunk_size=8,
            sparse_budget=8,
            num_outliers=2,
        )
        manager = xKVCacheManager(config, num_kv_heads=4)

        # Create test KV
        k = torch.randn(64, 4, 32)
        v = torch.randn(64, 4, 32)

        manager.compute_sparse_landmarks(0, k, v)

        selector = manager.get_sparse_selector(0)
        assert selector is not None
        assert selector.landmarks is not None

    def test_get_sparse_stats(self):
        """Test getting sparse statistics."""
        config = generate_consecutive_xKV_config(
            num_layers=4,
            end_layer=3,
            group_size=2,
            enable_sparse=True,
            chunk_size=8,
            sparse_budget=8,
            num_outliers=2,
        )
        manager = xKVCacheManager(config, num_kv_heads=4)

        # Before computing landmarks
        stats = manager.get_sparse_stats()
        assert stats["enabled"] is True
        assert stats["layers_with_landmarks"] == 0

        # After computing landmarks for one layer
        k = torch.randn(64, 4, 32)
        v = torch.randn(64, 4, 32)
        manager.compute_sparse_landmarks(0, k, v)

        stats = manager.get_sparse_stats()
        assert stats["layers_with_landmarks"] == 1
        assert stats["total_outlier_tokens"] == 2 * 8  # num_outliers * chunk_size

    def test_clear_sparse_selectors(self):
        """Test clearing sparse selectors."""
        config = generate_consecutive_xKV_config(
            num_layers=4,
            end_layer=3,
            group_size=2,
            enable_sparse=True,
        )
        manager = xKVCacheManager(config, num_kv_heads=4)

        k = torch.randn(64, 4, 32)
        v = torch.randn(64, 4, 32)
        manager.compute_sparse_landmarks(0, k, v)

        assert manager.get_sparse_selector(0).landmarks is not None

        manager.clear_sparse_selectors()

        assert manager.get_sparse_selector(0).landmarks is None

