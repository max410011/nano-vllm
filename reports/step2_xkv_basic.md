# Step 2: xKV Basic Integration Report

## Overview

Step 2 integrates xKV (Cross-layer KV-Cache Compression) into nano-vllm without paged attention writeback. This phase implements the core SVD and SLERP compression algorithms and integrates them into the model's forward pass during prefill.

## Design Architecture

### Module Structure

```
nanovllm/xkv/
├── __init__.py      # Public exports
├── config.py        # xKVConfig, LayerGroup classes
├── compressor.py    # fake_svd, fake_minicache_merge functions
└── cache_manager.py # xKVCacheManager class
```

### Design Patterns Used

1. **Strategy Pattern**: `xKVConfig.layer_merge_impl` allows switching between SVD and SLERP compression
2. **Builder Pattern**: `generate_consecutive_xKV_config()` helper for easy configuration
3. **Observer Pattern**: `xKVCacheManager` observes layer indices to trigger compression at group boundaries

### Core Components

#### 1. Compression Functions (`compressor.py`)

| Function | Description | Use Case |
|----------|-------------|----------|
| `fake_svd(tensor, rank)` | SVD decomposition with rank truncation | General compression for any group size |
| `slerp_merge_rows_batch(X1, X2, t, gamma)` | SLERP interpolation | Directional merging |
| `fake_minicache_merge(X1, X2, t, gamma)` | SLERP-based merge | Group size = 2 only |

#### 2. Configuration (`config.py`)

```python
@dataclass
class xKVConfig:
    num_layers: Optional[int] = None
    layer_merge_impl: str = "svd"  # "svd" or "slerp"
    rank_k: Optional[int] = None   # SVD rank for keys
    rank_v: Optional[int] = None   # SVD rank for values
    slerp_t: float = 0.5           # SLERP interpolation parameter
    slerp_gamma: float = 1.0       # SLERP divergence threshold
    merge_key: bool = True
    merge_value: bool = True
    layer_groups: List[LayerGroup] = field(default_factory=list)
    paged_writeback: bool = False  # For Step 3
```

#### 3. Cache Manager (`cache_manager.py`)

The `xKVCacheManager` orchestrates the compression flow:

1. **Store phase**: Captures pre-RoPE K, V tensors during prefill
2. **Compress phase**: At group boundary, applies compression
3. **Retrieve phase**: Returns compressed K, V for attention

### Integration Points

#### Modified Files

| File | Changes |
|------|---------|
| `nanovllm/config.py` | Added `enable_xkv`, `xkv_config` fields |
| `nanovllm/models/qwen3.py` | Added `layer_idx` to attention, xKV manager flow |
| `nanovllm/engine/model_runner.py` | Initialize/pass `xkv_manager` |
| `nanovllm/eval/lm_harness.py` | Added xKV parameters |
| `scripts/eval_lm_harness.py` | Added CLI options for xKV |

## Evaluation Results

### Test Environment
- Model: Qwen3-0.6B (28 layers)
- Task: HellaSwag (50 samples)
- Hardware: NVIDIA GPU with CUDA 12.4

### Accuracy Comparison

| Configuration | acc | acc_norm | Throughput |
|---------------|-----|----------|------------|
| **Baseline (no xKV)** | 0.40 | 0.54 | ~27 it/s |
| **xKV (rank_k=256, rank_v=768)** | 0.40 | 0.54 | ~26 it/s |
| **xKV (rank_k=128, rank_v=256)** | 0.40 | 0.54 | ~26 it/s |

### Key Findings

1. **No accuracy degradation**: Both conservative and aggressive compression settings maintain baseline accuracy
2. **Minimal throughput impact**: ~4% overhead during prefill due to SVD computation
3. **SVD convergence warning**: CuSolver occasionally falls back to more accurate method (acceptable)

## Unit Tests

All 15 tests passing:

```
tests/test_step2_xkv_basic.py::TestSVDCompression::test_svd_output_shape PASSED
tests/test_step2_xkv_basic.py::TestSVDCompression::test_svd_compression_effect PASSED
tests/test_step2_xkv_basic.py::TestSVDCompression::test_svd_low_rank_more_compression PASSED
tests/test_step2_xkv_basic.py::TestSLERPCompression::test_slerp_output_shape PASSED
tests/test_step2_xkv_basic.py::TestSLERPCompression::test_slerp_t_parameter_effect PASSED
tests/test_step2_xkv_basic.py::TestXKVConfig::test_config_creation PASSED
tests/test_step2_xkv_basic.py::TestXKVConfig::test_consecutive_config_generation PASSED
tests/test_step2_xkv_basic.py::TestXKVConfig::test_layer_map_building PASSED
tests/test_step2_xkv_basic.py::TestXKVConfig::test_invalid_layer_merge_impl PASSED
tests/test_step2_xkv_basic.py::TestXKVCacheManager::test_cache_manager_creation PASSED
tests/test_step2_xkv_basic.py::TestXKVCacheManager::test_should_store_temp PASSED
tests/test_step2_xkv_basic.py::TestXKVCacheManager::test_should_compress PASSED
tests/test_step2_xkv_basic.py::TestXKVCacheManager::test_store_and_retrieve_temp_kv PASSED
tests/test_step2_xkv_basic.py::TestXKVCacheManager::test_svd_compression_flow PASSED
tests/test_step2_xkv_basic.py::TestXKVCacheManager::test_clear_all PASSED
```

## Code Quality Assessment

### Strengths
- Clean separation of concerns (config, compression, management)
- Type hints throughout
- Comprehensive docstrings
- Minimal changes to existing code

### Areas for Future Improvement
- Consider caching SVD results for repeated sequences
- Add compression ratio logging
- Implement async compression for overlapping with computation

## Usage Example

```python
from nanovllm import LLM
from nanovllm.xkv import generate_consecutive_xKV_config

# Create xKV config
xkv_config = generate_consecutive_xKV_config(
    layer_merge_impl="svd",
    start_layer=0,
    end_layer=27,
    num_layers=28,
    group_size=2,
    rank_k=256,
    rank_v=768,
)

# Initialize LLM with xKV
llm = LLM(
    "/path/to/model",
    enable_xkv=True,
    xkv_config=xkv_config,
)
```

## Next Steps (Step 3)

1. Implement paged attention writeback for compressed KV
2. Integrate with existing KV cache block management
3. Test with longer sequences to measure memory savings

