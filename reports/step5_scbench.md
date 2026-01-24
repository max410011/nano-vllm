# Step 5: SCBench Integration Report

## Overview

This step integrates MInference's SCBench (Shared Context Benchmark) into nano-vllm using a git submodule approach. SCBench evaluates multi-turn long-context performance by testing KV cache reuse efficiency.

## Integration Approach

### Git Submodule
- MInference repository added as submodule at `third_party/MInference`
- SCBench modules imported from `third_party/MInference/scbench/`
- No code duplication - directly uses MInference's evaluation logic

### Key Components

1. **`nanovllm/eval/scbench.py`** - Main wrapper module
   - `NanoVLLMSCBench` class - Main evaluator class
   - `SCBenchConfig` - Configuration dataclass
   - `run_scbench()` - Convenience function
   - `_get_scbench_modules()` - Lazy loader for MInference modules

2. **`scripts/run_scbench.py`** - Command-line interface

3. **`tests/test_step5_scbench.py`** - Integration tests (12 tests)

## Supported Tasks

| Task | Description |
|------|-------------|
| `scbench_kv` | Key-value retrieval from JSON |
| `scbench_kv_hard` | Hard key-value retrieval |
| `scbench_passkey` | Passkey retrieval |
| `scbench_qa_eng` | English QA |
| `scbench_qa_chn` | Chinese QA |
| `scbench_choice_eng` | Multiple choice (English) |
| `scbench_mf` | Math find |
| `scbench_repoqa` | Repository QA |
| `scbench_summary` | Summarization |
| `scbench_vt` | Variable tracking |
| `scbench_many_shot` | Many-shot learning |
| `scbench_summary_with_needles` | Summary with needle retrieval |
| `scbench_repoqa_and_kv` | Combined RepoQA and KV |
| `scbench_prefix_suffix` | Prefix/suffix matching |

## Evaluation Modes

### SCDQ Mode (Same-Context-Different-Query)
- Context encoded once, reused for all queries
- Tests KV cache reuse efficiency
- Default mode for evaluation

### Multi-Turn Mode
- Each turn builds on previous conversation
- Tests conversational context handling

## Usage

### Python API
```python
from nanovllm.eval.scbench import NanoVLLMSCBench, SCBenchConfig

config = SCBenchConfig(scdq_mode=True)
evaluator = NanoVLLMSCBench(
    pretrained="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=32768,
    config=config,
)

results = evaluator.evaluate("scbench_kv", limit=10)
```

### Command Line
```bash
python scripts/run_scbench.py \
    --model_path /path/to/model \
    --task scbench_kv \
    --max_model_len 32768 \
    --limit 10 \
    --scdq_mode
```

### With xKV Compression
```bash
python scripts/run_scbench.py \
    --model_path /path/to/model \
    --task scbench_kv \
    --enable_xkv \
    --xkv_rank_k 64 \
    --xkv_rank_v 64
```

## Dependencies

Required packages (installed via `uv pip install`):
- `jieba` - Chinese tokenization
- `rouge` - ROUGE scoring
- `evaluate` - HuggingFace evaluate library

## Test Results

```
12 passed in 8.96s
```

### Test Coverage
- Module imports
- MInference submodule existence
- SCBench modules loadable
- Configuration defaults
- Task list validation
- Task-MInference mapping

## Design Patterns

1. **Adapter Pattern**: Wraps nano-vllm's LLM for SCBench interface
2. **Facade Pattern**: Provides simplified interface via `run_scbench()`
3. **Lazy Import Pattern**: Deferred loading of MInference modules to avoid heavy imports

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `nanovllm/eval/scbench.py` | Created | ~510 |
| `nanovllm/eval/__init__.py` | Modified | 19 |
| `scripts/run_scbench.py` | Created | 164 |
| `tests/test_step5_scbench.py` | Created | 175 |
| `reports/step5_scbench.md` | Created | This file |
| `.gitmodules` | Created | Submodule config |
| `third_party/MInference/` | Added | Submodule |

## Commit Information

Branch: `main`
Commit message: "Step 5: Integrate SCBench using MInference submodule"

Features added:
- Add MInference as git submodule at third_party/MInference
- Create NanoVLLMSCBench wrapper class for SCBench evaluations
- Support 14 SCBench tasks directly from MInference
- Implement SCDQ and multi-turn evaluation modes
- Add run_scbench.py command-line script
- Add integration tests (12 tests passed)
- Compatible with xKV compression
- Use lazy imports to avoid minference package dependency

