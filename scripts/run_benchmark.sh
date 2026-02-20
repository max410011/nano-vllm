#!/bin/bash
# Benchmark script for different xKV configurations
# Each configuration runs in a separate process to avoid process group conflicts

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
LIMIT=100
RESULTS_DIR="results/benchmark"

mkdir -p $RESULTS_DIR

echo "=============================================="
echo "xKV Benchmark Suite"
echo "Model: Qwen3-0.6B"
echo "Samples per config: $LIMIT"
echo "=============================================="

# Baseline (no xKV)
echo -e "\n[1/9] Running baseline..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --output $RESULTS_DIR/baseline.json

# Different ranks (no sparse)
echo -e "\n[2/9] Running rank_k=64, rank_v=128..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 64 --rank_v 128 \
    --output $RESULTS_DIR/rank_64_128.json

echo -e "\n[3/9] Running rank_k=128, rank_v=256..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 128 --rank_v 256 \
    --output $RESULTS_DIR/rank_128_256.json

echo -e "\n[4/9] Running rank_k=256, rank_v=512..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 256 --rank_v 512 \
    --output $RESULTS_DIR/rank_256_512.json

echo -e "\n[5/9] Running rank_k=512, rank_v=768..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 512 --rank_v 768 \
    --output $RESULTS_DIR/rank_512_768.json

# Sparse with different budgets (using rank_k=128, rank_v=256)
echo -e "\n[6/9] Running sparse_budget=64..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 128 --rank_v 256 --enable_sparse --sparse_budget 64 \
    --output $RESULTS_DIR/sparse_64.json

echo -e "\n[7/9] Running sparse_budget=128..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 128 --rank_v 256 --enable_sparse --sparse_budget 128 \
    --output $RESULTS_DIR/sparse_128.json

echo -e "\n[8/9] Running sparse_budget=256..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 128 --rank_v 256 --enable_sparse --sparse_budget 256 \
    --output $RESULTS_DIR/sparse_256.json

echo -e "\n[9/9] Running sparse_budget=512..."
python scripts/benchmark_xkv.py --model_path $MODEL_PATH --limit $LIMIT \
    --enable_xkv --rank_k 128 --rank_v 256 --enable_sparse --sparse_budget 512 \
    --output $RESULTS_DIR/sparse_512.json

# Aggregate results
echo -e "\n=============================================="
echo "Aggregating results..."
echo "=============================================="

python -c "
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
all_results = []

for f in sorted(results_dir.glob('*.json')):
    with open(f) as fp:
        all_results.append(json.load(fp))

# Save combined results
with open(results_dir / 'all_results.json', 'w') as fp:
    json.dump(all_results, fp, indent=2)

# Print summary table
print()
print('='*100)
print('BENCHMARK SUMMARY')
print('='*100)
print(f\"{'Configuration':<55} {'acc':>8} {'acc_norm':>10} {'throughput':>12} {'time':>8}\")
print('-'*100)
for r in all_results:
    print(f\"{r['config_name']:<55} {r['acc']:>8.4f} {r['acc_norm']:>10.4f} {r['throughput']:>10.2f}/s {r['total_time']:>7.1f}s\")
print('='*100)
"

echo -e "\nAll results saved to: $RESULTS_DIR/all_results.json"

