#!/bin/bash
# Systematic benchmark of different xKV rank configurations
# Each test runs in a separate process to avoid process group issues

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
LIMIT=100  # Use 100 samples for more reliable results

echo "========================================="
echo "xKV Rank Benchmark - $(date)"
echo "Model: Qwen3-0.6B"
echo "Samples: $LIMIT"
echo "========================================="
echo ""

# Baseline (no xKV)
echo "=== 1/8: Baseline (no xKV) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null 2>&1 | grep "Results:"
echo ""

# Conservative compression
echo "=== 2/8: rank_k=256, rank_v=512 (conservative) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 256 --rank_v 512 2>&1 | grep "Results:"
echo ""

# Moderate compression
echo "=== 3/8: rank_k=128, rank_v=256 (moderate) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 128 --rank_v 256 2>&1 | grep "Results:"
echo ""

# Aggressive compression
echo "=== 4/8: rank_k=64, rank_v=128 (aggressive) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 64 --rank_v 128 2>&1 | grep "Results:"
echo ""

# Very aggressive compression
echo "=== 5/8: rank_k=32, rank_v=64 (very aggressive) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 32 --rank_v 64 2>&1 | grep "Results:"
echo ""

# Extreme compression
echo "=== 6/8: rank_k=16, rank_v=32 (extreme) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 16 --rank_v 32 2>&1 | grep "Results:"
echo ""

# Very extreme compression
echo "=== 7/8: rank_k=8, rank_v=16 (very extreme) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 8 --rank_v 16 2>&1 | grep "Results:"
echo ""

# Ultra extreme compression
echo "=== 8/8: rank_k=4, rank_v=8 (ultra extreme) ==="
python scripts/benchmark_xkv.py --model_path "$MODEL_PATH" --limit $LIMIT --output /dev/null --enable_xkv --rank_k 4 --rank_v 8 2>&1 | grep "Results:"
echo ""

echo "========================================="
echo "Benchmark Complete!"
echo "========================================="

