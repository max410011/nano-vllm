#!/bin/bash
# Benchmark script for RULER/NIAH long-context evaluation with different xKV configurations
# Tests 32k context length with various compression ranks

set -e

source /home/max410011_l/nano-vllm/.venv/bin/activate
cd /home/max410011_l/nano-vllm

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MAX_MODEL_LEN=32768
TASK=${1:-"niah_single_1"}
LIMIT=${2:-20}
OUTPUT_DIR="results/ruler_benchmark"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "RULER/NIAH Long-Context Benchmark"
echo "=============================================="
echo "Model: Qwen3-0.6B"
echo "Task: $TASK"
echo "Max Context Length: $MAX_MODEL_LEN"
echo "Samples per config: $LIMIT"
echo "=============================================="
echo ""

# Function to run benchmark
run_benchmark() {
    local name=$1
    local enable_xkv=$2
    local rank_k=$3
    local rank_v=$4
    local output_file="$OUTPUT_DIR/${TASK}_${name}.json"
    
    echo "Running: $name"
    echo "  Output: $output_file"
    
    local cmd="python scripts/benchmark_long_context.py \
        --model_path $MODEL_PATH \
        --task $TASK \
        --max_model_len $MAX_MODEL_LEN \
        --limit $LIMIT \
        --output $output_file"
    
    if [ "$enable_xkv" = "true" ]; then
        cmd="$cmd --enable_xkv --rank_k $rank_k --rank_v $rank_v"
    fi
    
    eval $cmd 2>&1 | tail -5
    echo ""
}

# Run baseline
echo "========================================="
echo "1. Baseline (no xKV compression)"
echo "========================================="
run_benchmark "baseline" "false" 0 0

# Run xKV with different ranks
echo "========================================="
echo "2. xKV: rank_k=128, rank_v=256 (moderate)"
echo "========================================="
run_benchmark "rank_k128" "true" 128 256

echo "========================================="
echo "3. xKV: rank_k=64, rank_v=128 (aggressive)"
echo "========================================="
run_benchmark "rank_k64" "true" 64 128

echo "========================================="
echo "4. xKV: rank_k=32, rank_v=64 (very aggressive)"
echo "========================================="
run_benchmark "rank_k32" "true" 32 64

echo "========================================="
echo "5. xKV: rank_k=16, rank_v=32 (extreme)"
echo "========================================="
run_benchmark "rank_k16" "true" 16 32

echo "========================================="
echo "6. xKV: rank_k=8, rank_v=16 (very extreme)"
echo "========================================="
run_benchmark "rank_k8" "true" 8 16

# Summary
echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
for f in $OUTPUT_DIR/${TASK}_*.json; do
    if [ -f "$f" ]; then
        config=$(basename "$f" .json | sed "s/${TASK}_//")
        metrics=$(python -c "import json; d=json.load(open('$f')); print(f'  {\"$config\":>15}: acc={d[\"metrics\"].get(\"${MAX_MODEL_LEN},none\", \"N/A\")}, throughput={d[\"throughput\"]:.4f} s/s')" 2>/dev/null || echo "  $config: parse error")
        echo "$metrics"
    fi
done

