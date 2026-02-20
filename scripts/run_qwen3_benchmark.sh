#!/bin/bash
# Run xKV benchmark on Qwen3-8B
# KV dim = num_kv_heads(8) × head_dim(128) = 1024

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
TASK="ruler_vt"
MAX_MODEL_LEN=16384
LIMIT=50
OUTPUT_DIR="./results/qwen3_8b_benchmark"

# Use GPU from environment or default to 0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/max410011_l/nano-vllm
source .venv/bin/activate

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Qwen3-8B benchmark at $(date)" | tee "$LOG_FILE"
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "KV dim = 8 × 128 = 1024" | tee -a "$LOG_FILE"

run_config() {
    local rank_k=$1
    local rank_v=$2
    local config_name=$3

    echo "============================================" | tee -a "$LOG_FILE"
    echo "Running: $config_name at $(date)" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"

    python scripts/run_single_rank_config.py \
        --model_path "$MODEL_PATH" \
        --task "$TASK" \
        --max_model_len "$MAX_MODEL_LEN" \
        --limit "$LIMIT" \
        --output_dir "$OUTPUT_DIR" \
        --config_name "$config_name" \
        ${rank_k:+--rank_k $rank_k} \
        ${rank_v:+--rank_v $rank_v} \
        2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    sleep 3
}

# Qwen3-8B: num_kv_heads=8, head_dim=128, total KV dim=1024
# rank=1024 (100%), 768 (75%), 512 (50%), 256 (25%), 128 (12.5%)
run_config "" "" "baseline"
run_config 1024 1024 "xKV_k1024_v1024"
run_config 768 768 "xKV_k768_v768"
run_config 512 512 "xKV_k512_v512"
run_config 256 256 "xKV_k256_v256"
run_config 128 128 "xKV_k128_v128"

echo "============================================" | tee -a "$LOG_FILE"
echo "All benchmarks complete at $(date)" | tee -a "$LOG_FILE"

# Print summary
echo "" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY (Qwen3-8B, KV dim=1024):" | tee -a "$LOG_FILE"
echo "Config               Rank_K  Rank_V  Compress  Accuracy  Time(s)" | tee -a "$LOG_FILE"
echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
cat "$OUTPUT_DIR"/*_result.json 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        r = json.loads(line.strip())
        rk = r.get('rank_k') or '-'
        rv = r.get('rank_v') or '-'
        if rk != '-':
            compress = f'{int(rk)/1024*100:.1f}%'
        else:
            compress = '-'
        print(f\"{r['config_name']:<20} {str(rk):>6} {str(rv):>6} {compress:>8} {r['acc']:>8.4f} {r['elapsed']:>8.1f}\")
    except: pass
" | tee -a "$LOG_FILE"

