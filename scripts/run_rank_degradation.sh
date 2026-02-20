#!/bin/bash
# Run rank degradation benchmark - one config at a time (separate processes)

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TASK="ruler_vt"
MAX_MODEL_LEN=16384
LIMIT=50
OUTPUT_DIR="./results/rank_degradation_vt"

# Use GPU from environment or default to 3 (GPU 0 and 1 are often occupied)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

cd /home/max410011_l/nano-vllm
source .venv/bin/activate

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/rank_degradation_$(date +%Y%m%d_%H%M%S).log"
echo "Starting rank degradation benchmark at $(date)" | tee "$LOG_FILE"

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

# Run each config - from 100% to 12.5% of total KV dim
# Qwen2.5-7B: num_kv_heads=4, head_dim=128, total=512
# rank=512 (100%), 384 (75%), 256 (50%), 128 (25%), 64 (12.5%)
run_config "" "" "baseline"
run_config 512 512 "xKV_k512_v512"
run_config 384 384 "xKV_k384_v384"
run_config 256 256 "xKV_k256_v256"
run_config 128 128 "xKV_k128_v128"
run_config 64 64 "xKV_k64_v64"

echo "============================================" | tee -a "$LOG_FILE"
echo "All benchmarks complete at $(date)" | tee -a "$LOG_FILE"

# Print summary
echo "" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY:" | tee -a "$LOG_FILE"
cat "$OUTPUT_DIR"/*_result.json 2>/dev/null | python3 -c "
import sys, json
print('Config               Rank_K  Rank_V  Accuracy  Time(s)')
print('------------------------------------------------------')
for line in sys.stdin:
    try:
        r = json.loads(line.strip())
        rk = r.get('rank_k') or '-'
        rv = r.get('rank_v') or '-'
        print(f\"{r['config_name']:<20} {str(rk):>6} {str(rv):>6} {r['acc']:>8.4f} {r['elapsed']:>8.1f}\")
    except: pass
" | tee -a "$LOG_FILE"

