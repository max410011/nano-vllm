#!/bin/bash
# Run group_size benchmark - test different gs with scaled rank
# Base: rank=128 at gs=1, scale rank proportionally for larger gs

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TASK="ruler_vt"
MAX_MODEL_LEN=16384
LIMIT=50
OUTPUT_DIR="./results/gs_benchmark"

# Use GPU from environment or default to 3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

cd /home/max410011_l/nano-vllm
source .venv/bin/activate

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/gs_benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "Starting group_size benchmark at $(date)" | tee "$LOG_FILE"
echo "Testing: gs=1 rank=128, gs=2 rank=256, gs=4 rank=512" | tee -a "$LOG_FILE"

run_config() {
    local config_name=$1
    local rank_k=$2
    local rank_v=$3
    local group_size=$4
    
    echo "" | tee -a "$LOG_FILE"
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
        --rank_k "$rank_k" \
        --rank_v "$rank_v" \
        --group_size "$group_size" \
        2>&1 | tee -a "$LOG_FILE"
}

# gs=1, rank=128 (base configuration - each layer compressed independently)
run_config "gs1_r128" 128 128 1

# gs=2, rank=256 (2 layers share same compression, 2x rank to compensate)
run_config "gs2_r256" 256 256 2

# gs=4, rank=512 (4 layers share same compression, 4x rank to compensate)
run_config "gs4_r512" 512 512 4

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "All benchmarks complete at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY:" | tee -a "$LOG_FILE"
echo "Config               GS  Rank_K  Rank_V  Accuracy  Time(s)" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

# Print results from JSON files
for result_file in "$OUTPUT_DIR"/*_result.json; do
    if [ -f "$result_file" ]; then
        python3 -c "
import json
with open('$result_file') as f:
    r = json.load(f)
gs = r.get('group_size', 1)
print(f\"{r['config_name']:<20} {gs:>2} {r['rank_k']:>6} {r['rank_v']:>6} {r['acc']:>8.4f} {r['elapsed']:>8.1f}\")
" | tee -a "$LOG_FILE"
    fi
done

