#!/bin/bash
# Run xKV sparse benchmark - one config at a time to avoid process group issues

MODEL_PATH="/share2/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TASK="ruler_vt"
MAX_MODEL_LEN=32768
LIMIT=50
OUTPUT_DIR="./results/xkv_sparse_benchmark"

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Configs to run
CONFIGS=(
    "xK-SR-4_k384_sparse-2048"
    "xKV-SR-1_k96_v144_sparse-2048"
    "xKV-SR-4_k384_v576_sparse-2048"
)

cd /home/max410011_l/nano-vllm
source .venv/bin/activate

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/benchmark_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Starting benchmark at $(date)" | tee "$LOG_FILE"
echo "Configs: ${CONFIGS[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for config in "${CONFIGS[@]}"; do
    echo "============================================" | tee -a "$LOG_FILE"
    echo "Running: $config at $(date)" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    
    python scripts/benchmark_xkv_sparse.py \
        --model_path "$MODEL_PATH" \
        --task "$TASK" \
        --max_model_len "$MAX_MODEL_LEN" \
        --limit "$LIMIT" \
        --configs "$config" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo "Finished $config at $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Small delay between runs
    sleep 5
done

echo "============================================" | tee -a "$LOG_FILE"
echo "All benchmarks complete at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"

