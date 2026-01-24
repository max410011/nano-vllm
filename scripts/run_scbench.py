#!/usr/bin/env python3
"""
Command-line script to run SCBench evaluation with nano-vllm.

This script wraps MInference's SCBench benchmark for evaluating
multi-turn long-context performance with KV cache reuse.

Usage:
    python scripts/run_scbench.py \
        --model_path /path/to/model \
        --task scbench_kv \
        --output_dir ./results/scbench \
        --max_model_len 32768 \
        --limit 10

Reference:
    https://github.com/microsoft/MInference/tree/main/scbench
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nanovllm.eval.scbench import (
    NanoVLLMSCBench,
    SCBenchConfig,
    SCBENCH_TASKS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SCBench evaluation with nano-vllm"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="Maximum context length (default: 32768)"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism"
    )
    
    # Task arguments
    parser.add_argument(
        "--task", type=str, required=True,
        choices=SCBENCH_TASKS,
        help=f"SCBench task to evaluate. Available: {SCBENCH_TASKS}"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/scbench",
        help="Output directory for results"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--max_turns", type=int, default=-1,
        help="Maximum turns per example (-1 for all)"
    )
    
    # Mode arguments
    parser.add_argument(
        "--scdq_mode", action="store_true", default=True,
        help="Use SCDQ (Same-Context-Different-Query) mode (default: True)"
    )
    parser.add_argument(
        "--multi_turn_mode", action="store_true",
        help="Use multi-turn mode instead of SCDQ"
    )
    parser.add_argument(
        "--no_chat_template", action="store_true",
        help="Disable chat template"
    )
    parser.add_argument(
        "--rewrite", action="store_true",
        help="Overwrite existing results"
    )
    
    # xKV arguments
    parser.add_argument(
        "--enable_xkv", action="store_true",
        help="Enable xKV compression"
    )
    parser.add_argument(
        "--xkv_rank_k", type=int, default=64,
        help="xKV rank for K cache"
    )
    parser.add_argument(
        "--xkv_rank_v", type=int, default=64,
        help="xKV rank for V cache"
    )
    parser.add_argument(
        "--xkv_group_size", type=int, default=2,
        help="xKV group size"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine mode
    scdq_mode = not args.multi_turn_mode
    
    # Setup config
    config = SCBenchConfig(
        use_chat_template=not args.no_chat_template,
        scdq_mode=scdq_mode,
        disable_golden_context=True,
    )
    
    # Setup xKV config if enabled
    xkv_config = None
    if args.enable_xkv:
        from nanovllm.xkv import XKVConfig
        xkv_config = XKVConfig(
            rank_k=args.xkv_rank_k,
            rank_v=args.xkv_rank_v,
            group_size=args.xkv_group_size,
        )
    
    print(f"=== SCBench Evaluation ===")
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Max model len: {args.max_model_len}")
    print(f"SCDQ mode: {scdq_mode}")
    print(f"xKV enabled: {args.enable_xkv}")
    if args.enable_xkv:
        print(f"  rank_k: {args.xkv_rank_k}, rank_v: {args.xkv_rank_v}")
    print()
    
    # Initialize evaluator
    evaluator = NanoVLLMSCBench(
        pretrained=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_xkv=args.enable_xkv,
        xkv_config=xkv_config,
        config=config,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        task_name=args.task,
        output_dir=args.output_dir,
        limit=args.limit,
        max_turns=args.max_turns,
        rewrite=args.rewrite,
    )
    
    # Print results
    print("\n=== Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

