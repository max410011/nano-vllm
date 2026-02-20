#!/usr/bin/env python3
"""Run a single rank configuration for benchmark."""

import argparse
import json
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--rank_k", type=int, default=None)
    parser.add_argument("--rank_v", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=1)
    args = parser.parse_args()

    from nanovllm.xkv import generate_consecutive_xKV_config
    from nanovllm.eval.lm_harness import NanoVLLMHarness
    from lm_eval import simple_evaluate

    # Create xKV config if rank specified
    xkv_config = None
    enable_xkv = False
    if args.rank_k is not None:
        enable_xkv = True
        xkv_config = generate_consecutive_xKV_config(
            num_layers=28,
            group_size=args.group_size,
            start_layer=0,
            end_layer=-1,
            rank_k=args.rank_k,
            rank_v=args.rank_v,
        )
    
    harness = NanoVLLMHarness(
        pretrained=args.model_path,
        max_model_len=args.max_model_len,
        enable_xkv=enable_xkv,
        xkv_config=xkv_config,
    )
    
    start = time.time()
    results = simple_evaluate(
        model=harness,
        tasks=[args.task],
        num_fewshot=0,
        limit=args.limit,
        metadata={'pretrained': args.model_path, 'max_seq_lengths': [args.max_model_len]},
    )
    elapsed = time.time() - start
    
    task_results = results.get('results', {}).get(args.task, {})
    # RULER tasks use context length as key, e.g., "16384,none"
    acc = task_results.get(f'{args.max_model_len},none',
                           task_results.get('acc,none',
                           task_results.get('acc', 0)))

    print(f'RESULT: {args.config_name} | rank_k={args.rank_k} | rank_v={args.rank_v} | gs={args.group_size} | acc={acc:.4f} | time={elapsed:.1f}s')
    print(f'Full task_results: {task_results}')

    # Save result
    output_file = f"{args.output_dir}/{args.config_name}_result.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config_name': args.config_name,
            'rank_k': args.rank_k,
            'rank_v': args.rank_v,
            'group_size': args.group_size,
            'acc': acc,
            'elapsed': elapsed,
        }, f)

if __name__ == "__main__":
    main()

