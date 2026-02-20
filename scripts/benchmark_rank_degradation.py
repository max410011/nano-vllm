#!/usr/bin/env python3
"""Benchmark xKV rank degradation on RULER/VT"""

import argparse
import json
import time
import torch
from datetime import datetime
from pathlib import Path

def run_single_config(model_path, rank_k, rank_v, task_name, max_model_len, limit, group_size=1):
    """Run a single configuration and return results."""
    from nanovllm.xkv import xKVConfig, generate_consecutive_xKV_config
    from nanovllm.eval.lm_harness import NanoVLLMHarness
    from lm_eval import simple_evaluate
    
    config_name = f"xKV_k{rank_k}_v{rank_v}" if rank_k else "baseline"
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Task: {task_name}, max_model_len: {max_model_len}")
    print(f"{'='*60}")
    
    # Create xKV config if not baseline
    xkv_config = None
    enable_xkv = False
    if rank_k is not None:
        enable_xkv = True
        xkv_config = generate_consecutive_xKV_config(
            num_layers=28,
            group_size=group_size,
            start_layer=0,
            end_layer=-1,
            rank_k=rank_k,
            rank_v=rank_v,
        )
    
    harness = NanoVLLMHarness(
        pretrained=model_path,
        max_model_len=max_model_len,
        enable_xkv=enable_xkv,
        xkv_config=xkv_config,
    )
    
    start_time = time.time()
    
    eval_kwargs = {
        "model": harness,
        "tasks": [task_name],
        "num_fewshot": 0,
    }
    if limit:
        eval_kwargs["limit"] = limit
    
    if 'ruler' in task_name.lower() or 'niah' in task_name.lower():
        eval_kwargs["metadata"] = {
            "pretrained": model_path,
            "max_seq_lengths": [max_model_len],
        }
    
    results = simple_evaluate(**eval_kwargs)
    elapsed = time.time() - start_time
    
    task_results = results.get("results", {}).get(task_name, {})
    acc = task_results.get("acc,none", task_results.get("acc", 0))
    
    # Cleanup
    del harness
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return {
        "config_name": config_name,
        "rank_k": rank_k,
        "rank_v": rank_v,
        "acc": acc,
        "elapsed_seconds": elapsed,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="ruler_vt")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./results/rank_degradation")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations: baseline + various ranks
    configs = [
        (None, None),      # Baseline
        (128, 192),        # Conservative
        (96, 144),         # Medium
        (64, 96),          # Moderate
        (48, 72),          # Strong
        (32, 48),          # Very strong
    ]
    
    results = []
    for rank_k, rank_v in configs:
        try:
            result = run_single_config(
                model_path=args.model_path,
                rank_k=rank_k,
                rank_v=rank_v,
                task_name=args.task,
                max_model_len=args.max_model_len,
                limit=args.limit,
            )
            results.append(result)
            print(f"\n{result['config_name']}: acc={result['acc']:.4f}, time={result['elapsed_seconds']:.1f}s")
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            results.append({"config_name": f"xKV_k{rank_k}_v{rank_v}", "error": str(e)})
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"rank_degradation_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({"task": args.task, "max_model_len": args.max_model_len, "results": results}, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RANK DEGRADATION SUMMARY")
    print("="*60)
    print(f"{'Config':<20} {'Rank_K':>8} {'Rank_V':>8} {'Accuracy':>10} {'Time(s)':>10}")
    print("-"*60)
    for r in results:
        if "error" not in r:
            rk = r['rank_k'] if r['rank_k'] else '-'
            rv = r['rank_v'] if r['rank_v'] else '-'
            print(f"{r['config_name']:<20} {str(rk):>8} {str(rv):>8} {r['acc']:>10.4f} {r['elapsed_seconds']:>10.1f}")
        else:
            print(f"{r['config_name']:<20} {'ERROR':>8}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
