#!/usr/bin/env python3
"""
Benchmark script for xKV and xKV-sparse variants on RULER/VT 64k.

Tests 6 configurations:
1. xKV_1_k96_v144 - group_size=1, rank_k=96, rank_v=144
2. xKV_4_k384_v576 - group_size=4, rank_k=384, rank_v=576
3. xK-SR-1_rank_k-96_sparse-2048 - xK-SR mode, group_size=1 (basically ShadowKV)
4. xK-SR-4_rank_k-384_sparse-2048 - xK-SR mode, group_size=4
5. xKV-SR-1_k96_v144_sparse-2048 - xKV-SR mode, group_size=1
6. xKV-SR-4_k384_v576_sparse-2048 - xKV-SR mode, group_size=4
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch

from nanovllm.xkv import generate_consecutive_xKV_config


def get_configurations():
    """Define all 6 test configurations."""
    configs = {
        # Pure xKV (no sparse)
        "xKV_1_k96_v144": {
            "group_size": 1,
            "rank_k": 96,
            "rank_v": 144,
            "enable_sparse": False,
            "sparse_mode": None,
            "sparse_budget": None,
        },
        "xKV_4_k384_v576": {
            "group_size": 4,
            "rank_k": 384,
            "rank_v": 576,
            "enable_sparse": False,
            "sparse_mode": None,
            "sparse_budget": None,
        },
        # xK-SR (K uses SVD, V offloaded - original ShadowKV style)
        "xK-SR-1_k96_sparse-2048": {
            "group_size": 1,
            "rank_k": 96,
            "rank_v": None,  # V is offloaded, not SVD
            "enable_sparse": True,
            "sparse_mode": "xk_sr",
            "sparse_budget": 2048,
        },
        "xK-SR-4_k384_sparse-2048": {
            "group_size": 4,
            "rank_k": 384,
            "rank_v": None,  # V is offloaded, not SVD
            "enable_sparse": True,
            "sparse_mode": "xk_sr",
            "sparse_budget": 2048,
        },
        # xKV-SR (both K and V use SVD + sparse)
        "xKV-SR-1_k96_v144_sparse-2048": {
            "group_size": 1,
            "rank_k": 96,
            "rank_v": 144,
            "enable_sparse": True,
            "sparse_mode": "xkv_sr",
            "sparse_budget": 2048,
        },
        "xKV-SR-4_k384_v576_sparse-2048": {
            "group_size": 4,
            "rank_k": 384,
            "rank_v": 576,
            "enable_sparse": True,
            "sparse_mode": "xkv_sr",
            "sparse_budget": 2048,
        },
    }
    return configs


def create_xkv_config(config_name: str, config: dict, num_layers: int = 28):
    """Create xKVConfig from configuration dict."""
    if not config["enable_sparse"]:
        return generate_consecutive_xKV_config(
            num_layers=num_layers,
            end_layer=-1,  # Use num_layers - 1
            group_size=config["group_size"],
            rank_k=config["rank_k"],
            rank_v=config["rank_v"],
            paged_writeback=True,
            enable_sparse=False,
        )
    else:
        return generate_consecutive_xKV_config(
            num_layers=num_layers,
            end_layer=-1,  # Use num_layers - 1
            group_size=config["group_size"],
            rank_k=config["rank_k"],
            rank_v=config["rank_v"],
            paged_writeback=True,
            enable_sparse=True,
            sparse_mode=config["sparse_mode"],
            sparse_budget=config["sparse_budget"],
            chunk_size=8,
            num_outliers=48,
        )


def run_ruler_vt_evaluation(
    model_path: str,
    config_name: str,
    xkv_config,
    task_name: str = "ruler_vt",
    max_model_len: int = 65536,
    limit: int = None,
):
    """Run RULER/VT evaluation with given xKV config."""
    from nanovllm.eval.lm_harness import NanoVLLMHarness
    from lm_eval import simple_evaluate

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Task: {task_name}, max_model_len: {max_model_len}")
    print(f"{'='*60}")

    # Create model with xKV config
    harness = NanoVLLMHarness(
        pretrained=model_path,
        max_model_len=max_model_len,
        enable_xkv=True,
        xkv_config=xkv_config,
    )

    start_time = time.time()

    # Run evaluation
    eval_kwargs = {
        "model": harness,
        "tasks": [task_name],
        "num_fewshot": 0,
    }
    if limit is not None:
        eval_kwargs["limit"] = limit

    # For NIAH/RULER tasks, pass metadata with tokenizer and sequence lengths
    if 'niah' in task_name.lower() or 'ruler' in task_name.lower():
        eval_kwargs["metadata"] = {
            "pretrained": model_path,
            "max_seq_lengths": [max_model_len],
        }

    results = simple_evaluate(**eval_kwargs)

    elapsed = time.time() - start_time

    # Extract accuracy
    task_results = results.get("results", {}).get(task_name, {})
    acc = task_results.get("acc,none", task_results.get("acc", 0))
    acc_norm = task_results.get("acc_norm,none", task_results.get("acc_norm", 0))

    # Cleanup: must destroy process group to allow re-initialization
    del harness
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Destroy distributed process group if initialized
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    return {
        "config_name": config_name,
        "task": task_name,
        "max_model_len": max_model_len,
        "acc": acc,
        "acc_norm": acc_norm,
        "elapsed_seconds": elapsed,
        "samples_per_second": (limit or 100) / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark xKV and xKV-sparse on RULER/VT 64k")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--output_dir", type=str, default="./results/xkv_sparse_benchmark")
    parser.add_argument("--task", type=str, default="ruler_vt", help="Task to evaluate")
    parser.add_argument("--max_model_len", type=int, default=65536)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per config")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_configs = get_configurations()

    # Filter configs if specified
    if args.configs:
        configs_to_run = {k: v for k, v in all_configs.items() if k in args.configs}
    else:
        configs_to_run = all_configs

    results = []

    for config_name, config in configs_to_run.items():
        try:
            xkv_config = create_xkv_config(config_name, config)
            result = run_ruler_vt_evaluation(
                model_path=args.model_path,
                config_name=config_name,
                xkv_config=xkv_config,
                task_name=args.task,
                max_model_len=args.max_model_len,
                limit=args.limit,
            )
            result["config"] = config
            results.append(result)

            print(f"\n{config_name}:")
            print(f"  acc: {result['acc']:.4f}")
            if result['acc_norm']:
                print(f"  acc_norm: {result['acc_norm']:.4f}")
            print(f"  time: {result['elapsed_seconds']:.1f}s")

        except Exception as e:
            import traceback
            print(f"Error running {config_name}: {e}")
            traceback.print_exc()
            results.append({
                "config_name": config_name,
                "error": str(e),
                "config": config,
            })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Config':<40} {'Acc':>8} {'Acc_Norm':>10}")
    print("-"*60)
    for r in results:
        if "error" not in r:
            print(f"{r['config_name']:<40} {r['acc']:>8.4f} {r['acc_norm']:>10.4f}")
        else:
            print(f"{r['config_name']:<40} {'ERROR':>8} {'-':>10}")

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    main()

