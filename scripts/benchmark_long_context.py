#!/usr/bin/env python3
"""
Long-context benchmark script for xKV configurations.

Tests RULER and NIAH (Needle in a Haystack) benchmarks at different context lengths.

Usage:
    python scripts/benchmark_long_context.py --model_path /path/to/model --task ruler --max_model_len 32768
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from lm_eval import simple_evaluate
from nanovllm.eval import NanoVLLMHarness
from nanovllm.xkv import generate_consecutive_xKV_config
from transformers import AutoConfig


@dataclass
class LongContextBenchmarkResult:
    config_name: str
    task: str
    max_model_len: int
    rank_k: int
    rank_v: int
    enable_xkv: bool
    metrics: dict
    throughput: float
    total_time: float


def run_long_context_benchmark(
    model_path: str,
    task: str = "ruler",
    max_model_len: int = 32768,
    rank_k: int = 128,
    rank_v: int = 256,
    enable_xkv: bool = True,
    limit: Optional[int] = None,
) -> LongContextBenchmarkResult:
    """Run a long-context benchmark configuration."""

    config_name = f"rank_k={rank_k}_rank_v={rank_v}" if enable_xkv else "baseline"

    print(f"\n{'='*60}")
    print(f"Running: {config_name} on {task} (max_len={max_model_len})")
    print(f"{'='*60}")

    # Create xKV config
    xkv_config = None
    if enable_xkv:
        hf_config = AutoConfig.from_pretrained(model_path)
        num_layers = hf_config.num_hidden_layers

        xkv_config = generate_consecutive_xKV_config(
            layer_merge_impl="svd",
            start_layer=0,
            end_layer=num_layers - 1,
            num_layers=num_layers,
            group_size=2,
            rank_k=rank_k,
            rank_v=rank_v,
            paged_writeback=False,  # Disable for long context to avoid memory issues
        )

    # Create model with extended context
    model = NanoVLLMHarness(
        pretrained=model_path,
        batch_size=1,
        tensor_parallel_size=1,
        enable_xkv=enable_xkv,
        xkv_config=xkv_config,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,  # Must be >= max_model_len
        enforce_eager=True,
    )

    # Run evaluation with timing
    start_time = time.time()

    eval_kwargs = {
        "model": model,
        "tasks": [task],
        "batch_size": 1,
    }
    if limit is not None:
        eval_kwargs["limit"] = limit

    # For NIAH/RULER tasks, pass metadata with tokenizer and sequence lengths
    if 'niah' in task.lower() or 'ruler' in task.lower():
        eval_kwargs["task_manager"] = None  # Will use default
        eval_kwargs["metadata"] = {
            "pretrained": model_path,
            "max_seq_lengths": [max_model_len],
        }

    results = simple_evaluate(**eval_kwargs)
    total_time = time.time() - start_time

    # Extract metrics
    task_results = results["results"].get(task, {})
    
    # Get number of samples for throughput calculation
    n_samples = results.get("n_samples", {}).get(task, limit or 100)
    throughput = n_samples / total_time if total_time > 0 else 0

    print(f"Results: {task_results}")
    print(f"Throughput: {throughput:.4f} samples/s")

    return LongContextBenchmarkResult(
        config_name=config_name,
        task=task,
        max_model_len=max_model_len,
        rank_k=rank_k,
        rank_v=rank_v,
        enable_xkv=enable_xkv,
        metrics=task_results,
        throughput=throughput,
        total_time=total_time,
    )


def main():
    parser = argparse.ArgumentParser(description="Long-context benchmark for xKV")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="niah_single_1",
                        help="Task name (ruler, niah_single_1, niah_single_2, etc.)")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")

    # xKV parameters
    parser.add_argument("--enable_xkv", action="store_true")
    parser.add_argument("--rank_k", type=int, default=128)
    parser.add_argument("--rank_v", type=int, default=256)

    args = parser.parse_args()

    result = run_long_context_benchmark(
        model_path=args.model_path,
        task=args.task,
        max_model_len=args.max_model_len,
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        enable_xkv=args.enable_xkv,
        limit=args.limit,
    )

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()

