#!/usr/bin/env python3
"""
Benchmark script for xKV configurations.

Tests different rank_k, rank_v, and sparse settings to measure accuracy and throughput.

Usage:
    python scripts/benchmark_xkv.py --model_path /path/to/model --config "rank_k=128,rank_v=256"
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

from lm_eval import simple_evaluate
from nanovllm.eval import NanoVLLMHarness
from nanovllm.xkv import generate_consecutive_xKV_config
from transformers import AutoConfig


@dataclass
class BenchmarkResult:
    config_name: str
    rank_k: int
    rank_v: int
    enable_sparse: bool
    sparse_budget: int
    chunk_size: int
    num_outliers: int
    acc: float
    acc_norm: float
    throughput: float  # samples/sec
    total_time: float  # seconds


def run_single_benchmark(
    model_path: str,
    rank_k: int,
    rank_v: int,
    enable_sparse: bool = False,
    sparse_budget: int = 256,
    chunk_size: int = 8,
    num_outliers: int = 48,
    limit: int = 100,
    enable_xkv: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    config_name = f"rank_k={rank_k}_rank_v={rank_v}"
    if enable_sparse:
        config_name += f"_sparse(budget={sparse_budget})"
    if not enable_xkv:
        config_name = "baseline"

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
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
            paged_writeback=True,
            enable_sparse=enable_sparse,
            chunk_size=chunk_size,
            sparse_budget=sparse_budget,
            num_outliers=num_outliers,
        )

    # Create model
    model = NanoVLLMHarness(
        pretrained=model_path,
        batch_size=1,
        tensor_parallel_size=1,
        enable_xkv=enable_xkv,
        xkv_config=xkv_config,
    )

    # Run evaluation with timing
    start_time = time.time()
    results = simple_evaluate(
        model=model,
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=limit,
        batch_size=1,
    )
    total_time = time.time() - start_time

    # Extract metrics
    hellaswag = results["results"]["hellaswag"]
    acc = hellaswag["acc,none"]
    acc_norm = hellaswag["acc_norm,none"]
    throughput = limit / total_time

    print(f"Results: acc={acc:.4f}, acc_norm={acc_norm:.4f}, throughput={throughput:.2f} samples/s")

    return BenchmarkResult(
        config_name=config_name,
        rank_k=rank_k,
        rank_v=rank_v,
        enable_sparse=enable_sparse,
        sparse_budget=sparse_budget,
        chunk_size=chunk_size,
        num_outliers=num_outliers,
        acc=acc,
        acc_norm=acc_norm,
        throughput=throughput,
        total_time=total_time,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark a single xKV configuration")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")

    # xKV parameters
    parser.add_argument("--enable_xkv", action="store_true")
    parser.add_argument("--rank_k", type=int, default=128)
    parser.add_argument("--rank_v", type=int, default=256)
    parser.add_argument("--enable_sparse", action="store_true")
    parser.add_argument("--sparse_budget", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--num_outliers", type=int, default=48)

    args = parser.parse_args()

    result = run_single_benchmark(
        model_path=args.model_path,
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        enable_sparse=args.enable_sparse,
        sparse_budget=args.sparse_budget,
        chunk_size=args.chunk_size,
        num_outliers=args.num_outliers,
        limit=args.limit,
        enable_xkv=args.enable_xkv,
    )

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()

