#!/usr/bin/env python3
"""
Evaluation script for nano-vllm using lm-evaluation-harness.

Usage:
    python scripts/eval_lm_harness.py --model_path ./models/Qwen3-0.6B --tasks hellaswag --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

from lm_eval import simple_evaluate

from nanovllm.eval import NanoVLLMHarness


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nano-vllm with lm-evaluation-harness"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the model"
    )
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="hellaswag", 
        help="Comma-separated list of tasks"
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0, 
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of examples per task"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="Path to save results JSON"
    )
    args = parser.parse_args()
    
    # Create nano-vllm model wrapper
    print(f"Loading model from: {args.model_path}")
    model = NanoVLLMHarness(
        pretrained=args.model_path, 
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Run evaluation
    print(f"Running evaluation on tasks: {args.tasks}")
    results = simple_evaluate(
        model=model,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if not metric.endswith("_stderr"):
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    # Save results if output path specified
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()

