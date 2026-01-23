#!/usr/bin/env python3
"""
Environment Verification Script for Nano-vLLM

This script verifies that all required dependencies are properly installed
and that the environment is correctly configured for running nano-vllm.

Usage:
    python scripts/verify_env.py [--model-path PATH] [--quick]
"""

import sys
import argparse
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class VerificationResult:
    """Result of a verification check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


def print_result(result: VerificationResult) -> None:
    """Print verification result with color coding."""
    status = "✓" if result.passed else "✗"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {result.name}: {result.message}")
    if result.details and not result.passed:
        print(f"    Details: {result.details}")


def verify_python_version() -> VerificationResult:
    """Verify Python version is compatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and 10 <= version.minor <= 12:
        return VerificationResult("Python Version", True, f"Python {version_str}")
    else:
        return VerificationResult(
            "Python Version", False, 
            f"Python {version_str} (requires 3.10-3.12)",
            "Install a compatible Python version"
        )


def verify_cuda() -> VerificationResult:
    """Verify CUDA availability and version."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return VerificationResult(
                "CUDA", True,
                f"CUDA {cuda_version}, {device_count} GPU(s): {device_name}"
            )
        else:
            return VerificationResult("CUDA", False, "CUDA not available")
    except ImportError:
        return VerificationResult("CUDA", False, "PyTorch not installed")


def verify_torch() -> VerificationResult:
    """Verify PyTorch installation."""
    try:
        import torch
        return VerificationResult("PyTorch", True, f"torch {torch.__version__}")
    except ImportError as e:
        return VerificationResult("PyTorch", False, "Not installed", str(e))


def verify_triton() -> VerificationResult:
    """Verify Triton installation."""
    try:
        import triton
        return VerificationResult("Triton", True, f"triton {triton.__version__}")
    except ImportError as e:
        return VerificationResult("Triton", False, "Not installed", str(e))


def verify_flash_attn() -> VerificationResult:
    """Verify flash-attn installation."""
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "unknown")
        return VerificationResult("Flash-Attn", True, f"flash-attn {version}")
    except ImportError as e:
        return VerificationResult("Flash-Attn", False, "Not installed", str(e))


def verify_transformers() -> VerificationResult:
    """Verify transformers installation."""
    try:
        import transformers
        return VerificationResult(
            "Transformers", True, f"transformers {transformers.__version__}"
        )
    except ImportError as e:
        return VerificationResult("Transformers", False, "Not installed", str(e))


def verify_lm_eval() -> VerificationResult:
    """Verify lm-evaluation-harness installation."""
    try:
        import lm_eval
        version = getattr(lm_eval, "__version__", "unknown")
        return VerificationResult("LM-Eval", True, f"lm-eval {version}")
    except ImportError as e:
        return VerificationResult("LM-Eval", False, "Not installed", str(e))


def verify_nanovllm_import() -> VerificationResult:
    """Verify nano-vllm can be imported."""
    try:
        from nanovllm import LLM, SamplingParams
        return VerificationResult("Nano-vLLM Import", True, "Successfully imported")
    except ImportError as e:
        return VerificationResult("Nano-vLLM Import", False, "Import failed", str(e))


def verify_nanovllm_inference(model_path: str) -> VerificationResult:
    """Verify nano-vllm can perform basic inference."""
    try:
        from nanovllm import LLM, SamplingParams
        
        start_time = time.time()
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        load_time = time.time() - start_time
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=16)
        start_time = time.time()
        outputs = llm.generate(["Hello, world!"], sampling_params, use_tqdm=False)
        inference_time = time.time() - start_time
        
        output_text = outputs[0]["text"][:50]  # Truncate for display
        return VerificationResult(
            "Nano-vLLM Inference", True,
            f"OK (load: {load_time:.1f}s, infer: {inference_time:.2f}s)",
            f"Output: {output_text}..."
        )
    except Exception as e:
        return VerificationResult("Nano-vLLM Inference", False, "Failed", str(e))


def get_default_model_path() -> str:
    """Get the default model path from HuggingFace cache."""
    import os
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

    # Check for Qwen3-0.6B in the cache
    model_dirs = [
        os.path.join(hf_home, "models--Qwen--Qwen3-0.6B"),
        "/share2/huggingface/hub/models--Qwen--Qwen3-0.6B",
        os.path.expanduser("~/huggingface/Qwen3-0.6B"),
    ]

    for model_dir in model_dirs:
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])

    return None


def main():
    parser = argparse.ArgumentParser(description="Verify nano-vllm environment")
    parser.add_argument("--model-path", type=str, help="Path to model for inference test")
    parser.add_argument("--quick", action="store_true", help="Skip inference test")
    args = parser.parse_args()

    # Auto-detect model path if not provided
    if not args.model_path:
        args.model_path = get_default_model_path()
        if args.model_path:
            print(f"Auto-detected model path: {args.model_path}")
    
    print("=" * 60)
    print("Nano-vLLM Environment Verification")
    print("=" * 60)
    
    results = [
        verify_python_version(),
        verify_torch(),
        verify_cuda(),
        verify_triton(),
        verify_flash_attn(),
        verify_transformers(),
        verify_lm_eval(),
        verify_nanovllm_import(),
    ]
    
    for result in results:
        print_result(result)
    
    # Inference test (optional)
    if args.model_path and not args.quick:
        print("\n" + "-" * 60)
        print("Running inference test...")
        inference_result = verify_nanovllm_inference(args.model_path)
        print_result(inference_result)
        results.append(inference_result)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    if passed == total:
        print(f"\033[92mAll {total} checks passed!\033[0m")
        return 0
    else:
        print(f"\033[91m{passed}/{total} checks passed\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())

