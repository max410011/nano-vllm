#!/usr/bin/env python3
"""
Step 0: Environment Setup Tests
Validates that all required dependencies are properly installed and functional.
"""

import pytest
import sys


class TestPythonEnvironment:
    """Test Python version and basic environment."""

    def test_python_version(self):
        """Python version should be 3.11+."""
        assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


class TestPyTorchCUDA:
    """Test PyTorch and CUDA availability."""

    def test_torch_import(self):
        """PyTorch should be importable."""
        import torch
        assert torch is not None

    def test_torch_version(self):
        """PyTorch version should be 2.x."""
        import torch
        major_version = int(torch.__version__.split('.')[0])
        assert major_version >= 2, f"PyTorch 2.x required, got {torch.__version__}"

    def test_cuda_available(self):
        """CUDA should be available."""
        import torch
        assert torch.cuda.is_available(), "CUDA is not available"

    def test_cuda_device_count(self):
        """At least one CUDA device should be available."""
        import torch
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_cuda_tensor_operations(self):
        """Basic CUDA tensor operations should work."""
        import torch
        x = torch.randn(10, 10, device='cuda')
        y = torch.randn(10, 10, device='cuda')
        z = torch.matmul(x, y)
        assert z.shape == (10, 10)
        assert z.device.type == 'cuda'


class TestTriton:
    """Test Triton availability."""

    def test_triton_import(self):
        """Triton should be importable."""
        import triton
        assert triton is not None


class TestFlashAttention:
    """Test Flash Attention availability."""

    def test_flash_attn_import(self):
        """Flash Attention should be importable."""
        from flash_attn import flash_attn_func
        assert flash_attn_func is not None

    def test_flash_attn_varlen_import(self):
        """Flash Attention varlen functions should be importable."""
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        assert flash_attn_varlen_func is not None
        assert flash_attn_with_kvcache is not None


class TestTransformers:
    """Test Transformers availability."""

    def test_transformers_import(self):
        """Transformers should be importable."""
        import transformers
        assert transformers is not None

    def test_auto_model_import(self):
        """AutoModelForCausalLM should be importable."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert AutoModelForCausalLM is not None
        assert AutoTokenizer is not None


class TestLMEval:
    """Test LM Evaluation Harness availability."""

    def test_lm_eval_import(self):
        """lm_eval should be importable."""
        import lm_eval
        assert lm_eval is not None

    def test_lm_eval_models_import(self):
        """lm_eval.models should be importable."""
        from lm_eval.models import huggingface
        assert huggingface is not None


class TestNanoVLLM:
    """Test Nano-vLLM availability."""

    def test_nanovllm_import(self):
        """nanovllm should be importable."""
        import nanovllm
        assert nanovllm is not None

    def test_llm_import(self):
        """LLM class should be importable."""
        from nanovllm import LLM
        assert LLM is not None

    def test_sampling_params_import(self):
        """SamplingParams should be importable."""
        from nanovllm import SamplingParams
        assert SamplingParams is not None

    def test_config_import(self):
        """Config module should be importable."""
        from nanovllm.config import Config
        assert Config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

