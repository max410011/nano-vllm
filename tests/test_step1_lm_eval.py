"""
Step 1: lm_eval Integration Tests

Tests for lm-evaluation-harness integration with nano-vllm.
"""

import pytest
import torch
import os


# Auto-detect model path
def get_model_path():
    """Auto-detect model path from HF cache."""
    hf_paths = [
        os.environ.get("HF_HOME", ""),
        "/share2/huggingface/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    
    for base_path in hf_paths:
        if not base_path or not os.path.exists(base_path):
            continue
        qwen_path = os.path.join(base_path, "models--Qwen--Qwen3-0.6B")
        if os.path.exists(qwen_path):
            snapshots = os.path.join(qwen_path, "snapshots")
            if os.path.exists(snapshots):
                versions = os.listdir(snapshots)
                if versions:
                    return os.path.join(snapshots, versions[0])
    return None


MODEL_PATH = get_model_path()


class TestLMHarnessImports:
    """Test that all lm_eval related imports work."""
    
    def test_nanovllm_harness_import(self):
        """Test NanoVLLMHarness can be imported."""
        from nanovllm.eval import NanoVLLMHarness
        assert NanoVLLMHarness is not None
    
    def test_lm_eval_import(self):
        """Test lm_eval can be imported."""
        import lm_eval
        assert lm_eval is not None
    
    def test_simple_evaluate_import(self):
        """Test simple_evaluate can be imported."""
        from lm_eval import simple_evaluate
        assert simple_evaluate is not None
    
    def test_template_lm_import(self):
        """Test TemplateLM can be imported."""
        from lm_eval.api.model import TemplateLM
        assert TemplateLM is not None


class TestNanoVLLMHarnessInterface:
    """Test NanoVLLMHarness interface without loading model."""
    
    def test_harness_is_template_lm_subclass(self):
        """Test NanoVLLMHarness inherits from TemplateLM."""
        from nanovllm.eval import NanoVLLMHarness
        from lm_eval.api.model import TemplateLM
        assert issubclass(NanoVLLMHarness, TemplateLM)
    
    def test_harness_has_required_methods(self):
        """Test NanoVLLMHarness has all required methods."""
        from nanovllm.eval import NanoVLLMHarness
        required_methods = [
            '_loglikelihood_tokens',
            'loglikelihood_rolling', 
            'generate_until',
            'tok_encode',
            'tok_decode',
        ]
        for method in required_methods:
            assert hasattr(NanoVLLMHarness, method), f"Missing method: {method}"
    
    def test_harness_has_required_properties(self):
        """Test NanoVLLMHarness has all required properties."""
        from nanovllm.eval import NanoVLLMHarness
        required_properties = [
            'eot_token_id',
            'max_length',
            'max_gen_toks',
            'batch_size',
            'device',
        ]
        for prop in required_properties:
            assert hasattr(NanoVLLMHarness, prop), f"Missing property: {prop}"


# Note: Tests that require loading the model should be run separately
# to avoid process group initialization conflicts.
# Use: python -m pytest tests/test_step1_lm_eval.py::TestComputePromptLogprobs -v
# or run the integration test script directly.

