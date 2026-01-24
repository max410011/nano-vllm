"""
Tests for Step 5: SCBench integration using MInference submodule.

These tests verify that the SCBench wrapper correctly integrates with
MInference's SCBench benchmark for multi-turn long-context evaluation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSCBenchModuleImports:
    """Test that SCBench modules can be imported."""
    
    def test_scbench_module_exists(self):
        """Test that scbench module exists."""
        from nanovllm.eval import scbench
        assert scbench is not None
    
    def test_scbench_classes_importable(self):
        """Test that main classes are importable."""
        from nanovllm.eval.scbench import (
            NanoVLLMSCBench,
            SCBenchConfig,
            SCBENCH_TASKS,
            run_scbench,
        )
        assert NanoVLLMSCBench is not None
        assert SCBenchConfig is not None
        assert SCBENCH_TASKS is not None
        assert run_scbench is not None
    
    def test_scbench_exports_from_init(self):
        """Test that scbench classes are exported from __init__."""
        from nanovllm.eval import (
            NanoVLLMSCBench,
            SCBenchConfig,
            SCBENCH_TASKS,
            run_scbench,
        )
        assert NanoVLLMSCBench is not None
        assert SCBenchConfig is not None


class TestMInferenceSubmodule:
    """Test MInference submodule integration."""
    
    def test_minference_path_exists(self):
        """Test that MInference submodule path exists."""
        from nanovllm.eval.scbench import MINFERENCE_PATH, SCBENCH_PATH
        assert MINFERENCE_PATH.exists(), f"MInference not found at {MINFERENCE_PATH}"
        assert SCBENCH_PATH.exists(), f"SCBench not found at {SCBENCH_PATH}"
    
    def test_minference_eval_utils_exists(self):
        """Test that eval_utils.py exists in MInference."""
        from nanovllm.eval.scbench import SCBENCH_PATH
        eval_utils_path = SCBENCH_PATH / "eval_utils.py"
        assert eval_utils_path.exists(), f"eval_utils.py not found at {eval_utils_path}"
    
    def test_minference_compute_scores_exists(self):
        """Test that compute_scores.py exists in MInference."""
        from nanovllm.eval.scbench import SCBENCH_PATH
        compute_scores_path = SCBENCH_PATH / "compute_scores.py"
        assert compute_scores_path.exists()
    
    def test_scbench_modules_loadable(self):
        """Test that SCBench modules can be loaded."""
        from nanovllm.eval.scbench import _get_scbench_modules
        modules = _get_scbench_modules()
        
        assert "DATA_NAME_TO_MAX_NEW_TOKENS" in modules
        assert "DATA_NAME_TO_PATH" in modules
        assert "create_scdq_prompt" in modules
        assert "create_multiturn_prompt" in modules
        assert "get_ground_truth" in modules
        assert "compute_scores" in modules
        assert "dump_jsonl" in modules


class TestSCBenchConfig:
    """Test SCBenchConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from nanovllm.eval.scbench import SCBenchConfig
        config = SCBenchConfig()
        
        assert config.use_chat_template is True
        assert config.scdq_mode is True
        assert config.disable_golden_context is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        from nanovllm.eval.scbench import SCBenchConfig
        config = SCBenchConfig(
            use_chat_template=False,
            scdq_mode=False,
            disable_golden_context=False,
        )
        
        assert config.use_chat_template is False
        assert config.scdq_mode is False
        assert config.disable_golden_context is False


class TestSCBenchTasks:
    """Test SCBench task definitions."""
    
    def test_tasks_list_not_empty(self):
        """Test that task list is not empty."""
        from nanovllm.eval.scbench import SCBENCH_TASKS
        assert len(SCBENCH_TASKS) > 0
    
    def test_expected_tasks_present(self):
        """Test that expected tasks are in the list."""
        from nanovllm.eval.scbench import SCBENCH_TASKS
        
        expected_tasks = [
            "scbench_kv",
            "scbench_passkey",
            "scbench_qa_eng",
            "scbench_summary",
        ]
        
        for task in expected_tasks:
            assert task in SCBENCH_TASKS, f"Expected task {task} not found"
    
    def test_tasks_match_minference(self):
        """Test that tasks match MInference's DATA_NAME_TO_PATH."""
        from nanovllm.eval.scbench import SCBENCH_TASKS, _get_scbench_modules
        
        modules = _get_scbench_modules()
        minference_tasks = set(modules["DATA_NAME_TO_PATH"].keys())
        
        # All our tasks should be in MInference
        for task in SCBENCH_TASKS:
            assert task in minference_tasks, f"Task {task} not in MInference"


class TestDataLoading:
    """Test data loading from HuggingFace."""
    
    @pytest.mark.slow
    def test_load_scbench_data(self):
        """Test loading SCBench data from HuggingFace."""
        from datasets import load_dataset
        
        ds = load_dataset("microsoft/SCBench", "scbench_kv", split="test")
        assert len(ds) > 0
        
        # Check data structure
        sample = ds[0]
        assert "context" in sample or "input" in sample
        assert "multi_turns" in sample
        assert len(sample["multi_turns"]) > 0
    
    @pytest.mark.slow  
    def test_data_has_required_fields(self):
        """Test that data has required fields for evaluation."""
        from datasets import load_dataset
        
        ds = load_dataset("microsoft/SCBench", "scbench_kv", split="test")
        sample = ds[0]
        
        # Multi-turns should have input and answer
        turn = sample["multi_turns"][0]
        assert "input" in turn
        assert "answer" in turn

