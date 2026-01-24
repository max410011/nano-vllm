"""
SCBench evaluation wrapper for nano-vllm.

This module provides integration with MInference's SCBench benchmark
for evaluating multi-turn long-context performance.

SCBench (Shared Context Benchmark) tests KV cache reuse efficiency
by providing a shared context and asking multiple queries.

Reference:
    https://github.com/microsoft/MInference/tree/main/scbench
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from nanovllm import LLM, SamplingParams

# Add MInference scbench to path
MINFERENCE_PATH = Path(__file__).parent.parent.parent / "third_party" / "MInference"
SCBENCH_PATH = MINFERENCE_PATH / "scbench"

if SCBENCH_PATH.exists():
    sys.path.insert(0, str(SCBENCH_PATH))


def _get_scbench_modules():
    """Lazy import SCBench modules from MInference."""
    try:
        from eval_utils import (
            DATA_NAME_TO_MAX_NEW_TOKENS,
            DATA_NAME_TO_PATH,
            create_multiturn_prompt,
            create_scdq_prompt,
            dump_jsonl,
            get_ground_truth,
        )

        # Import compute_scores functions directly to avoid minference dependency
        # in args.py. We manually import what we need.
        import importlib.util
        compute_scores_path = SCBENCH_PATH / "compute_scores.py"

        # Load compute_scores module without running args.py import
        spec = importlib.util.spec_from_file_location("compute_scores_mod", compute_scores_path)
        compute_scores_mod = importlib.util.module_from_spec(spec)

        # Inject required modules before loading
        import sys
        # Mock args module to avoid minference dependency
        class MockArgs:
            @staticmethod
            def parse_args():
                return None
        sys.modules['args'] = MockArgs()

        # Also need repo_qa_utils
        repo_qa_path = SCBENCH_PATH / "repo_qa_utils.py"
        if repo_qa_path.exists():
            repo_spec = importlib.util.spec_from_file_location("repo_qa_utils", repo_qa_path)
            repo_mod = importlib.util.module_from_spec(repo_spec)
            sys.modules['repo_qa_utils'] = repo_mod
            try:
                repo_spec.loader.exec_module(repo_mod)
            except Exception:
                # If repo_qa_utils fails, provide a mock
                class MockRepoQA:
                    @staticmethod
                    def compute_score(*args, **kwargs):
                        return {}
                sys.modules['repo_qa_utils'] = MockRepoQA()

        spec.loader.exec_module(compute_scores_mod)
        compute_scores = compute_scores_mod.compute_scores

        return {
            "DATA_NAME_TO_MAX_NEW_TOKENS": DATA_NAME_TO_MAX_NEW_TOKENS,
            "DATA_NAME_TO_PATH": DATA_NAME_TO_PATH,
            "create_multiturn_prompt": create_multiturn_prompt,
            "create_scdq_prompt": create_scdq_prompt,
            "dump_jsonl": dump_jsonl,
            "get_ground_truth": get_ground_truth,
            "compute_scores": compute_scores,
        }
    except ImportError as e:
        raise ImportError(
            f"Failed to import SCBench modules from MInference. "
            f"Make sure the submodule is initialized: "
            f"git submodule update --init --recursive\n"
            f"Error: {e}"
        )


# Available SCBench tasks
SCBENCH_TASKS = [
    "scbench_kv",
    "scbench_kv_hard", 
    "scbench_passkey",
    "scbench_qa_eng",
    "scbench_qa_chn",
    "scbench_choice_eng",
    "scbench_mf",
    "scbench_repoqa",
    "scbench_summary",
    "scbench_vt",
    "scbench_many_shot",
    "scbench_summary_with_needles",
    "scbench_repoqa_and_kv",
    "scbench_prefix_suffix",
]


@dataclass
class SCBenchConfig:
    """Configuration for SCBench evaluation."""
    use_chat_template: bool = True
    scdq_mode: bool = True  # Same-Context-Different-Query mode
    disable_golden_context: bool = True  # Don't use gold answers in follow-up


class NanoVLLMSCBench:
    """
    SCBench evaluation wrapper for nano-vllm.
    
    This class wraps MInference's SCBench evaluation to work with nano-vllm.
    It supports both multi-turn mode and SCDQ (Same-Context-Different-Query) mode.
    
    Example:
        >>> from nanovllm.eval.scbench import NanoVLLMSCBench
        >>> evaluator = NanoVLLMSCBench(
        ...     pretrained="Qwen/Qwen2.5-7B-Instruct",
        ...     max_model_len=32768,
        ... )
        >>> results = evaluator.evaluate("scbench_kv", limit=10)
    """
    
    def __init__(
        self,
        pretrained: str,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        enable_xkv: bool = False,
        xkv_config: Optional[Any] = None,
        config: Optional[SCBenchConfig] = None,
        **model_kwargs,
    ):
        """
        Initialize SCBench evaluator.
        
        Args:
            pretrained: Path or name of the pretrained model
            max_model_len: Maximum context length
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enable_xkv: Enable xKV compression
            xkv_config: xKV configuration (if enable_xkv=True)
            config: SCBench configuration
            **model_kwargs: Additional arguments for LLM
        """
        self.config = config or SCBenchConfig()
        self.max_model_len = max_model_len
        
        # Build model kwargs
        model_kwargs.update({
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
        })
        
        if enable_xkv and xkv_config is not None:
            model_kwargs["xkv_config"] = xkv_config
        
        # Lazy import to avoid heavy module-level imports
        from nanovllm import LLM
        
        self.llm = LLM(pretrained, **model_kwargs)
        self.tokenizer = self.llm.tokenizer
        self.enable_xkv = enable_xkv
        self.model_name = pretrained.split("/")[-1]
        
        # Load SCBench modules
        self._scbench = _get_scbench_modules()

    def _truncate_input(self, input_ids: list, max_length: int) -> list:
        """Truncate input from the middle if too long."""
        if max_length < 0 or len(input_ids) <= max_length:
            return input_ids
        split = max_length // 2
        return input_ids[:split] + input_ids[-split:]

    def test_scdq(self, example: dict, max_length: int = 100) -> dict:
        """
        Test in SCDQ (Same-Context-Different-Query) mode.

        In this mode, the context is encoded once and reused for all queries.
        This tests the KV cache reuse efficiency.
        """
        from nanovllm import SamplingParams

        results = []
        init_prompt_ids = None

        for idx, prompt in enumerate(example["prompts"]):
            if idx == 0:
                # First prompt is context - just store the token ids
                init_prompt_ids = prompt
            else:
                # Subsequent prompts are queries
                if isinstance(max_length, dict):
                    max_length_per_turn = max_length[example["task"][idx - 1]]
                else:
                    max_length_per_turn = max_length

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_length_per_turn,
                )

                # Encode query and concatenate with context
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = init_prompt_ids + current_ids

                # Generate using nano-vllm
                result = self.llm.generate(
                    prompt_token_ids=[input_ids],
                    sampling_params=sampling_params,
                )
                results.append(result[0].outputs[0].text)

        output = {"answers": results, "gt": example["ground_truth"]}
        if isinstance(max_length, dict):
            output["task"] = example["task"]
        return output

    def test(self, example: dict, max_length: int = 100,
             disable_golden_context: bool = False) -> dict:
        """
        Test in multi-turn mode.

        In this mode, each turn builds on the previous conversation.
        """
        from nanovllm import SamplingParams

        results = []
        input_ids = None

        for idx, prompt in enumerate(example["prompts"]):
            if isinstance(max_length, dict):
                max_length_per_turn = max_length[example["task"][idx]]
            else:
                max_length_per_turn = max_length

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_length_per_turn,
            )

            if idx == 0:
                input_ids = prompt
            else:
                current_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

                if disable_golden_context and results:
                    # Add previous generation to context
                    prev_output_ids = self.tokenizer.encode(
                        results[-1], add_special_tokens=False
                    )
                    input_ids = input_ids + prev_output_ids + [self.tokenizer.eos_token_id]

                input_ids = input_ids + current_ids

            # Generate
            result = self.llm.generate(
                prompt_token_ids=[input_ids],
                sampling_params=sampling_params,
            )
            results.append(result[0].outputs[0].text)

        output = {"answers": results, "gt": example["ground_truth"]}
        if isinstance(max_length, dict):
            output["task"] = example["task"]
        return output

    def evaluate(
        self,
        task_name: str,
        output_dir: str = "./results/scbench",
        limit: Optional[int] = None,
        max_turns: int = -1,
        rewrite: bool = False,
    ) -> dict:
        """
        Run SCBench evaluation on a task.

        Args:
            task_name: Name of the SCBench task (e.g., 'scbench_kv')
            output_dir: Directory to save results
            limit: Number of examples to evaluate (None for all)
            max_turns: Maximum number of turns per example (-1 for all)
            rewrite: Whether to overwrite existing results

        Returns:
            Dictionary with evaluation results including scores
        """
        import torch
        from datasets import load_dataset

        scbench = self._scbench

        # Validate task name
        if task_name not in SCBENCH_TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {SCBENCH_TASKS}")

        # Get max new tokens for this task
        max_new_tokens = scbench["DATA_NAME_TO_MAX_NEW_TOKENS"].get(task_name, 100)

        # Load data from HuggingFace
        print(f"Loading SCBench task: {task_name}")
        examples = load_dataset("microsoft/SCBench", task_name, split="test")
        examples = list(examples)

        if limit is not None:
            examples = examples[:limit]

        # Limit turns if specified
        max_turn_size = len(examples[0]["multi_turns"])
        if max_turns > 0 and max_turns < max_turn_size:
            examples = [
                {**eg, "multi_turns": eg["multi_turns"][:max_turns]}
                for eg in examples
            ]
            max_turn_size = max_turns

        # Setup output directory
        result_dir = Path(output_dir)
        result_dir.mkdir(exist_ok=True, parents=True)

        scdq_suffix = "_scdq" if self.config.scdq_mode else "_multi_turn"
        output_path = result_dir / f"prediction_{task_name}{scdq_suffix}.jsonl"

        print(f"==== Evaluation {task_name} ====")
        print(f"# examples: {len(examples)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Num of turns: {max_turn_size}")
        print(f"SCDQ mode: {self.config.scdq_mode}")

        # Check for existing results
        preds = []
        done = set()
        if output_path.exists() and not rewrite:
            print(f"Loading existing results from {output_path}")
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    tmp = json.loads(line)
                    done.add(int(tmp["id"]))
                    preds.append(tmp)

        # Run evaluation
        start_time = time.time()

        for i, eg in tqdm(enumerate(examples), total=len(examples), desc=task_name):
            if i in done:
                continue

            # Calculate max input length
            if isinstance(max_new_tokens, dict):
                max_input_length = self.max_model_len - (
                    sum(list(max_new_tokens.values())) * max_turn_size // 2
                )
            else:
                max_input_length = self.max_model_len - max_new_tokens * max_turn_size

            if self.config.scdq_mode:
                max_input_length -= 1000

            # Create prompts using MInference's function
            if self.config.scdq_mode:
                encoded_eg = scbench["create_scdq_prompt"](
                    eg,
                    data_name=task_name,
                    tok=self.tokenizer,
                    use_chat_template=self.config.use_chat_template,
                    use_vllm=True,
                )
            else:
                encoded_eg = scbench["create_multiturn_prompt"](
                    eg,
                    data_name=task_name,
                    tok=self.tokenizer,
                    use_chat_template=self.config.use_chat_template,
                    use_vllm=True,
                    disable_golden_context=self.config.disable_golden_context,
                )

            # Truncate context if needed
            context = self._truncate_input(
                self.tokenizer.encode(encoded_eg["prompts"][0])
                if isinstance(encoded_eg["prompts"][0], str)
                else encoded_eg["prompts"][0],
                max_input_length
            )
            encoded_eg["prompts"][0] = context

            # Run inference
            if self.config.scdq_mode:
                pred = self.test_scdq(encoded_eg, max_length=max_new_tokens)
            else:
                pred = self.test(
                    encoded_eg,
                    max_length=max_new_tokens,
                    disable_golden_context=self.config.disable_golden_context,
                )

            # Get ground truth using MInference's function
            gts = scbench["get_ground_truth"](eg, task_name)

            # Save predictions
            for turn_idx, (ans, gt, turn) in enumerate(
                zip(pred["answers"], gts, eg["multi_turns"])
            ):
                case = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in pred:
                    case["task"] = pred["task"][turn_idx]
                preds.append(case)

            # Save incrementally
            scbench["dump_jsonl"](preds, output_path)
            done.add(i)

            # Clear CUDA cache
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        # Compute scores using MInference's function
        score = scbench["compute_scores"](
            output_path,
            task_name,
            self.model_name,
            max_seq_length=self.max_model_len,
            scdq_mode=self.config.scdq_mode,
        )

        return {
            "task": task_name,
            "score": score,
            "num_examples": len(examples),
            "elapsed_time": elapsed_time,
            "throughput": len(examples) / elapsed_time if elapsed_time > 0 else 0,
            "output_path": str(output_path),
        }


def run_scbench(
    pretrained: str,
    task: str,
    output_dir: str = "./results/scbench",
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    limit: Optional[int] = None,
    scdq_mode: bool = True,
    enable_xkv: bool = False,
    xkv_config: Optional[Any] = None,
    **kwargs,
) -> dict:
    """
    Convenience function to run SCBench evaluation.

    Args:
        pretrained: Model path or name
        task: SCBench task name
        output_dir: Output directory for results
        max_model_len: Maximum context length
        tensor_parallel_size: Number of GPUs
        limit: Number of examples (None for all)
        scdq_mode: Use SCDQ mode
        enable_xkv: Enable xKV compression
        xkv_config: xKV configuration
        **kwargs: Additional model arguments

    Returns:
        Evaluation results dictionary
    """
    config = SCBenchConfig(scdq_mode=scdq_mode)

    evaluator = NanoVLLMSCBench(
        pretrained=pretrained,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enable_xkv=enable_xkv,
        xkv_config=xkv_config,
        config=config,
        **kwargs,
    )

    return evaluator.evaluate(
        task_name=task,
        output_dir=output_dir,
        limit=limit,
    )

