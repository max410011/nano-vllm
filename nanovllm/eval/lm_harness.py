"""
LM-Evaluation-Harness compatible wrapper for nano-vllm.

This module implements the TemplateLM interface from lm-evaluation-harness,
enabling standardized evaluation of nano-vllm models on various benchmarks.

Design Pattern:
    - Adapter Pattern: Wraps nano-vllm's LLM class to conform to TemplateLM interface
    - Facade Pattern: Provides simplified interface for evaluation operations

Key Methods:
    - _loglikelihood_tokens: Compute log-likelihood for continuation tokens
    - loglikelihood_rolling: Compute rolling log-likelihood (perplexity)
    - generate_until: Generate text until stop sequences
"""

from typing import List, Tuple, Optional
import torch

from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance

from nanovllm import LLM, SamplingParams


class NanoVLLMHarness(TemplateLM):
    """
    LM-Evaluation-Harness compatible wrapper for nano-vllm.

    This class adapts nano-vllm's LLM interface to work with
    lm-evaluation-harness for standardized model evaluation.

    Attributes:
        model: The underlying nano-vllm LLM instance
        tokenizer: The tokenizer from the model
        _max_gen_toks: Maximum tokens to generate
        _batch_size: Batch size for evaluation
    """

    def __init__(
        self,
        pretrained: str,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
        max_model_len: Optional[int] = None,
        enable_xkv: bool = False,
        xkv_config: Optional["xKVConfig"] = None,
        **kwargs,
    ):
        """
        Initialize NanoVLLMHarness wrapper.

        Args:
            pretrained: Path to the pretrained model
            max_gen_toks: Maximum tokens to generate in generation tasks
            batch_size: Batch size for evaluation (currently single-threaded)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enforce_eager: If True, disable CUDA graphs for flexibility
            max_model_len: Maximum model context length
            enable_xkv: If True, enable xKV compression
            xkv_config: Configuration for xKV compression
            **kwargs: Additional arguments (ignored)
        """
        super().__init__()
        self.model_path = pretrained
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size

        # Initialize nano-vllm with appropriate settings
        model_kwargs = {
            "enforce_eager": enforce_eager,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if max_model_len is not None:
            model_kwargs["max_model_len"] = max_model_len

        # xKV configuration
        if enable_xkv:
            model_kwargs["enable_xkv"] = True
            model_kwargs["xkv_config"] = xkv_config

        self.model = LLM(pretrained, **model_kwargs)
        self.tokenizer = self.model.tokenizer

    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Maximum context length supported by the model."""
        return getattr(
            self.model.model_runner.config.hf_config,
            'max_position_embeddings',
            2048
        )

    @property
    def max_gen_toks(self) -> int:
        """Maximum tokens to generate."""
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        """Batch size for evaluation."""
        return self._batch_size

    @property
    def device(self) -> str:
        """Device used for computation."""
        return "cuda"

    def tok_encode(
        self,
        string: str,
        add_special_tokens: Optional[bool] = None,
        **kwargs
    ) -> List[int]:
        """Encode string to token IDs."""
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for continuation tokens.

        For each request, computes the sum of log probabilities for the
        continuation tokens given the context, and whether the greedy
        decoding would produce the exact continuation.

        Args:
            requests: List of (cache_key, context_enc, continuation_enc) tuples
            disable_tqdm: If True, disable progress bar

        Returns:
            List of (log_likelihood_sum, is_greedy) tuples
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(requests, disable=disable_tqdm, desc="Running loglikelihood")

        for cache_key, context_enc, continuation_enc in iterator:
            full_ids = context_enc + continuation_enc

            # Truncate from left if exceeds max length
            if len(full_ids) > self.max_length:
                full_ids = full_ids[-self.max_length:]
                overflow = len(context_enc) + len(continuation_enc) - self.max_length
                ctxlen = max(0, len(context_enc) - overflow)
            else:
                ctxlen = len(context_enc)

            # Get log probabilities: shape [seq_len, vocab_size]
            log_probs = self.model.compute_prompt_logprobs(full_ids)

            # Gather logprobs for continuation tokens
            # Position i predicts token i+1
            cont_tokens = torch.tensor(full_ids[ctxlen:], device=log_probs.device)
            cont_logprobs = log_probs[ctxlen-1:len(full_ids)-1].gather(
                1, cont_tokens.unsqueeze(1)
            ).squeeze(1)

            # Check if greedy decoding matches
            greedy_tokens = log_probs[ctxlen-1:len(full_ids)-1].argmax(dim=-1)
            is_greedy = (greedy_tokens == cont_tokens).all().item()

            results.append((cont_logprobs.sum().item(), is_greedy))

        return results

    def loglikelihood_rolling(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False
    ) -> List[float]:
        """
        Compute rolling log-likelihood (perplexity-style).

        Args:
            requests: List of Instance objects with string arguments
            disable_tqdm: If True, disable progress bar

        Returns:
            List of total log-likelihood values
        """
        from tqdm import tqdm

        results = []
        for request in tqdm(requests, disable=disable_tqdm, desc="Running loglikelihood_rolling"):
            (string,) = request.args
            input_ids = self.tok_encode(string)

            if not input_ids:
                results.append(0.0)
                continue

            total_logprob = 0.0
            for start in range(0, len(input_ids), self.max_length):
                chunk = input_ids[start:start + self.max_length]
                log_probs = self.model.compute_prompt_logprobs(chunk)

                # Sum logprobs for tokens 1..len (predicted by positions 0..len-1)
                tokens = torch.tensor(chunk[1:], device=log_probs.device)
                total_logprob += log_probs[:-1].gather(1, tokens.unsqueeze(1)).sum().item()

            results.append(total_logprob)

        return results

    def generate_until(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False
    ) -> List[str]:
        """
        Generate text until stop sequences.

        Args:
            requests: List of Instance objects with (context, gen_kwargs) arguments
            disable_tqdm: If True, disable progress bar

        Returns:
            List of generated text strings
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(requests, disable=disable_tqdm, desc="Generating")

        for request in iterator:
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 1.0)

            # nano-vllm requires temperature > 0
            if temperature < 1e-10:
                temperature = 1e-5

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_gen_toks,
            )

            outputs = self.model.generate([context], sampling_params, use_tqdm=False)
            generated_text = outputs[0]["text"]

            # Handle stop sequences
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]

            results.append(generated_text)

        return results

