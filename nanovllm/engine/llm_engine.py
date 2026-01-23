import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    @torch.inference_mode()
    def compute_prompt_logprobs(self, token_ids: list[int]) -> torch.Tensor:
        """
        Compute log probabilities for all tokens in the prompt.

        This method is used by lm-evaluation-harness to compute perplexity
        and likelihood-based metrics.

        Args:
            token_ids: List of token IDs for the prompt

        Returns:
            Tensor of shape [seq_len, vocab_size] containing log probabilities
        """
        from nanovllm.utils.context import set_context, reset_context

        # Create input tensors
        seq_len = len(token_ids)
        input_ids = torch.tensor(token_ids, dtype=torch.int64, device="cuda")
        positions = torch.arange(seq_len, dtype=torch.int64, device="cuda")

        # Set up context for prefill
        # Use slot_mapping with -1 values to skip KV cache storage
        # (store_kvcache_kernel skips slots with -1)
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        slot_mapping = torch.full((seq_len,), -1, dtype=torch.int32, device="cuda")

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None,
        )

        try:
            # Forward pass to get hidden states
            model = self.model_runner.model
            hidden_states = model(input_ids, positions)
            # Get logits for ALL tokens (bypass the last-token-only logic in lm_head)
            # lm_head.forward normally extracts only last token for generation,
            # but we need all tokens for perplexity computation
            logits = F.linear(hidden_states, model.lm_head.weight)
            # Handle tensor parallelism if needed
            if model.lm_head.tp_size > 1:
                import torch.distributed as dist
                all_logits = [torch.empty_like(logits) for _ in range(model.lm_head.tp_size)] \
                    if model.lm_head.tp_rank == 0 else None
                dist.gather(logits, all_logits, 0)
                logits = torch.cat(all_logits, -1) if model.lm_head.tp_rank == 0 else None
            # Convert to log probabilities
            log_probs = F.log_softmax(logits.float(), dim=-1)
            return log_probs
        finally:
            reset_context()
