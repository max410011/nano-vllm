"""
Sparse KV Selection for xKV (ShadowKV-style).

This module implements sparse attention selection based on ShadowKV:
- Chunk-based landmarks: Mean of each chunk for approximating attention scores
- Outlier detection: Identify chunks with low cosine similarity
- TopK selection: Select important chunks based on landmark attention scores
- Temporal locality cache: Reuse selected chunks across decoding steps

Reference: ShadowKV (arXiv:2410.21465v3)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class SparseSelector:
    """
    Sparse KV selector using ShadowKV's landmark-based selection.

    Supports two modes:
    - xk_sr: K uses SVD, V is offloaded to CPU (original ShadowKV style)
    - xkv_sr: Both K and V use SVD compression + sparse selection

    During prefill:
    1. Compute landmarks (mean of each chunk)
    2. Detect outlier chunks (low cosine similarity within chunk)
    3. Store outliers as static cache on GPU
    4. For xk_sr mode: offload full V to CPU

    During decode:
    1. Use landmarks to approximate attention scores
    2. Select top-k chunks based on scores
    3. Return selected chunk indices
    4. For xk_sr mode: load selected V chunks from CPU
    """

    def __init__(
        self,
        chunk_size: int = 8,
        sparse_budget: int = 256,
        num_outliers: int = 48,
        sparse_mode: str = "xkv_sr",  # "xk_sr" or "xkv_sr"
    ):
        """
        Initialize sparse selector.

        Args:
            chunk_size: Number of tokens per chunk (default: 8)
            sparse_budget: Number of chunks to select (default: 256)
            num_outliers: Number of outlier chunks to keep (default: 48)
            sparse_mode: "xk_sr" (V offload) or "xkv_sr" (both SVD)
        """
        self.chunk_size = chunk_size
        self.sparse_budget = sparse_budget
        self.num_outliers = num_outliers
        self.sparse_mode = sparse_mode

        # Landmarks: mean of each chunk (post-RoPE keys)
        # Shape: (num_chunks, num_kv_heads, head_dim)
        self.landmarks: Optional[Tensor] = None

        # Outlier KV cache (static, stored on GPU)
        # Shape: (num_outliers * chunk_size, num_kv_heads, head_dim)
        self.outlier_k: Optional[Tensor] = None
        self.outlier_v: Optional[Tensor] = None
        self.outlier_indices: Optional[Tensor] = None

        # For xk_sr mode: full V offloaded to CPU (chunked)
        # Shape: (num_chunks, chunk_size, num_kv_heads, head_dim)
        self.v_offload_cpu: Optional[Tensor] = None

        # Cache for temporal locality (previous selected indices)
        self.cached_indices: Optional[Tensor] = None

        # Total number of chunks
        self.num_chunks: int = 0
    
    @torch.no_grad()
    def compute_landmarks_and_outliers(
        self,
        k_post_rope: Tensor,
        v: Tensor,
    ) -> None:
        """
        Compute landmarks and detect outliers during prefill.

        For xk_sr mode: offload full V to CPU (not SVD compressed)
        For xkv_sr mode: V stays on GPU (will be SVD compressed)

        Args:
            k_post_rope: Post-RoPE key cache, shape (N, num_kv_heads, head_dim)
            v: Value cache, shape (N, num_kv_heads, head_dim)
        """
        N, num_kv_heads, head_dim = k_post_rope.shape

        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - N % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            k_padded = torch.nn.functional.pad(k_post_rope, (0, 0, 0, 0, 0, pad_len))
            v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
        else:
            k_padded = k_post_rope
            v_padded = v

        N_padded = k_padded.shape[0]
        self.num_chunks = N_padded // self.chunk_size

        # Reshape into chunks: (num_chunks, chunk_size, num_kv_heads, head_dim)
        k_chunks = k_padded.view(self.num_chunks, self.chunk_size, num_kv_heads, head_dim)
        v_chunks = v_padded.view(self.num_chunks, self.chunk_size, num_kv_heads, head_dim)

        # For xk_sr mode: offload full V to CPU (original ShadowKV style)
        if self.sparse_mode == "xk_sr":
            self.v_offload_cpu = v_chunks.to("cpu", non_blocking=True)

        # Compute landmarks: mean of each chunk
        # Shape: (num_chunks, num_kv_heads, head_dim)
        self.landmarks = k_chunks.mean(dim=1)

        # Detect outliers: compute cosine similarity within each chunk
        # For each token in chunk, compute cosine similarity with chunk mean
        # Shape: (num_chunks, chunk_size, num_kv_heads)
        landmarks_expanded = self.landmarks.unsqueeze(1)  # (num_chunks, 1, num_kv_heads, head_dim)

        # Normalize for cosine similarity
        k_norm = torch.nn.functional.normalize(k_chunks, dim=-1)
        landmarks_norm = torch.nn.functional.normalize(landmarks_expanded, dim=-1)

        # Cosine similarity: (num_chunks, chunk_size, num_kv_heads)
        cos_sim = (k_norm * landmarks_norm).sum(dim=-1)

        # Get minimum cosine similarity per chunk: (num_chunks, num_kv_heads)
        min_cos_sim = cos_sim.min(dim=1).values

        # Average across heads to get per-chunk score: (num_chunks,)
        chunk_scores = min_cos_sim.mean(dim=-1)

        # Find outlier chunks (lowest cosine similarity)
        num_outliers = min(self.num_outliers, self.num_chunks)
        outlier_indices = chunk_scores.topk(num_outliers, largest=False).indices
        self.outlier_indices = outlier_indices.sort().values

        # Extract outlier KV
        # Gather outlier chunks: (num_outliers, chunk_size, num_kv_heads, head_dim)
        outlier_k_chunks = k_chunks[self.outlier_indices]
        outlier_v_chunks = v_chunks[self.outlier_indices]

        # Flatten: (num_outliers * chunk_size, num_kv_heads, head_dim)
        self.outlier_k = outlier_k_chunks.reshape(-1, num_kv_heads, head_dim)
        self.outlier_v = outlier_v_chunks.reshape(-1, num_kv_heads, head_dim)

        # Remove outliers from landmarks (set to zero so they won't be selected)
        # Create mask for non-outlier chunks
        non_outlier_mask = torch.ones(self.num_chunks, device=k_post_rope.device, dtype=torch.bool)
        non_outlier_mask[self.outlier_indices] = False
        self.non_outlier_mask = non_outlier_mask
    
    @torch.no_grad()
    def select_sparse_chunks(
        self,
        q: Tensor,
        scale: float,
    ) -> Tensor:
        """
        Select sparse chunk indices based on query-landmark attention.

        Args:
            q: Query tensor, shape (1, num_heads, head_dim) for single decode
            scale: Attention scale factor

        Returns:
            Selected chunk indices, shape (sparse_budget,)
        """
        if self.landmarks is None:
            raise RuntimeError("Landmarks not computed. Call compute_landmarks_and_outliers first.")

        # q: (1, num_heads, head_dim) -> (num_heads, head_dim)
        q = q.squeeze(0)
        num_heads = q.shape[0]
        num_kv_heads = self.landmarks.shape[1]
        head_ratio = num_heads // num_kv_heads

        # Expand landmarks for GQA: (num_chunks, num_kv_heads, head_dim)
        # Compute attention scores: Q @ L^T
        # landmarks: (num_chunks, num_kv_heads, head_dim)
        # q: (num_heads, head_dim) -> (num_kv_heads, head_ratio, head_dim)
        q_grouped = q.view(num_kv_heads, head_ratio, -1)  # (num_kv_heads, head_ratio, head_dim)

        # Compute attention: (num_kv_heads, head_ratio, num_chunks)
        attn_scores = torch.einsum('khd,ckd->khc', q_grouped, self.landmarks) * scale

        # Softmax per head group: (num_kv_heads, head_ratio, num_chunks)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Sum across query heads, max across kv heads: (num_chunks,)
        chunk_importance = attn_probs.sum(dim=1).max(dim=0).values

        # Mask out outlier chunks (already stored separately)
        chunk_importance = chunk_importance.masked_fill(~self.non_outlier_mask, float('-inf'))

        # Select top-k chunks
        budget = min(self.sparse_budget, self.non_outlier_mask.sum().item())
        selected_indices = chunk_importance.topk(int(budget), largest=True).indices

        # Sort indices for efficient memory access
        selected_indices = selected_indices.sort().values

        # Update cache for temporal locality
        self.cached_indices = selected_indices

        return selected_indices

    def get_outlier_kv(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get outlier KV cache (static, always included in attention)."""
        return self.outlier_k, self.outlier_v

    def get_num_outlier_tokens(self) -> int:
        """Get number of outlier tokens."""
        if self.outlier_k is None:
            return 0
        return self.outlier_k.shape[0]

    @torch.no_grad()
    def load_selected_v_from_cpu(
        self,
        selected_indices: Tensor,
        device: torch.device,
    ) -> Optional[Tensor]:
        """
        Load selected V chunks from CPU to GPU (for xk_sr mode).

        Args:
            selected_indices: Chunk indices to load, shape (budget,)
            device: Target device (GPU)

        Returns:
            Selected V tensor, shape (budget * chunk_size, num_kv_heads, head_dim)
            Returns None if not in xk_sr mode or V not offloaded.
        """
        if self.sparse_mode != "xk_sr" or self.v_offload_cpu is None:
            return None

        # Gather selected chunks from CPU: (budget, chunk_size, num_kv_heads, head_dim)
        selected_v_chunks = self.v_offload_cpu[selected_indices.cpu()]

        # Move to GPU and flatten: (budget * chunk_size, num_kv_heads, head_dim)
        num_kv_heads = selected_v_chunks.shape[2]
        head_dim = selected_v_chunks.shape[3]
        selected_v = selected_v_chunks.to(device, non_blocking=True).reshape(-1, num_kv_heads, head_dim)

        return selected_v

    def has_v_offload(self) -> bool:
        """Check if V is offloaded to CPU (xk_sr mode)."""
        return self.sparse_mode == "xk_sr" and self.v_offload_cpu is not None

    def clear(self) -> None:
        """Clear all stored data."""
        self.landmarks = None
        self.outlier_k = None
        self.outlier_v = None
        self.outlier_indices = None
        self.cached_indices = None
        self.v_offload_cpu = None
        self.num_chunks = 0

