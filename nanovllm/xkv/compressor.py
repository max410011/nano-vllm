"""
SVD compression functions for xKV.

This module implements the core compression algorithms:
- fake_svd: SVD-based compression (truncate and reconstruct)
- slerp_merge: SLERP-based compression for layer groups of size 2
"""

import torch


def fake_svd(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Perform fake SVD: SVD -> Truncate -> Multiply back.
    
    This function performs SVD decomposition on the input tensor,
    truncates to the specified rank, and reconstructs the tensor.
    The result is an approximation of the original tensor with reduced rank.
    
    Args:
        tensor: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        rank: Number of singular values to keep
        
    Returns:
        Approximated tensor of the same shape as input
    """
    bs, nh, sl, hd = tensor.shape
    
    # Reshape: (bs, nh, sl, hd) -> (bs, sl, nh*hd)
    tensor_reshaped = tensor.transpose(1, 2).reshape(bs, sl, nh * hd)
    
    # Step 1: Perform SVD
    U, S, V_h = torch.linalg.svd(tensor_reshaped, full_matrices=False)
    U_trunc = U[:, :, :rank]
    S_trunc = S[:, :rank]
    Vt_trunc = V_h[:, :rank, :]
    
    # Step 2: Split singular values evenly between U and V
    sqrt_S = torch.sqrt(S_trunc)
    U_scaled = U_trunc * sqrt_S.unsqueeze(1)
    Vt_scaled = sqrt_S.unsqueeze(-1) * Vt_trunc
    
    # Step 3: Multiply back to approximate the original tensor
    approx_tensor = torch.matmul(U_scaled, Vt_scaled)
    
    # Reshape back: (bs, sl, nh*hd) -> (bs, nh, sl, hd)
    approx_tensor = approx_tensor.view(bs, sl, nh, hd).transpose(1, 2)
    
    return approx_tensor


def slerp_merge_rows_batch(
    X1: torch.Tensor,
    X2: torch.Tensor,
    t: float = 0.5,
    gamma: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized row-wise SLERP merge of X1, X2 in shape (L, d), returning E of shape (L, d).
    
    Args:
        X1: First input tensor of shape (L, d)
        X2: Second input tensor of shape (L, d)
        t: Interpolation parameter (0 to 1)
        gamma: Divergence threshold parameter
        
    Returns:
        Tuple of (E, diverge_mask, norm1, norm2)
    """
    norm1 = X1.norm(dim=1, keepdim=True)
    norm2 = X2.norm(dim=1, keepdim=True)

    u1 = X1 / norm1
    u2 = X2 / norm2

    dot_val = (u1 * u2).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    Omega = torch.acos(dot_val)
    sinOmega = torch.sin(Omega)

    d_min = Omega.min()
    d_max = Omega.max()
    threshold = d_min + (d_max - d_min) * gamma
    diverge_mask = Omega > threshold

    parallel_mask = (Omega < 1e-7)

    alpha = torch.sin((1.0 - t) * Omega) / sinOmega
    beta = torch.sin(t * Omega) / sinOmega

    E_slerp = alpha * u1 + beta * u2

    E_linear = (1.0 - t) * X1 + t * X2
    fallback_mask = parallel_mask

    fallback_mask_full = fallback_mask.expand(-1, X1.shape[1])
    E = torch.where(fallback_mask_full, E_linear, E_slerp)

    return E, diverge_mask, norm1, norm2


def fake_minicache_merge(
    X1: torch.Tensor,
    X2: torch.Tensor,
    t: float = 0.5,
    gamma: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SLERP-based merge for two tensors (used for layer group size 2).
    
    Args:
        X1: First input tensor of shape (L, d)
        X2: Second input tensor of shape (L, d)
        t: Interpolation parameter (0 to 1)
        gamma: Divergence threshold parameter
        
    Returns:
        Tuple of (E1, E2) - the merged tensors
    """
    E, diverge_mask, n1, n2 = slerp_merge_rows_batch(X1, X2, t=t, gamma=gamma)
    diverge_mask = diverge_mask.squeeze(-1)
    E1 = torch.clone(E) * n1
    E1[~diverge_mask] = X1[~diverge_mask]
    E2 = torch.clone(E) * n2
    E2[~diverge_mask] = X2[~diverge_mask]
    return E1, E2

