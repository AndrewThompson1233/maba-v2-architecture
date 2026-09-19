
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from maba_sparse.kernels.common import (
    validate_indexer_inputs,
    validate_superposition_inputs,
)
from maba_sparse.kernels.dispatcher import register_kernel


@register_kernel("cpu", "compute_centroids")
def cpu_compute_centroids(
    k_idx: torch.Tensor,
    block_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be 3D [B, L, d_idx], got {list(k_idx.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, L, d_idx = k_idx.shape
    if d_idx <= 0:
        raise ValueError(f"d_idx must be positive, got {d_idx}")

    orig_dtype = k_idx.dtype
    orig_device = k_idx.device

    if L == 0:
        return torch.empty(B, 0, d_idx, dtype=orig_dtype, device=orig_device)

    k_f = k_idx.float() if orig_dtype in (torch.float16, torch.bfloat16) else k_idx

    nb = (L + block_size - 1) // block_size
    pad = nb * block_size - L
    if pad > 0:
        kp = F.pad(k_f, (0, 0, 0, pad), value=0.0)
    else:
        kp = k_f

    kb = kp.view(B, nb, block_size, d_idx)
    mean_val = kb.mean(dim=2)
    max_val = kb.amax(dim=2)
    c = 0.5 * (mean_val + max_val)

    return c.to(orig_dtype)


@register_kernel("cpu", "index_topk")
def cpu_index_topk(
    q_idx: torch.Tensor,
    centroids: torch.Tensor,
    lambda_dist: float = 0.5,
    top_k: int = 32,
    block_size: int = 64,
    return_scores: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    lam = kwargs.get("dist_lambda", lambda_dist)

    if q_idx.device != centroids.device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, got {q_idx.device} and {centroids.device}"
        )
    if q_idx.dim() != 3 or centroids.dim() != 3:
        raise ValueError(f"q_idx and centroids must be 3D, got {q_idx.dim()}D and {centroids.dim()}D")
    if q_idx.shape[0] != centroids.shape[0] or q_idx.shape[2] != centroids.shape[2]:
        raise ValueError(
            f"Dimension mismatch between q_idx {list(q_idx.shape)} and centroids {list(centroids.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, L, d_idx = q_idx.shape
    nb = centroids.shape[1]
    orig_dtype = q_idx.dtype

    if L == 0 or nb == 0:
        ak = min(top_k, nb)
        empty_idx = torch.zeros(B, L, ak, dtype=torch.long, device=q_idx.device)
        if return_scores:
            empty_scores = torch.zeros(B, L, nb, dtype=orig_dtype, device=q_idx.device)
            return empty_idx, empty_scores
        return empty_idx

    q_f = q_idx.float() if orig_dtype in (torch.float16, torch.bfloat16) else q_idx
    c_f = centroids.float() if orig_dtype in (torch.float16, torch.bfloat16) else centroids

    scale = 1.0 / math.sqrt(d_idx)
    s = torch.bmm(q_f * scale, c_f.transpose(1, 2))

    qi_idx = torch.arange(L, device=q_idx.device).unsqueeze(1) // block_size
    ni_idx = torch.arange(nb, device=centroids.device).unsqueeze(0)
    dist = (qi_idx - ni_idx).abs().float()
    pen = lam * torch.log1p(dist)

    scores = s - pen.unsqueeze(0)
    causal_mask = ni_idx > qi_idx
    scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    ak = min(top_k, nb)
    top_indices = torch.topk(scores, k=ak, dim=-1, largest=True, sorted=True).indices

    if return_scores:
        return top_indices.to(torch.long), scores.to(orig_dtype)
    return top_indices.to(torch.long)


@register_kernel("cpu", "stream_superposition")
def cpu_stream_superposition(
    o_local: torch.Tensor,
    o_sparse: torch.Tensor,
    o_hca: torch.Tensor,
    gate_logits: Optional[torch.Tensor] = None,
    gate_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    gate_t = gate_logits if gate_logits is not None else gate_weights
    validate_superposition_inputs(o_local, o_sparse, o_hca, gate_t)

    if o_local.shape[2] == 0:
        return torch.empty_like(o_local)

    orig_dtype = o_local.dtype

    if gate_weights is None:
        if gate_logits is None:
            raise ValueError("Either gate_logits or gate_weights must be provided")
        g = F.softmax(gate_logits.float(), dim=-1).to(orig_dtype)
    else:
        g = gate_weights.to(orig_dtype)

    if g.dim() == 3:
        gl = g[:, :, 0:1].unsqueeze(1)
        gs = g[:, :, 1:2].unsqueeze(1)
        gh = g[:, :, 2:3].unsqueeze(1)
    elif g.dim() == 4:
        gl = g[:, :, :, 0:1]
        gs = g[:, :, :, 1:2]
        gh = g[:, :, :, 2:3]
    else:
        raise ValueError(f"Gate tensor must be 3D or 4D, got {g.dim()}D")

    return gl * o_local + gs * o_sparse + gh * o_hca
