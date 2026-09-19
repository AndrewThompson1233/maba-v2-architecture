
import math
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from maba_sparse.kernels.common import (
    validate_indexer_inputs,
    validate_superposition_inputs,
)
from maba_sparse.kernels.cpu_indexer import cpu_stream_superposition
from maba_sparse.kernels.dispatcher import register_kernel

DEFAULT_STATIC_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


from .xla_dgda import get_static_bucket_length


@register_kernel("xla", "compute_centroids")
def xla_compute_centroids(
    k_idx: torch.Tensor,
    block_size: int = 64,
    static_seq_len: Optional[int] = None,
    use_bucketing: bool = False,
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

    if static_seq_len is not None:
        target_len = ((static_seq_len + block_size - 1) // block_size) * block_size
    elif use_bucketing:
        target_len = get_static_bucket_length(L, buckets=DEFAULT_STATIC_BUCKETS, chunk_size=block_size)
    else:
        target_len = ((L + block_size - 1) // block_size) * block_size

    pad_len = target_len - L
    if pad_len > 0:
        kp = F.pad(k_idx, (0, 0, 0, pad_len), value=0.0)
    else:
        kp = k_idx

    nb_static = target_len // block_size
    kb = kp.view(B, nb_static, block_size, d_idx)

    mean_val = kb.mean(dim=2)
    max_val = kb.amax(dim=2)
    c_static = 0.5 * (mean_val + max_val)

    nb_real = (L + block_size - 1) // block_size
    return c_static[:, :nb_real, :].to(orig_dtype)


@register_kernel("xla", "index_topk")
def xla_index_topk(
    q_idx: torch.Tensor,
    centroids: torch.Tensor,
    lambda_dist: float = 0.5,
    top_k: int = 32,
    block_size: int = 64,
    static_seq_len: Optional[int] = None,
    use_bucketing: bool = False,
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
    nb_real = centroids.shape[1]
    orig_dtype = q_idx.dtype

    if L == 0 or nb_real == 0:
        ak = min(top_k, nb_real)
        empty_idx = torch.zeros(B, L, ak, dtype=torch.long, device=q_idx.device)
        if return_scores:
            empty_scores = torch.zeros(B, L, nb_real, dtype=orig_dtype, device=q_idx.device)
            return empty_idx, empty_scores
        return empty_idx

    if static_seq_len is not None:
        target_len = ((static_seq_len + block_size - 1) // block_size) * block_size
    elif use_bucketing:
        target_len = get_static_bucket_length(L, buckets=DEFAULT_STATIC_BUCKETS, chunk_size=block_size)
    else:
        target_len = ((L + block_size - 1) // block_size) * block_size
    nb_static = target_len // block_size

    pad_q = target_len - L
    pad_c = nb_static - nb_real

    qp = F.pad(q_idx.float(), (0, 0, 0, pad_q), value=0.0) if pad_q > 0 else q_idx.float()
    cp = F.pad(centroids.float(), (0, 0, 0, pad_c), value=0.0) if pad_c > 0 else centroids.float()

    scale = 1.0 / math.sqrt(d_idx)
    s_static = torch.bmm(qp * scale, cp.transpose(1, 2))

    q_blk = torch.arange(target_len, device=q_idx.device).unsqueeze(1) // block_size
    k_blk = torch.arange(nb_static, device=centroids.device).unsqueeze(0)
    dist = (q_blk - k_blk).abs().float()
    pen = lam * torch.log1p(dist)

    causal_mask = (k_blk > q_blk) | (k_blk >= nb_real)
    scores_static = (s_static - pen.unsqueeze(0)).masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    ak = min(top_k, nb_real)
    scores_real = scores_static[:, :L, :nb_real]
    top_indices = torch.topk(scores_real, k=ak, dim=-1, largest=True, sorted=True).indices

    if return_scores:
        return top_indices.to(torch.long), scores_real.to(orig_dtype)
    return top_indices.to(torch.long)


@register_kernel("xla", "stream_superposition")
def xla_stream_superposition(
    o_local: torch.Tensor,
    o_sparse: torch.Tensor,
    o_hca: torch.Tensor,
    gate_logits: Optional[torch.Tensor] = None,
    gate_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return cpu_stream_superposition(
        o_local=o_local,
        o_sparse=o_sparse,
        o_hca=o_hca,
        gate_logits=gate_logits,
        gate_weights=gate_weights,
        **kwargs,
    )
