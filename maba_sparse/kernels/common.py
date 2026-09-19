
import math
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def normalize_keys(k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.linalg.vector_norm(k, dim=-1, keepdim=True)
    return k / (norm + eps)


def validate_dgda_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    check_finite: bool = False,
    log_alpha: Optional[torch.Tensor] = None,
) -> None:
    if alpha is None and log_alpha is None:
        raise ValueError("Either alpha or log_alpha must be provided")
    if b is None or w is None:
        raise ValueError("Both b and w gate tensors must be provided")

    device = q.device
    tensors_to_check = [("k", k), ("v", v), ("b", b), ("w", w)]
    if alpha is not None:
        tensors_to_check.append(("alpha", alpha))
    if log_alpha is not None:
        tensors_to_check.append(("log_alpha", log_alpha))

    for name, t in tensors_to_check:
        if t.device != device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {device} and {t.device} for {name}"
            )
    if initial_state is not None and initial_state.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got {device} and "
            f"{initial_state.device} for initial_state"
        )

    dtype = q.dtype
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    if dtype not in supported_dtypes:
        raise TypeError(f"Unsupported dtype {dtype}. Supported: {supported_dtypes}")

    for name, t in tensors_to_check:
        if t.dtype != dtype:
            raise TypeError(
                f"Tensor dtypes must match, but got {dtype} for q and {t.dtype} for {name}"
            )
    if initial_state is not None and initial_state.dtype != dtype:
        raise TypeError(
            f"Tensor dtypes must match, but got {dtype} for q and {initial_state.dtype} for initial_state"
        )

    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B, H, L, dk], got shape {list(q.shape)}")
    B, H, L, dk = q.shape

    if k.shape != (B, H, L, dk):
        raise ValueError(f"k shape {list(k.shape)} must match q shape {(B, H, L, dk)}")
    if v.dim() != 4 or v.shape[:3] != (B, H, L):
        raise ValueError(f"v shape {list(v.shape)} must have first 3 dims {(B, H, L)}")
    dv = v.shape[3]

    if alpha is not None and alpha.shape != (B, H, L, dk):
        raise ValueError(f"alpha shape {list(alpha.shape)} must match {(B, H, L, dk)}")
    if log_alpha is not None and log_alpha.shape != (B, H, L, dk):
        raise ValueError(f"log_alpha shape {list(log_alpha.shape)} must match {(B, H, L, dk)}")
    if b.shape != (B, H, L, dk):
        raise ValueError(f"b shape {list(b.shape)} must match {(B, H, L, dk)}")
    if w.shape != (B, H, L, dv):
        raise ValueError(f"w shape {list(w.shape)} must match {(B, H, L, dv)}")

    if initial_state is not None:
        if initial_state.shape != (B, H, dk, dv):
            raise ValueError(
                f"initial_state shape {list(initial_state.shape)} must match {(B, H, dk, dv)}"
            )

    if check_finite:
        for name, t in [("q", q)] + tensors_to_check:
            if not torch.isfinite(t).all():
                raise ValueError(f"Tensor {name} contains non-finite values (NaN or Inf)")


def validate_dgda_step_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    check_finite: bool = False,
    log_alpha: Optional[torch.Tensor] = None,
) -> None:
    if alpha is None and log_alpha is None:
        raise ValueError("Either alpha or log_alpha must be provided")
    if b is None or w is None:
        raise ValueError("Both b and w gate tensors must be provided")

    device = q.device
    tensors_to_check = [("k", k), ("v", v), ("b", b), ("w", w)]
    if alpha is not None:
        tensors_to_check.append(("alpha", alpha))
    if log_alpha is not None:
        tensors_to_check.append(("log_alpha", log_alpha))

    for name, t in tensors_to_check:
        if t.device != device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {device} and {t.device} for {name}"
            )
    if state is not None and state.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got {device} and {state.device} for state"
        )

    dtype = q.dtype
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    if dtype not in supported_dtypes:
        raise TypeError(f"Unsupported dtype {dtype}. Supported: {supported_dtypes}")
    for name, t in tensors_to_check:
        if t.dtype != dtype:
            raise TypeError(
                f"Tensor dtypes must match, but got {dtype} for q and {t.dtype} for {name}"
            )
    if state is not None and state.dtype != dtype:
        raise TypeError(
            f"Tensor dtypes must match, but got {dtype} for q and {state.dtype} for state"
        )

    q_3d = q.squeeze(2) if (q.dim() == 4 and q.shape[2] == 1) else q
    if q_3d.dim() != 3:
        raise ValueError(f"q must be [B, H, dk] or [B, H, 1, dk], got {list(q.shape)}")

    B, H, dk = q_3d.shape
    v_3d = v.squeeze(2) if (v.dim() == 4 and v.shape[2] == 1) else v
    if v_3d.dim() != 3 or v_3d.shape[:2] != (B, H):
        raise ValueError(f"v must be [B, H, dv] matching B={B}, H={H}, got {list(v.shape)}")
    dv = v_3d.shape[-1]

    if state is not None:
        if state.shape != (B, H, dk, dv):
            raise ValueError(f"state shape {list(state.shape)} must match {(B, H, dk, dv)}")

    if check_finite:
        for name, t in [("q", q)] + tensors_to_check:
            if not torch.isfinite(t).all():
                raise ValueError(f"Tensor {name} contains non-finite values")


def validate_indexer_inputs(
    k_idx: torch.Tensor,
    q_idx: Optional[torch.Tensor] = None,
    centroids: Optional[torch.Tensor] = None,
    check_finite: bool = False,
) -> None:
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be 3D [B, L, d_idx], got shape {list(k_idx.shape)}")
    B, L, d_idx = k_idx.shape

    if q_idx is not None:
        if q_idx.device != k_idx.device:
            raise RuntimeError("Expected all tensors to be on the same device")
        if q_idx.dim() != 3 or q_idx.shape[0] != B or q_idx.shape[2] != d_idx:
            raise ValueError(
                f"q_idx shape {list(q_idx.shape)} incompatible with k_idx {(B, L, d_idx)}"
            )

    if centroids is not None:
        if centroids.device != k_idx.device:
            raise RuntimeError("Expected all tensors to be on the same device")
        if centroids.dim() != 3 or centroids.shape[0] != B or centroids.shape[2] != d_idx:
            raise ValueError(
                f"centroids shape {list(centroids.shape)} incompatible with B={B}, d_idx={d_idx}"
            )

    if check_finite:
        if not torch.isfinite(k_idx).all():
            raise ValueError("k_idx contains non-finite values")
        if q_idx is not None and not torch.isfinite(q_idx).all():
            raise ValueError("q_idx contains non-finite values")
        if centroids is not None and not torch.isfinite(centroids).all():
            raise ValueError("centroids contains non-finite values")


def validate_superposition_inputs(
    o_local: torch.Tensor,
    o_sparse: torch.Tensor,
    o_hca: torch.Tensor,
    gate_logits_or_weights: Optional[torch.Tensor] = None,
    check_finite: bool = False,
) -> None:
    device = o_local.device
    if o_sparse.device != device or o_hca.device != device:
        raise RuntimeError("Expected all tensors to be on the same device")

    if o_local.shape != o_sparse.shape or o_local.shape != o_hca.shape:
        raise ValueError(
            f"Shape mismatch: o_local={list(o_local.shape)}, "
            f"o_sparse={list(o_sparse.shape)}, o_hca={list(o_hca.shape)}"
        )

    if gate_logits_or_weights is not None:
        if gate_logits_or_weights.device != device:
            raise RuntimeError("Expected all tensors to be on the same device")
        B = o_local.shape[0]
        if gate_logits_or_weights.shape[-1] != 3:
            raise ValueError(
                f"Gate tensor last dimension must be 3, got {gate_logits_or_weights.shape[-1]}"
            )
        if gate_logits_or_weights.shape[0] != B:
            raise ValueError(f"Gate tensor batch size must match {B}")

    if check_finite:
        for name, t in [("o_local", o_local), ("o_sparse", o_sparse), ("o_hca", o_hca)]:
            if not torch.isfinite(t).all():
                raise ValueError(f"Tensor {name} contains non-finite values")


def ref_dgda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_dgda_prefill_inputs(q, k, v, alpha, b, w, initial_state, log_alpha=log_alpha)
    if alpha is None and log_alpha is not None:
        alpha = torch.exp(log_alpha)

    B, H, L, dk = q.shape
    dv = v.shape[-1]

    if initial_state is None:
        S = torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
    else:
        S = initial_state.clone()

    if L == 0:
        return torch.empty(B, H, 0, dv, dtype=q.dtype, device=q.device), S

    outs = []
    for t in range(L):
        S_decay = alpha[:, :, t].unsqueeze(-1) * S
        beta_t = (b[:, :, t] * k[:, :, t]).unsqueeze(-2)
        pred = torch.matmul(beta_t, S_decay)
        u_t = (w[:, :, t] * v[:, :, t]).unsqueeze(-2)
        delta = u_t - pred
        S = S_decay + torch.matmul(k[:, :, t].unsqueeze(-1), delta)
        o_t = torch.matmul(q[:, :, t].unsqueeze(-2), S).squeeze(-2)
        outs.append(o_t)

    out = torch.stack(outs, dim=2)
    return out, S


def ref_dgda_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_dgda_step_inputs(q, k, v, alpha, b, w, state, log_alpha=log_alpha)
    if alpha is None and log_alpha is not None:
        alpha = torch.exp(log_alpha)

    if q.dim() == 4 and q.shape[2] == 1:
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        alpha = alpha.squeeze(2)
        b = b.squeeze(2)
        w = w.squeeze(2)

    B, H, dk = q.shape
    dv = v.shape[-1]

    if state is None:
        S = torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
    else:
        S = state.clone()

    S_decay = alpha.unsqueeze(-1) * S
    beta = (b * k).unsqueeze(-2)
    pred = torch.matmul(beta, S_decay)
    u = (w * v).unsqueeze(-2)
    delta = u - pred
    new_state = S_decay + torch.matmul(k.unsqueeze(-1), delta)
    out = torch.matmul(q.unsqueeze(-2), new_state).squeeze(-2)
    return out, new_state


def ref_dgda_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ref_dgda_prefill(
        q=q, k=k, v=v, alpha=alpha, b=b, w=w,
        chunk_size=16, initial_state=initial_state, log_alpha=log_alpha, **kwargs
    )


def ref_compute_centroids(
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

    if L == 0:
        return torch.empty(B, 0, d_idx, dtype=k_idx.dtype, device=k_idx.device)

    nb = (L + block_size - 1) // block_size
    pad = nb * block_size - L
    kp = F.pad(k_idx, (0, 0, 0, pad), value=0.0) if pad > 0 else k_idx
    kb = kp.view(B, nb, block_size, d_idx)
    return 0.5 * (kb.mean(dim=2) + kb.max(dim=2)[0])


def ref_index_topk(
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
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, L, d_idx = q_idx.shape
    nb = centroids.shape[1]

    if L == 0 or nb == 0:
        ak = min(top_k, nb)
        empty_idx = torch.zeros(B, L, ak, dtype=torch.long, device=q_idx.device)
        if return_scores:
            empty_scores = torch.zeros(B, L, nb, dtype=q_idx.dtype, device=q_idx.device)
            return empty_idx, empty_scores
        return empty_idx

    scale = 1.0 / math.sqrt(d_idx)
    s = torch.einsum("bld,bnd->bln", q_idx * scale, centroids)

    qi_idx = torch.arange(L, device=q_idx.device).unsqueeze(1) // block_size
    ni_idx = torch.arange(nb, device=centroids.device).unsqueeze(0)
    dist = (qi_idx - ni_idx).abs().float()
    pen = lam * torch.log(1.0 + dist)
    sc = s - pen.unsqueeze(0)

    msk = ni_idx > qi_idx
    sc = sc.masked_fill(msk.unsqueeze(0), float("-inf"))

    ak = min(top_k, nb)
    _, idx = torch.topk(sc, k=ak, dim=-1)

    if return_scores:
        return idx, sc
    return idx


def ref_stream_superposition(
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

    if gate_weights is None:
        if gate_logits is None:
            raise ValueError("Either gate_logits or gate_weights must be provided")
        g = F.softmax(gate_logits, dim=-1)
    else:
        g = gate_weights

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

reference_dgda_prefill = ref_dgda_prefill
reference_dgda_step = ref_dgda_step
reference_dgda_sequential = ref_dgda_sequential
reference_compute_centroids = ref_compute_centroids
reference_index_topk = ref_index_topk
reference_stream_superposition = ref_stream_superposition

ref_centroids = ref_compute_centroids
ref_topk = ref_index_topk
ref_superposition = ref_stream_superposition

__all__ = [
    "normalize_keys",
    "validate_dgda_prefill_inputs",
    "validate_dgda_step_inputs",
    "validate_indexer_inputs",
    "validate_superposition_inputs",
    "ref_dgda_prefill",
    "ref_dgda_step",
    "ref_dgda_sequential",
    "ref_compute_centroids",
    "ref_index_topk",
    "ref_stream_superposition",
    "reference_dgda_prefill",
    "reference_dgda_step",
    "reference_dgda_sequential",
    "reference_compute_centroids",
    "reference_index_topk",
    "reference_stream_superposition",
    "ref_centroids",
    "ref_topk",
    "ref_superposition",
]
