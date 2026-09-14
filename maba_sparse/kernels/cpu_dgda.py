"""CPU Vectorized Decoupled Gated Delta Attention (DGDA) Kernel Engine.

This module provides high-performance vectorized execution on CPU by:
1. Upcasting float16 / bfloat16 tensors to float32 to eliminate underflow and emulate-mode stalls.
2. Vectorizing intra-chunk log cumulative decay, interaction matrix L, and Neumann series inversion
   across all chunks simultaneously in 5D tensors.
3. Running a minimal, tight inter-chunk recurrent state propagation scan.
4. Leveraging PyTorch's OpenMP multi-threading across batch and heads.
"""

from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from maba_sparse.kernels.common import (
    validate_dgda_prefill_inputs,
    validate_dgda_step_inputs,
)

# ==============================================================================
# OpenMP Multi-Threading Utilities
# ==============================================================================


def get_cpu_num_threads() -> int:
    """Return the current number of intra-op threads utilized by PyTorch CPU BLAS."""
    return torch.get_num_threads()


def set_cpu_num_threads(num_threads: int) -> None:
    """Set the number of intra-op threads for PyTorch CPU execution."""
    if num_threads <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    torch.set_num_threads(num_threads)


@contextmanager
def cpu_threads(num_threads: int):
    """Context manager for temporarily overriding PyTorch CPU thread count."""
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(num_threads)
        yield
    finally:
        torch.set_num_threads(prev_threads)


# ==============================================================================
# CPU Vectorized DGDA Prefill Kernel
# ==============================================================================


def cpu_dgda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    inversion_method: str = "adaptive",
    adaptive_tol: float = 7e-5,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute vectorized DGDA chunkwise prefill on CPU with float32 safety.

    Args:
        q: [B, H, L, dk]
        k: [B, H, L, dk] (normalized)
        v: [B, H, L, dv]
        alpha: [B, H, L, dk]
        b: [B, H, L, dk]
        w: [B, H, L, dv]
        chunk_size: Chunk size C (default 16).
        initial_state: Optional initial state [B, H, dk, dv].
        log_alpha: Optional precomputed log(alpha).
        inversion_method: "adaptive", "neumann", or "exact".
        adaptive_tol: Threshold for adaptive triangular solve fallback.

    Returns:
        Tuple of (out [B, H, L, dv], final_state [B, H, dk, dv]).
    """
    validate_dgda_prefill_inputs(q, k, v, alpha, b, w, initial_state)

    orig_dt = q.dtype
    # Float32 upcasting safety on CPU
    if orig_dt in (torch.float16, torch.bfloat16):
        q = q.float()
        k = k.float()
        v = v.float()
        alpha = alpha.float()
        b = b.float()
        w = w.float()
        if initial_state is not None:
            initial_state = initial_state.float()
        if log_alpha is not None:
            log_alpha = log_alpha.float()

    B, H, L, dk = q.shape
    dv = v.shape[-1]

    if L == 0:
        empty_out = torch.empty(B, H, 0, dv, dtype=orig_dt, device=q.device)
        state = (
            initial_state.clone().to(orig_dt)
            if initial_state is not None
            else torch.zeros(B, H, dk, dv, dtype=orig_dt, device=q.device)
        )
        return empty_out, state

    if log_alpha is None:
        log_alpha = torch.log(alpha.clamp(min=1e-20, max=1.0))

    cs = (
        torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
        if initial_state is None
        else initial_state.clone()
    )

    n_chunks = L // chunk_size
    rem = L % chunk_size
    outs = []

    # Vectorized batch 5D tensor precomputation for all full chunks
    if n_chunks > 0:
        L_main = n_chunks * chunk_size
        qc = q[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)
        kc = k[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)
        vc = v[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dv)
        bc = b[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)
        wc = w[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dv)
        lac = log_alpha[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)

        # Vectorized intra-chunk cumulative log-decay
        cla = torch.cumsum(lac, dim=-2)
        diff = cla.unsqueeze(-2) - cla.unsqueeze(-3)
        dec = torch.exp(torch.clamp(diff, max=0.0))

        # Intra-chunk interaction matrix L
        bk = (bc * kc).unsqueeze(-2)
        ks = kc.unsqueeze(-3)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

        # Neumann series order 3 inversion
        eye_c = torch.eye(chunk_size, dtype=q.dtype, device=q.device).view(1, 1, 1, chunk_size, chunk_size)
        eye_2d = torch.eye(chunk_size, dtype=q.dtype, device=q.device).view(1, 1, chunk_size, chunk_size)
        l2 = torch.matmul(l_mat, l_mat)
        l3 = torch.matmul(l2, l_mat)
        inv_l = eye_c - l_mat + l2 - l3

        lam = torch.exp(cla)
        bh = (bc * kc) * lam
        wv = wc * vc

        clae = cla[..., -1:, :]
        dte = torch.exp(clae - cla)
        kd = kc * dte
        ds0 = torch.exp(clae).squeeze(-2).unsqueeze(-1)  # [B, H, n_chunks, dk, 1]

        qs = qc.unsqueeze(-2)
        a_mat = torch.tril((qs * dec * ks).sum(dim=-1), diagonal=0)

        # Sequential inter-chunk recurrent state progression
        for c in range(n_chunks):
            ve = wv[:, :, c] - torch.matmul(bh[:, :, c], cs)
            u = torch.matmul(inv_l[:, :, c], ve)

            if inversion_method in ("exact", "adaptive"):
                r = ve - u - torch.matmul(l_mat[:, :, c], u)
                if (
                    inversion_method == "exact"
                    or r.abs().max() > adaptive_tol
                    or torch.isnan(u).any()
                    or torch.isinf(u).any()
                ):
                    u = torch.linalg.solve_triangular(eye_2d + l_mat[:, :, c], ve, upper=False)

            oi = torch.matmul(qc[:, :, c] * lam[:, :, c], cs)
            ot = torch.matmul(a_mat[:, :, c], u)
            outs.append(oi + ot)
            cs = ds0[:, :, c] * cs + torch.matmul(kd[:, :, c].transpose(-1, -2), u)

    # Remainder handling for sequences not divisible by chunk_size
    if rem > 0:
        s = n_chunks * chunk_size
        qr, kr, vr = q[:, :, s:], k[:, :, s:], v[:, :, s:]
        br, wr, lar = b[:, :, s:], w[:, :, s:], log_alpha[:, :, s:]
        t = rem

        cla = torch.cumsum(lar, dim=-2)
        diff = cla.unsqueeze(3) - cla.unsqueeze(2)
        dec = torch.exp(torch.clamp(diff, max=0.0))

        bk = (br * kr).unsqueeze(3)
        ks = kr.unsqueeze(2)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

        eye_t = torch.eye(t, dtype=q.dtype, device=q.device).view(1, 1, t, t)
        l2 = torch.matmul(l_mat, l_mat)
        l3 = torch.matmul(l2, l_mat)
        inv_l = eye_t - l_mat + l2 - l3

        lam = torch.exp(cla)
        bh = (br * kr) * lam
        ve = (wr * vr) - torch.matmul(bh, cs)
        u = torch.matmul(inv_l, ve)

        if inversion_method in ("exact", "adaptive"):
            r = ve - u - torch.matmul(l_mat, u)
            if (
                inversion_method == "exact"
                or r.abs().max() > adaptive_tol
                or torch.isnan(u).any()
                or torch.isinf(u).any()
            ):
                u = torch.linalg.solve_triangular(eye_t + l_mat, ve, upper=False)

        clae = cla[:, :, -1:, :]
        dte = torch.exp(clae - cla)
        kd = kr * dte
        ds0 = torch.exp(clae).squeeze(-2).unsqueeze(-1)

        oi = torch.matmul(qr * lam, cs)
        qs = qr.unsqueeze(3)
        a_mat = torch.tril((qs * dec * ks).sum(dim=-1), diagonal=0)
        ot = torch.matmul(a_mat, u)

        outs.append(oi + ot)
        cs = ds0 * cs + torch.matmul(kd.transpose(-1, -2), u)

    out = torch.cat(outs, dim=2)
    return out.to(orig_dt), cs.to(orig_dt)


# ==============================================================================
# CPU Fused Single-Step Decode Kernel
# ==============================================================================


def cpu_dgda_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute fused O(1) single-step autoregressive state transition on CPU.

    Args:
        q: [B, H, dk] or [B, H, 1, dk]
        k: [B, H, dk] or [B, H, 1, dk] (normalized)
        v: [B, H, dv] or [B, H, 1, dv]
        alpha: [B, H, dk] or [B, H, 1, dk]
        b: [B, H, dk] or [B, H, 1, dk]
        w: [B, H, dv] or [B, H, 1, dv]
        state: [B, H, dk, dv] or None

    Returns:
        Tuple of (out [B, H, dv], new_state [B, H, dk, dv]).
    """
    validate_dgda_step_inputs(q, k, v, alpha, b, w, state)

    orig_dt = q.dtype
    if orig_dt in (torch.float16, torch.bfloat16):
        q = q.float()
        k = k.float()
        v = v.float()
        alpha = alpha.float()
        b = b.float()
        w = w.float()
        if state is not None:
            state = state.float()

    if q.dim() == 4 and q.shape[2] == 1:
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        alpha = alpha.squeeze(2)
        b = b.squeeze(2)
        w = w.squeeze(2)

    B, H, dk = q.shape
    dv = v.shape[-1]

    sp = (
        torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
        if state is None
        else state.clone()
    )

    # Decay state along dk
    sd = alpha.unsqueeze(-1) * sp
    beta = b * k
    pred = torch.matmul(beta.unsqueeze(-2), sd).squeeze(-2)
    delta = (w * v) - pred

    new_state = sd + torch.matmul(k.unsqueeze(-1), delta.unsqueeze(-2))
    out = torch.matmul(q.unsqueeze(-2), new_state).squeeze(-2)

    return out.to(orig_dt), new_state.to(orig_dt)


__all__ = [
    "get_cpu_num_threads",
    "set_cpu_num_threads",
    "cpu_threads",
    "cpu_dgda_prefill",
    "cpu_dgda_step",
]
