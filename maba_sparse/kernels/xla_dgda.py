
from typing import Any, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

from maba_sparse.kernels.common import (
    validate_dgda_prefill_inputs,
    validate_dgda_step_inputs,
)
from maba_sparse.kernels.dispatcher import register_kernel

DEFAULT_STATIC_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


def get_static_bucket_length(
    seq_len: int,
    buckets: Sequence[int] = DEFAULT_STATIC_BUCKETS,
    chunk_size: int = 16,
) -> int:
    for b in buckets:
        if b >= seq_len:
            return ((b + chunk_size - 1) // chunk_size) * chunk_size
    return ((seq_len + chunk_size - 1) // chunk_size) * chunk_size


@register_kernel("xla", "dgda_prefill")
def xla_dgda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    static_seq_len: Optional[int] = None,
    use_bucketing: bool = False,
    neumann_order: int = 4,
    log_alpha: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha is None and log_alpha is not None:
        alpha = torch.exp(log_alpha)
    elif log_alpha is None and alpha is not None:
        log_alpha = torch.log(alpha.clamp(min=1e-20, max=1.0))

    validate_dgda_prefill_inputs(q, k, v, alpha, b, w, initial_state, log_alpha=log_alpha)

    orig_dtype = q.dtype
    orig_device = q.device
    B, H, L, dk = q.shape
    dv = v.shape[-1]

    if L == 0:
        empty_out = torch.empty(B, H, 0, dv, dtype=orig_dtype, device=orig_device)
        state = (
            initial_state.clone()
            if initial_state is not None
            else torch.zeros(B, H, dk, dv, dtype=orig_dtype, device=orig_device)
        )
        return empty_out, state

    if static_seq_len is not None:
        if static_seq_len < L:
            raise ValueError(f"static_seq_len ({static_seq_len}) must be >= sequence length L ({L})")
        target_len = ((static_seq_len + chunk_size - 1) // chunk_size) * chunk_size
    elif use_bucketing:
        target_len = get_static_bucket_length(L, chunk_size=chunk_size)
    else:
        target_len = ((L + chunk_size - 1) // chunk_size) * chunk_size

    pad_len = target_len - L
    n_chunks = target_len // chunk_size

    if pad_len > 0:
        q_p = F.pad(q, (0, 0, 0, pad_len), value=0.0)
        k_p = F.pad(k, (0, 0, 0, pad_len), value=0.0)
        v_p = F.pad(v, (0, 0, 0, pad_len), value=0.0)
        b_p = F.pad(b, (0, 0, 0, pad_len), value=0.0)
        w_p = F.pad(w, (0, 0, 0, pad_len), value=0.0)
        log_alpha_p = F.pad(log_alpha, (0, 0, 0, pad_len), value=0.0)
    else:
        q_p, k_p, v_p, b_p, w_p, log_alpha_p = q, k, v, b, w, log_alpha

    q_f = q_p.float()
    k_f = k_p.float()
    v_f = v_p.float()
    b_f = b_p.float()
    w_f = w_p.float()
    lac_f = log_alpha_p.float()

    s0_f = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=orig_device)
    )

    qc = q_f.reshape(B, H, n_chunks, chunk_size, dk)
    kc = k_f.reshape(B, H, n_chunks, chunk_size, dk)
    vc = v_f.reshape(B, H, n_chunks, chunk_size, dv)
    bc = b_f.reshape(B, H, n_chunks, chunk_size, dk)
    wc = w_f.reshape(B, H, n_chunks, chunk_size, dv)
    lac = lac_f.reshape(B, H, n_chunks, chunk_size, dk)

    cla = torch.cumsum(lac, dim=-2)
    diff = cla.unsqueeze(-2) - cla.unsqueeze(-3)
    dec = torch.exp(torch.clamp(diff, max=0.0))

    bk = (bc * kc).unsqueeze(-2)
    ks = kc.unsqueeze(-3)
    l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

    eye_c = torch.eye(chunk_size, dtype=torch.float32, device=orig_device).view(1, 1, 1, chunk_size, chunk_size)
    inv_l = eye_c - l_mat
    for _ in range(chunk_size - 2):
        inv_l = eye_c - torch.matmul(l_mat, inv_l)

    lam = torch.exp(cla)
    bh = (bc * kc) * lam
    wv = wc * vc

    clae = cla[..., -1:, :]
    dte = torch.exp(clae - cla)
    kd = kc * dte
    ds0 = torch.exp(clae).squeeze(-2).unsqueeze(-1)

    qs = qc.unsqueeze(-2)
    a_mat = torch.tril((qs * dec * ks).sum(dim=-1), diagonal=0)

    eye_2d = torch.eye(chunk_size, dtype=torch.float32, device=orig_device)
    cs = s0_f
    outs = []
    for c in range(n_chunks):
        ve = wv[:, :, c] - torch.matmul(bh[:, :, c], cs)
        u = torch.matmul(inv_l[:, :, c], ve)

        r = ve - u - torch.matmul(l_mat[:, :, c], u)
        if r.abs().max() > 1e-4 or torch.isnan(u).any() or torch.isinf(u).any():
            u = torch.linalg.solve_triangular(eye_2d + l_mat[:, :, c], ve, upper=False)

        oi = torch.matmul(qc[:, :, c] * lam[:, :, c], cs)
        ot = torch.matmul(a_mat[:, :, c], u)
        outs.append(oi + ot)
        cs = ds0[:, :, c] * cs + torch.matmul(kd[:, :, c].transpose(-1, -2), u)

    out_pad = torch.cat(outs, dim=2)
    out = out_pad[:, :, :L].to(orig_dtype)
    final_state = cs.to(orig_dtype)

    return out, final_state


@register_kernel("xla", "dgda_step")
def xla_dgda_step(
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
    if alpha is None and log_alpha is not None:
        alpha = torch.exp(log_alpha)
    elif log_alpha is None and alpha is not None:
        log_alpha = torch.log(alpha.clamp(min=1e-20, max=1.0))

    validate_dgda_step_inputs(q, k, v, alpha, b, w, state, log_alpha=log_alpha)

    orig_dtype = q.dtype
    orig_device = q.device

    if q.dim() == 4 and q.shape[2] == 1:
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        alpha = alpha.squeeze(2)
        b = b.squeeze(2)
        w = w.squeeze(2)

    B, H, dk = q.shape
    dv = v.shape[-1]

    q_f, k_f, v_f = q.float(), k.float(), v.float()
    alpha_f = alpha.float()
    b_f, w_f = b.float(), w.float()

    sp = (
        state.float().clone()
        if state is not None
        else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=orig_device)
    )

    sd = alpha_f.unsqueeze(-1) * sp
    beta = b_f * k_f
    pred = torch.matmul(beta.unsqueeze(-2), sd).squeeze(-2)
    delta = (w_f * v_f) - pred

    new_state = sd + torch.matmul(k_f.unsqueeze(-1), delta.unsqueeze(-2))
    out = torch.matmul(q_f.unsqueeze(-2), new_state).squeeze(-2)

    return out.to(orig_dtype), new_state.to(orig_dtype)


__all__ = [
    "DEFAULT_STATIC_BUCKETS",
    "get_static_bucket_length",
    "xla_dgda_prefill",
    "xla_dgda_step",
]
