
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from maba_sparse.kernels.common import (
    validate_dgda_prefill_inputs,
    validate_dgda_step_inputs,
)


def get_cpu_num_threads() -> int:
    return torch.get_num_threads()


def set_cpu_num_threads(num_threads: int) -> None:
    if num_threads <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    torch.set_num_threads(num_threads)


@contextmanager
def cpu_threads(num_threads: int):
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(num_threads)
        yield
    finally:
        torch.set_num_threads(prev_threads)


def cpu_dgda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    inversion_method: str = "adaptive",
    adaptive_tol: float = 7e-5,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_dgda_prefill_inputs(q, k, v, alpha, b, w, initial_state, log_alpha=log_alpha)

    orig_dt = q.dtype
    if orig_dt in (torch.float16, torch.bfloat16):
        q = q.float()
        k = k.float()
        v = v.float()
        if alpha is not None:
            alpha = alpha.float()
        b = b.float()
        w = w.float()
        if initial_state is not None:
            initial_state = initial_state.float()
        if log_alpha is not None:
            log_alpha = log_alpha.float()

    if alpha is None and log_alpha is not None:
        alpha = torch.exp(log_alpha)
    elif log_alpha is None and alpha is not None:
        log_alpha = torch.log(alpha.clamp(min=1e-20, max=1.0))
    elif log_alpha is not None:
        log_alpha = torch.clamp(log_alpha, min=-46.0, max=0.0)

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

    cs = (
        torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
        if initial_state is None
        else initial_state.clone()
    )

    n_chunks = L // chunk_size
    rem = L % chunk_size
    outs = []

    if n_chunks > 0:
        L_main = n_chunks * chunk_size
        qc = q[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)
        kc = k[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)
        vc = v[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dv)
        bc = b[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)
        wc = w[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dv)
        lac = log_alpha[:, :, :L_main].reshape(B, H, n_chunks, chunk_size, dk)

        cla = torch.cumsum(lac, dim=-2)
        diff = cla.unsqueeze(-2) - cla.unsqueeze(-3)
        dec = torch.exp(torch.clamp(diff, max=0.0))

        bk = (bc * kc).unsqueeze(-2)
        ks = kc.unsqueeze(-3)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

        eye_2d = torch.eye(chunk_size, dtype=q.dtype, device=q.device).view(1, 1, chunk_size, chunk_size)
        if inversion_method != "exact":
            eye_c = torch.eye(chunk_size, dtype=q.dtype, device=q.device).view(1, 1, 1, chunk_size, chunk_size)
            l2 = torch.matmul(l_mat, l_mat)
            l3 = torch.matmul(l2, l_mat)
            inv_l = eye_c - l_mat + l2 - l3

        lam = torch.exp(cla)
        bh = (bc * kc) * lam
        wv = wc * vc

        clae = cla[..., -1:, :]
        dte = torch.exp(clae - cla)
        kd = kc * dte
        ds0 = torch.exp(clae).squeeze(-2).unsqueeze(-1)

        qs = qc.unsqueeze(-2)
        a_mat = torch.tril((qs * dec * ks).sum(dim=-1), diagonal=0)

        for c in range(n_chunks):
            ve = wv[:, :, c] - torch.matmul(bh[:, :, c], cs)
            if inversion_method == "exact":
                u = torch.linalg.solve_triangular(eye_2d + l_mat[:, :, c], ve, upper=False)
            else:
                u = torch.matmul(inv_l[:, :, c], ve)
                if inversion_method == "adaptive":
                    r = ve - u - torch.matmul(l_mat[:, :, c], u)
                    if (
                        r.abs().max() > adaptive_tol
                        or torch.isnan(u).any()
                        or torch.isinf(u).any()
                    ):
                        u = torch.linalg.solve_triangular(eye_2d + l_mat[:, :, c], ve, upper=False)

            oi = torch.matmul(qc[:, :, c] * lam[:, :, c], cs)
            ot = torch.matmul(a_mat[:, :, c], u)
            outs.append(oi + ot)
            cs = ds0[:, :, c] * cs + torch.matmul(kd[:, :, c].transpose(-1, -2), u)

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
        lam = torch.exp(cla)
        bh = (br * kr) * lam
        ve = (wr * vr) - torch.matmul(bh, cs)

        if inversion_method == "exact":
            u = torch.linalg.solve_triangular(eye_t + l_mat, ve, upper=False)
        else:
            l2 = torch.matmul(l_mat, l_mat)
            l3 = torch.matmul(l2, l_mat)
            inv_l = eye_t - l_mat + l2 - l3
            u = torch.matmul(inv_l, ve)
            if inversion_method == "adaptive":
                r = ve - u - torch.matmul(l_mat, u)
                if (
                    r.abs().max() > adaptive_tol
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


def cpu_dgda_step(
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

    orig_dt = q.dtype
    if orig_dt in (torch.float16, torch.bfloat16):
        q = q.float()
        k = k.float()
        v = v.float()
        if alpha is not None:
            alpha = alpha.float()
        b = b.float()
        w = w.float()
        if state is not None:
            state = state.float()
        if log_alpha is not None:
            log_alpha = log_alpha.float()

    if alpha is None and log_alpha is not None:
        alpha = torch.exp(log_alpha)
    elif log_alpha is None and alpha is not None:
        log_alpha = torch.log(alpha.clamp(min=1e-20, max=1.0))
    elif log_alpha is not None:
        log_alpha = torch.clamp(log_alpha, min=-46.0, max=0.0)

    if q.dim() == 4 and q.shape[2] == 1:
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        alpha = alpha.squeeze(2)
        if log_alpha is not None and log_alpha.dim() == 4 and log_alpha.shape[2] == 1:
            log_alpha = log_alpha.squeeze(2)
        b = b.squeeze(2)
        w = w.squeeze(2)

    B, H, dk = q.shape
    dv = v.shape[-1]

    sp = (
        torch.zeros(B, H, dk, dv, dtype=q.dtype, device=q.device)
        if state is None
        else state.clone()
    )

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
