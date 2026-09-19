
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

from maba_sparse.kernels.common import (
    ref_dgda_prefill,
    ref_dgda_step,
    validate_dgda_prefill_inputs,
    validate_dgda_step_inputs,
)
from maba_sparse.kernels.dispatcher import (
    is_cuda_sm75_available,
    register_kernel,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_dgda_prefill_kernel(
        Q_ptr, K_ptr, V_ptr, Alpha_ptr, B_ptr, W_ptr, Out_ptr, State_ptr, InitState_ptr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ab, stride_ah, stride_al, stride_ad,
        stride_bb, stride_bh, stride_bl, stride_bd,
        stride_wb, stride_wh, stride_wl, stride_wd,
        stride_ob, stride_oh, stride_ol, stride_od,
        stride_sb, stride_sh, stride_sk, stride_sv,
        stride_isb, stride_ish, stride_isk, stride_isv,
        H: tl.constexpr,
        N_CHUNKS: tl.constexpr,
        has_init_state: tl.constexpr,
        is_log_alpha: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_v = tl.program_id(1)
        b_idx = pid_bh // H
        h_idx = pid_bh % H

        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

        if has_init_state:
            init_ptrs = (
                InitState_ptr + b_idx * stride_isb + h_idx * stride_ish
                + offs_k[:, None] * stride_isk + offs_v[None, :] * stride_isv
            )
            S = tl.load(init_ptrs).to(tl.float32)
        else:
            S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

        tril_strict = offs_m[:, None] > offs_m[None, :]
        tril_diag_f = tl.where(offs_m[:, None] >= offs_m[None, :], 1.0, 0.0)
        last_row_mask = offs_m[:, None] == (BLOCK_M - 1)

        base_q = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
        base_k = K_ptr + b_idx * stride_kb + h_idx * stride_kh
        base_v = V_ptr + b_idx * stride_vb + h_idx * stride_vh
        base_a = Alpha_ptr + b_idx * stride_ab + h_idx * stride_ah
        base_b = B_ptr + b_idx * stride_bb + h_idx * stride_bh
        base_w = W_ptr + b_idx * stride_wb + h_idx * stride_wh
        base_o = Out_ptr + b_idx * stride_ob + h_idx * stride_oh

        for c in range(0, N_CHUNKS):
            chunk_offset = c * BLOCK_M

            q_ptrs = base_q + (chunk_offset + offs_m[:, None]) * stride_ql + offs_k[None, :] * stride_qd
            k_ptrs = base_k + (chunk_offset + offs_m[:, None]) * stride_kl + offs_k[None, :] * stride_kd
            v_ptrs = base_v + (chunk_offset + offs_m[:, None]) * stride_vl + offs_v[None, :] * stride_vd
            a_ptrs = base_a + (chunk_offset + offs_m[:, None]) * stride_al + offs_k[None, :] * stride_ad
            b_ptrs = base_b + (chunk_offset + offs_m[:, None]) * stride_bl + offs_k[None, :] * stride_bd
            w_ptrs = base_w + (chunk_offset + offs_m[:, None]) * stride_wl + offs_v[None, :] * stride_wd

            q = tl.load(q_ptrs).to(tl.float32)
            k = tl.load(k_ptrs).to(tl.float32)
            v = tl.load(v_ptrs).to(tl.float32)
            alpha_in = tl.load(a_ptrs).to(tl.float32)
            b = tl.load(b_ptrs).to(tl.float32)
            w = tl.load(w_ptrs).to(tl.float32)

            log_alpha = alpha_in if is_log_alpha else tl.log(tl.maximum(alpha_in, 1e-20))
            cla = tl.dot(tril_diag_f, log_alpha)
            diff = cla[:, None, :] - cla[None, :, :]
            dec = tl.exp(tl.minimum(diff, 0.0))

            bk = b * k
            L_raw = tl.sum(bk[:, None, :] * dec * k[None, :, :], axis=2)
            L = tl.where(tril_strict, L_raw, 0.0)

            pred = tl.dot(bk * tl.exp(cla), S)
            w_v = w * v
            v_e = w_v - pred

            u = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            row0 = tl.sum(tl.where(offs_m[:, None] == 0, v_e, 0.0), axis=0)
            u = tl.where(offs_m[:, None] == 0, row0[None, :], u)

            for i in range(1, 16):
                l_row = tl.sum(tl.where(offs_m[:, None] == i, L, 0.0), axis=0)
                sum_lu = tl.sum(l_row[:, None] * u, axis=0)
                ve_row = tl.sum(tl.where(offs_m[:, None] == i, v_e, 0.0), axis=0)
                ui = ve_row - sum_lu
                u = tl.where(offs_m[:, None] == i, ui[None, :], u)

            A_raw = tl.sum(q[:, None, :] * dec * k[None, :, :], axis=2)
            A = tl.where(tril_diag_f > 0.5, A_raw, 0.0)

            o_hist = tl.dot(q * tl.exp(cla), S)
            o_intra = tl.dot(A, u)
            o_chunk = o_hist + o_intra

            out_ptrs = base_o + (chunk_offset + offs_m[:, None]) * stride_ol + offs_v[None, :] * stride_od
            tl.store(out_ptrs, o_chunk)

            cla_end = tl.sum(tl.where(last_row_mask, cla, 0.0), axis=0)
            decay_to_end = tl.exp(cla_end[None, :] - cla)
            k_tilde = k * decay_to_end

            decay_step = tl.exp(cla_end)[:, None]
            S_decay = S * decay_step
            S_update = tl.dot(tl.trans(k_tilde), u)
            S = S_decay + S_update

        state_ptrs = (
            State_ptr + b_idx * stride_sb + h_idx * stride_sh
            + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv
        )
        tl.store(state_ptrs, S)


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_dgda_step_kernel(
        Q_ptr, K_ptr, V_ptr, Alpha_ptr, B_ptr, W_ptr, StateIn_ptr, Out_ptr, StateOut_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_ab, stride_ah, stride_ad,
        stride_bb, stride_bh, stride_bd,
        stride_wb, stride_wh, stride_wd,
        stride_sib, stride_sih, stride_sik, stride_siv,
        stride_ob, stride_oh, stride_od,
        stride_sob, stride_soh, stride_sok, stride_sov,
        H: tl.constexpr,
        has_state: tl.constexpr,
        is_log_alpha: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b_idx = pid_bh // H
        h_idx = pid_bh % H

        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)

        q_ptrs = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_k * stride_qd
        k_ptrs = K_ptr + b_idx * stride_kb + h_idx * stride_kh + offs_k * stride_kd
        v_ptrs = V_ptr + b_idx * stride_vb + h_idx * stride_vh + offs_v * stride_vd
        a_ptrs = Alpha_ptr + b_idx * stride_ab + h_idx * stride_ah + offs_k * stride_ad
        b_ptrs = B_ptr + b_idx * stride_bb + h_idx * stride_bh + offs_k * stride_bd
        w_ptrs = W_ptr + b_idx * stride_wb + h_idx * stride_wh + offs_v * stride_wd

        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        alpha_in = tl.load(a_ptrs).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)

        alpha = tl.exp(alpha_in) if is_log_alpha else alpha_in

        if has_state:
            si_ptrs = (
                StateIn_ptr + b_idx * stride_sib + h_idx * stride_sih
                + offs_k[:, None] * stride_sik + offs_v[None, :] * stride_siv
            )
            S_prev = tl.load(si_ptrs).to(tl.float32)
        else:
            S_prev = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

        S_decay = alpha[:, None] * S_prev
        beta = b * k
        pred = tl.sum(beta[:, None] * S_decay, axis=0)

        u = w * v
        delta = u - pred

        S_new = S_decay + k[:, None] * delta[None, :]
        out = tl.sum(q[:, None] * S_new, axis=0)

        out_ptrs = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, out)

        so_ptrs = (
            StateOut_ptr + b_idx * stride_sob + h_idx * stride_soh
            + offs_k[:, None] * stride_sok + offs_v[None, :] * stride_sov
        )
        tl.store(so_ptrs, S_new)


def _dgda_prefill_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor],
    b: torch.Tensor,
    w: torch.Tensor,
    init_state: Optional[torch.Tensor],
    grad_out: torch.Tensor,
    grad_final_state: Optional[torch.Tensor],
    chunk_size: int = 16,
    log_alpha: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    Optional[torch.Tensor], torch.Tensor, torch.Tensor,
    Optional[torch.Tensor], Optional[torch.Tensor]
]:
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    orig_device = q.device

    curr_s = (
        init_state.float().clone()
        if init_state is not None
        else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=orig_device)
    )

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    b_f = b.float()
    w_f = w.float()

    if alpha is not None:
        alpha_f = alpha.float()
    else:
        alpha_f = torch.exp(log_alpha.float())

    chunk_states = [curr_s]
    for t in range(L):
        kt = k_f[:, :, t]
        vt = v_f[:, :, t]
        at = alpha_f[:, :, t]
        bt = b_f[:, :, t]
        wt = w_f[:, :, t]

        s_dec = at.unsqueeze(-1) * curr_s
        beta = bt * kt
        pred = torch.matmul(beta.unsqueeze(-2), s_dec).squeeze(-2)
        delta = (wt * vt) - pred
        curr_s = s_dec + torch.matmul(kt.unsqueeze(-1), delta.unsqueeze(-2))

        if (t + 1) % chunk_size == 0 and (t + 1) < L:
            chunk_states.append(curr_s)

    grad_q = torch.zeros_like(q_f)
    grad_k = torch.zeros_like(k_f)
    grad_v = torch.zeros_like(v_f)
    grad_alpha = torch.zeros_like(alpha_f)
    grad_b = torch.zeros_like(b_f)
    grad_w = torch.zeros_like(w_f)

    s_adj = (
        grad_final_state.float().clone()
        if grad_final_state is not None
        else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=orig_device)
    )

    grad_out_f = grad_out.float()
    n_chunks = (L + chunk_size - 1) // chunk_size

    for c in range(n_chunks - 1, -1, -1):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, L)
        c_len = c_end - c_start

        c_curr_s = chunk_states[c]
        c_states = []
        c_s_decays = []
        c_deltas = []

        for i in range(c_len):
            t = c_start + i
            kt = k_f[:, :, t]
            vt = v_f[:, :, t]
            at = alpha_f[:, :, t]
            bt = b_f[:, :, t]
            wt = w_f[:, :, t]

            s_dec = at.unsqueeze(-1) * c_curr_s
            c_s_decays.append(s_dec)
            beta = bt * kt
            pred = torch.matmul(beta.unsqueeze(-2), s_dec).squeeze(-2)
            delta = (wt * vt) - pred
            c_deltas.append(delta)
            c_curr_s = s_dec + torch.matmul(kt.unsqueeze(-1), delta.unsqueeze(-2))
            c_states.append(c_curr_s)

        for i in range(c_len - 1, -1, -1):
            t = c_start + i
            s_t = c_states[i]
            s_dec = c_s_decays[i]
            delta = c_deltas[i]
            s_prev = c_states[i - 1] if i > 0 else chunk_states[c]

            go_t = grad_out_f[:, :, t]
            grad_q[:, :, t] = torch.matmul(s_t, go_t.unsqueeze(-1)).squeeze(-1)
            s_adj = s_adj + torch.matmul(q_f[:, :, t].unsqueeze(-1), go_t.unsqueeze(-2))

            grad_k[:, :, t] += torch.matmul(s_adj, delta.unsqueeze(-1)).squeeze(-1)
            grad_delta = torch.matmul(s_adj.transpose(-1, -2), k_f[:, :, t].unsqueeze(-1)).squeeze(-1)

            grad_w[:, :, t] = grad_delta * v_f[:, :, t]
            grad_v[:, :, t] = grad_delta * w_f[:, :, t]

            beta = b_f[:, :, t] * k_f[:, :, t]
            grad_beta = -torch.matmul(s_dec, grad_delta.unsqueeze(-1)).squeeze(-1)
            grad_b[:, :, t] = grad_beta * k_f[:, :, t]
            grad_k[:, :, t] += grad_beta * b_f[:, :, t]

            s_dec_adj = s_adj - torch.matmul(beta.unsqueeze(-1), grad_delta.unsqueeze(-2))
            grad_alpha[:, :, t] = (s_dec_adj * s_prev).sum(dim=-1)
            s_adj = alpha_f[:, :, t].unsqueeze(-1) * s_dec_adj

    grad_init_state = s_adj.to(init_state.dtype) if init_state is not None else None

    grad_alpha_out = grad_alpha.to(alpha.dtype) if alpha is not None else None
    grad_log_alpha_out = (
        (grad_alpha * alpha_f).to(log_alpha.dtype) if log_alpha is not None else None
    )

    return (
        grad_q.to(q.dtype),
        grad_k.to(k.dtype),
        grad_v.to(v.dtype),
        grad_alpha_out,
        grad_b.to(b.dtype),
        grad_w.to(w.dtype),
        grad_init_state,
        grad_log_alpha_out,
    )


class _TritonDGDAPrefillFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        chunk_size: int = 16,
        initial_state: Optional[torch.Tensor] = None,
        log_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(q, k, v, alpha, b, w, initial_state, log_alpha)
        ctx.chunk_size = chunk_size

        B, H, L, dk = q.shape
        dv = v.shape[-1]
        orig_dtype = q.dtype
        device = q.device

        if L == 0:
            empty_out = torch.empty(B, H, 0, dv, dtype=orig_dtype, device=device)
            state = (
                initial_state.clone()
                if initial_state is not None
                else torch.zeros(B, H, dk, dv, dtype=orig_dtype, device=device)
            )
            return empty_out, state

        if (
            device.type == "cuda"
            and TRITON_AVAILABLE
            and is_cuda_sm75_available(device)
            and dk in (32, 64, 128)
            and dv in (32, 64, 128)
            and chunk_size == 16
        ):
            block_v = 32 if dv == 32 else 64
            pad_len = (16 - (L % 16)) % 16
            if pad_len > 0:
                q_p = F.pad(q, (0, 0, 0, pad_len), value=0.0)
                k_p = F.pad(k, (0, 0, 0, pad_len), value=0.0)
                v_p = F.pad(v, (0, 0, 0, pad_len), value=0.0)
                b_p = F.pad(b, (0, 0, 0, pad_len), value=0.0)
                w_p = F.pad(w, (0, 0, 0, pad_len), value=0.0)
                if log_alpha is not None:
                    alpha_p = F.pad(log_alpha, (0, 0, 0, pad_len), value=0.0)
                    is_log_a = True
                else:
                    alpha_p = F.pad(alpha, (0, 0, 0, pad_len), value=1.0)
                    is_log_a = False
            else:
                q_p, k_p, v_p, b_p, w_p = q, k, v, b, w
                if log_alpha is not None:
                    alpha_p = log_alpha
                    is_log_a = True
                else:
                    alpha_p = alpha
                    is_log_a = False

            L_pad = q_p.shape[2]
            n_chunks = L_pad // 16
            out_pad = torch.zeros(B, H, L_pad, dv, dtype=orig_dtype, device=device)
            final_state = torch.zeros(B, H, dk, dv, dtype=orig_dtype, device=device)

            has_init = initial_state is not None
            init_s = (
                initial_state
                if has_init
                else torch.zeros(1, 1, 1, 1, dtype=orig_dtype, device=device)
            )

            num_v_blocks = (dv + block_v - 1) // block_v
            grid = (B * H, num_v_blocks)
            _fused_dgda_prefill_kernel[grid](
                q_p, k_p, v_p, alpha_p, b_p, w_p, out_pad, final_state, init_s,
                q_p.stride(0), q_p.stride(1), q_p.stride(2), q_p.stride(3),
                k_p.stride(0), k_p.stride(1), k_p.stride(2), k_p.stride(3),
                v_p.stride(0), v_p.stride(1), v_p.stride(2), v_p.stride(3),
                alpha_p.stride(0), alpha_p.stride(1), alpha_p.stride(2), alpha_p.stride(3),
                b_p.stride(0), b_p.stride(1), b_p.stride(2), b_p.stride(3),
                w_p.stride(0), w_p.stride(1), w_p.stride(2), w_p.stride(3),
                out_pad.stride(0), out_pad.stride(1), out_pad.stride(2), out_pad.stride(3),
                final_state.stride(0), final_state.stride(1), final_state.stride(2), final_state.stride(3),
                init_s.stride(0), init_s.stride(1), init_s.stride(2), init_s.stride(3) if has_init else 0,
                H=H, N_CHUNKS=n_chunks, has_init_state=has_init,
                is_log_alpha=is_log_a,
                BLOCK_M=16, BLOCK_K=dk, BLOCK_V=block_v,
                num_stages=1, num_warps=4
            )
            return out_pad[:, :, :L], final_state

        return ref_dgda_prefill(
            q=q, k=k, v=v, alpha=alpha, b=b, w=w,
            chunk_size=chunk_size, initial_state=initial_state,
            log_alpha=log_alpha
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
        grad_final_state: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], torch.Tensor, torch.Tensor,
        None, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        q, k, v, alpha, b, w, initial_state, log_alpha = ctx.saved_tensors
        g_q, g_k, g_v, g_a, g_b, g_w, g_s0, g_la = _dgda_prefill_backward(
            q=q, k=k, v=v, alpha=alpha, b=b, w=w,
            init_state=initial_state,
            grad_out=grad_out,
            grad_final_state=grad_final_state,
            chunk_size=ctx.chunk_size,
            log_alpha=log_alpha,
        )
        if alpha is not None:
            return g_q, g_k, g_v, g_a, g_b, g_w, None, g_s0, None
        else:
            return g_q, g_k, g_v, None, g_b, g_w, None, g_s0, g_la


class _TritonDGDAStepFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        log_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(q, k, v, alpha, b, w, state, log_alpha)

        orig_dtype = q.dtype
        device = q.device

        q_3d = q.squeeze(2) if q.dim() == 4 and q.shape[2] == 1 else q
        k_3d = k.squeeze(2) if k.dim() == 4 and k.shape[2] == 1 else k
        v_3d = v.squeeze(2) if v.dim() == 4 and v.shape[2] == 1 else v
        b_3d = b.squeeze(2) if b.dim() == 4 and b.shape[2] == 1 else b
        w_3d = w.squeeze(2) if w.dim() == 4 and w.shape[2] == 1 else w

        if log_alpha is not None:
            a_3d = log_alpha.squeeze(2) if log_alpha.dim() == 4 and log_alpha.shape[2] == 1 else log_alpha
            is_log_a = True
        else:
            a_3d = alpha.squeeze(2) if alpha.dim() == 4 and alpha.shape[2] == 1 else alpha
            is_log_a = False

        B, H, dk = q_3d.shape
        dv = v_3d.shape[-1]

        if (
            device.type == "cuda"
            and TRITON_AVAILABLE
            and is_cuda_sm75_available(device)
            and dk in (32, 64, 128)
            and dv in (32, 64, 128)
        ):
            out = torch.zeros(B, H, dv, dtype=orig_dtype, device=device)
            new_state = torch.zeros(B, H, dk, dv, dtype=orig_dtype, device=device)

            has_s = state is not None
            st_in = (
                state
                if has_s
                else torch.zeros(1, 1, 1, 1, dtype=orig_dtype, device=device)
            )

            grid = (B * H,)
            _fused_dgda_step_kernel[grid](
                q_3d, k_3d, v_3d, a_3d, b_3d, w_3d, st_in, out, new_state,
                q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
                k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
                v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
                a_3d.stride(0), a_3d.stride(1), a_3d.stride(2),
                b_3d.stride(0), b_3d.stride(1), b_3d.stride(2),
                w_3d.stride(0), w_3d.stride(1), w_3d.stride(2),
                st_in.stride(0), st_in.stride(1), st_in.stride(2), st_in.stride(3) if has_s else 0,
                out.stride(0), out.stride(1), out.stride(2),
                new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
                H=H, has_state=has_s, is_log_alpha=is_log_a,
                BLOCK_K=dk, BLOCK_V=dv,
                num_stages=1, num_warps=4
            )
            return out, new_state

        return ref_dgda_step(
            q=q, k=k, v=v, alpha=alpha, b=b, w=w, state=state, log_alpha=log_alpha
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
        grad_state: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        q, k, v, alpha, b, w, state, log_alpha = ctx.saved_tensors

        q_3d = q.squeeze(2) if q.dim() == 4 and q.shape[2] == 1 else q
        k_3d = k.squeeze(2) if k.dim() == 4 and k.shape[2] == 1 else k
        v_3d = v.squeeze(2) if v.dim() == 4 and v.shape[2] == 1 else v
        b_3d = b.squeeze(2) if b.dim() == 4 and b.shape[2] == 1 else b
        w_3d = w.squeeze(2) if w.dim() == 4 and w.shape[2] == 1 else w

        if alpha is not None:
            a_3d = alpha.squeeze(2) if alpha.dim() == 4 and alpha.shape[2] == 1 else alpha
        else:
            a_3d = torch.exp(log_alpha.squeeze(2) if log_alpha.dim() == 4 and log_alpha.shape[2] == 1 else log_alpha)

        B, H, dk = q_3d.shape
        dv = v_3d.shape[-1]
        device = q.device

        sp = (
            state.float().clone()
            if state is not None
            else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=device)
        )

        sd = a_3d.float().unsqueeze(-1) * sp
        beta = b_3d.float() * k_3d.float()
        pred = torch.matmul(beta.unsqueeze(-2), sd).squeeze(-2)
        delta = (w_3d.float() * v_3d.float()) - pred
        new_state = sd + torch.matmul(k_3d.float().unsqueeze(-1), delta.unsqueeze(-2))

        go = grad_out.float()
        gs = (
            grad_state.float().clone()
            if grad_state is not None
            else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=device)
        )

        grad_q = torch.matmul(new_state, go.unsqueeze(-1)).squeeze(-1)
        s_adj = gs + torch.matmul(q_3d.float().unsqueeze(-1), go.unsqueeze(-2))

        grad_k = torch.matmul(s_adj, delta.unsqueeze(-1)).squeeze(-1)
        grad_delta = torch.matmul(s_adj.transpose(-1, -2), k_3d.float().unsqueeze(-1)).squeeze(-1)

        grad_w = grad_delta * v_3d.float()
        grad_v = grad_delta * w_3d.float()

        grad_beta = -torch.matmul(sd, grad_delta.unsqueeze(-1)).squeeze(-1)
        grad_b = grad_beta * k_3d.float()
        grad_k += grad_beta * b_3d.float()

        s_dec_adj = s_adj - torch.matmul(beta.unsqueeze(-1), grad_delta.unsqueeze(-2))
        grad_alpha = (s_dec_adj * sp).sum(dim=-1)
        grad_s_in = (a_3d.float().unsqueeze(-1) * s_dec_adj).to(state.dtype) if state is not None else None

        if q.dim() == 4 and q.shape[2] == 1:
            grad_q = grad_q.unsqueeze(2)
            grad_k = grad_k.unsqueeze(2)
            grad_v = grad_v.unsqueeze(2)
            grad_alpha = grad_alpha.unsqueeze(2)
            grad_b = grad_b.unsqueeze(2)
            grad_w = grad_w.unsqueeze(2)

        grad_alpha_out = grad_alpha.to(alpha.dtype) if alpha is not None else None
        grad_log_alpha_out = (
            (grad_alpha * a_3d.float().unsqueeze(2) if q.dim() == 4 and q.shape[2] == 1 else grad_alpha * a_3d.float()).to(log_alpha.dtype)
            if log_alpha is not None
            else None
        )

        return (
            grad_q.to(q.dtype),
            grad_k.to(k.dtype),
            grad_v.to(v.dtype),
            grad_alpha_out,
            grad_b.to(b.dtype),
            grad_w.to(w.dtype),
            grad_s_in,
            grad_log_alpha_out,
        )


@register_kernel("triton", "dgda_prefill")
def triton_dgda_prefill(
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

    return _TritonDGDAPrefillFunction.apply(
        q, k, v, alpha, b, w, chunk_size, initial_state, log_alpha
    )


@register_kernel("triton", "dgda_step")
def triton_dgda_step(
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

    return _TritonDGDAStepFunction.apply(
        q, k, v, alpha, b, w, state, log_alpha
    )


__all__ = [
    "triton_dgda_prefill",
    "triton_dgda_step",
]
