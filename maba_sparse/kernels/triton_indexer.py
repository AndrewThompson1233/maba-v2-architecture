
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from maba_sparse.kernels.common import (
    validate_indexer_inputs,
    validate_superposition_inputs,
)
from maba_sparse.kernels.cpu_indexer import (
    cpu_compute_centroids,
    cpu_index_topk,
    cpu_stream_superposition,
)
from maba_sparse.kernels.dispatcher import (
    is_cuda_sm75_available,
    is_triton_available,
    register_kernel,
)

TRITON_AVAILABLE = is_triton_available()
if TRITON_AVAILABLE:
    try:
        import triton
        import triton.language as tl
    except Exception:
        TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_centroid_kernel(
        K_ptr,
        Out_ptr,
        Argmax_ptr,
        stride_kb,
        stride_kl,
        stride_kd,
        stride_ob,
        stride_on,
        stride_od,
        stride_ab,
        stride_an,
        stride_ad,
        L: int,
        d_idx: int,
        nb: int,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_nb = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        token_idx = pid_nb * BLOCK_M + offs_m
        mask = (token_idx[:, None] < L) & (offs_d[None, :] < d_idx)

        k_ptrs = (
            K_ptr
            + pid_b * stride_kb
            + token_idx[:, None] * stride_kl
            + offs_d[None, :] * stride_kd
        )
        vals = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)

        sum_vals = tl.sum(vals, axis=0)
        mean_vals = sum_vals / float(BLOCK_M)
        max_vals = tl.max(vals, axis=0)
        centroid = 0.5 * (mean_vals + max_vals)

        out_ptrs = (
            Out_ptr
            + pid_b * stride_ob
            + pid_nb * stride_on
            + offs_d * stride_od
        )
        tl.store(out_ptrs, centroid, mask=(offs_d < d_idx))

        if Argmax_ptr is not None:
            argmax_idx = tl.argmax(vals, axis=0)
            argmax_ptrs = (
                Argmax_ptr
                + pid_b * stride_ab
                + pid_nb * stride_an
                + offs_d * stride_ad
            )
            tl.store(argmax_ptrs, argmax_idx, mask=(offs_d < d_idx))

    @triton.jit
    def _fused_centroid_backward_kernel(
        GradOut_ptr,
        Argmax_ptr,
        GradK_ptr,
        stride_gb,
        stride_gn,
        stride_gd,
        stride_ab,
        stride_an,
        stride_ad,
        stride_kb,
        stride_kl,
        stride_kd,
        L: int,
        d_idx: int,
        nb: int,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_nb = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        token_idx = pid_nb * BLOCK_M + offs_m
        mask = (token_idx[:, None] < L) & (offs_d[None, :] < d_idx)

        go_ptrs = (
            GradOut_ptr
            + pid_b * stride_gb
            + pid_nb * stride_gn
            + offs_d * stride_gd
        )
        grad_c = tl.load(go_ptrs, mask=(offs_d < d_idx), other=0.0).to(tl.float32)

        argmax_ptrs = (
            Argmax_ptr
            + pid_b * stride_ab
            + pid_nb * stride_an
            + offs_d * stride_ad
        )
        argmax_idx = tl.load(argmax_ptrs, mask=(offs_d < d_idx), other=0)

        is_argmax = offs_m[:, None] == argmax_idx[None, :]
        weight = 0.5 * (1.0 / float(BLOCK_M) + tl.where(is_argmax, 1.0, 0.0))
        grad_k = grad_c[None, :] * weight

        gk_ptrs = (
            GradK_ptr
            + pid_b * stride_kb
            + token_idx[:, None] * stride_kl
            + offs_d[None, :] * stride_kd
        )
        tl.store(gk_ptrs, grad_k, mask=mask)

    @triton.jit
    def _fused_score_penalty_kernel(
        Q_ptr,
        C_ptr,
        Scores_ptr,
        stride_qb,
        stride_ql,
        stride_qd,
        stride_cb,
        stride_cn,
        stride_cd,
        stride_sb,
        stride_sl,
        stride_sn,
        L: int,
        nb: int,
        d_idx: int,
        scale: float,
        lambda_dist: float,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_l = tl.program_id(0)
        pid_b = tl.program_id(1)

        t = pid_l
        q_block = t // BLOCK_SIZE

        offs_d = tl.arange(0, BLOCK_D)
        q_mask = offs_d < d_idx

        q_ptrs = Q_ptr + pid_b * stride_qb + t * stride_ql + offs_d * stride_qd
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32) * scale

        for i in range(0, nb):
            c_ptrs = C_ptr + pid_b * stride_cb + i * stride_cn + offs_d * stride_cd
            c = tl.load(c_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            dot = tl.sum(q * c)
            dist = tl.maximum(0.0, (q_block - i).to(tl.float32))
            penalty = lambda_dist * tl.log(1.0 + dist)
            score = tl.where(i > q_block, -float("inf"), dot - penalty)

            out_ptr = Scores_ptr + pid_b * stride_sb + t * stride_sl + i * stride_sn
            tl.store(out_ptr, score)

    @triton.jit
    def _fused_stream_superposition_kernel(
        OL_ptr,
        OS_ptr,
        OH_ptr,
        Gate_ptr,
        Out_ptr,
        stride_lb,
        stride_lh,
        stride_ll,
        stride_ld,
        stride_sb,
        stride_sh,
        stride_sl,
        stride_sd,
        stride_hb,
        stride_hh,
        stride_hl,
        stride_hd,
        stride_gb,
        stride_gh,
        stride_gl,
        stride_gm,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        H: int,
        L: int,
        D: int,
        IS_WEIGHTS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t_idx = pid % L
        h_idx = (pid // L) % H
        b_idx = pid // (L * H)

        g_base = Gate_ptr + b_idx * stride_gb + h_idx * stride_gh + t_idx * stride_gl
        z0 = tl.load(g_base + 0 * stride_gm).to(tl.float32)
        z1 = tl.load(g_base + 1 * stride_gm).to(tl.float32)
        z2 = tl.load(g_base + 2 * stride_gm).to(tl.float32)

        if IS_WEIGHTS:
            g0 = z0
            g1 = z1
            g2 = z2
        else:
            m = tl.maximum(z0, tl.maximum(z1, z2))
            e0 = tl.exp(z0 - m)
            e1 = tl.exp(z1 - m)
            e2 = tl.exp(z2 - m)
            inv_sum = 1.0 / (e0 + e1 + e2)
            g0 = e0 * inv_sum
            g1 = e1 * inv_sum
            g2 = e2 * inv_sum

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            ol_ptrs = (
                OL_ptr
                + b_idx * stride_lb
                + h_idx * stride_lh
                + t_idx * stride_ll
                + offs_d * stride_ld
            )
            os_ptrs = (
                OS_ptr
                + b_idx * stride_sb
                + h_idx * stride_sh
                + t_idx * stride_sl
                + offs_d * stride_sd
            )
            oh_ptrs = (
                OH_ptr
                + b_idx * stride_hb
                + h_idx * stride_hh
                + t_idx * stride_hl
                + offs_d * stride_hd
            )

            ol = tl.load(ol_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            os = tl.load(os_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            oh = tl.load(oh_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            fused = g0 * ol + g1 * os + g2 * oh

            out_ptrs = (
                Out_ptr
                + b_idx * stride_ob
                + h_idx * stride_oh
                + t_idx * stride_ol
                + offs_d * stride_od
            )
            tl.store(out_ptrs, fused, mask=mask_d)


class _TritonCentroidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k_idx: torch.Tensor, block_size: int = 64) -> torch.Tensor:
        B, L, d_idx = k_idx.shape
        nb = (L + block_size - 1) // block_size

        if L == 0:
            return torch.empty(B, 0, d_idx, dtype=k_idx.dtype, device=k_idx.device)

        centroids = torch.empty((B, nb, d_idx), dtype=k_idx.dtype, device=k_idx.device)
        argmax_idx = torch.empty((B, nb, d_idx), dtype=torch.int32, device=k_idx.device)

        BLOCK_M = block_size
        BLOCK_D = 128 if d_idx > 64 else (64 if d_idx > 32 else 32)

        grid = (nb, B)
        _fused_centroid_kernel[grid](
            k_idx,
            centroids,
            argmax_idx,
            k_idx.stride(0),
            k_idx.stride(1),
            k_idx.stride(2),
            centroids.stride(0),
            centroids.stride(1),
            centroids.stride(2),
            argmax_idx.stride(0),
            argmax_idx.stride(1),
            argmax_idx.stride(2),
            L,
            d_idx,
            nb,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(argmax_idx)
        ctx.L = L
        ctx.d_idx = d_idx
        ctx.nb = nb
        ctx.block_size = block_size
        ctx.k_dtype = k_idx.dtype
        ctx.k_device = k_idx.device
        ctx.B = B

        return centroids

    @staticmethod
    def backward(ctx, grad_centroids: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        (argmax_idx,) = ctx.saved_tensors
        L = ctx.L
        d_idx = ctx.d_idx
        nb = ctx.nb
        block_size = ctx.block_size
        B = ctx.B

        grad_k = torch.zeros((B, L, d_idx), dtype=ctx.k_dtype, device=ctx.k_device)

        BLOCK_M = block_size
        BLOCK_D = 128 if d_idx > 64 else (64 if d_idx > 32 else 32)

        grid = (nb, B)
        _fused_centroid_backward_kernel[grid](
            grad_centroids,
            argmax_idx,
            grad_k,
            grad_centroids.stride(0),
            grad_centroids.stride(1),
            grad_centroids.stride(2),
            argmax_idx.stride(0),
            argmax_idx.stride(1),
            argmax_idx.stride(2),
            grad_k.stride(0),
            grad_k.stride(1),
            grad_k.stride(2),
            L,
            d_idx,
            nb,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
        )

        return grad_k, None


class _TritonScoreFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q_idx: torch.Tensor,
        centroids: torch.Tensor,
        lambda_dist: float = 0.5,
        block_size: int = 64,
    ) -> torch.Tensor:
        B, L, d_idx = q_idx.shape
        nb = centroids.shape[1]

        scores = torch.empty((B, L, nb), dtype=q_idx.dtype, device=q_idx.device)
        scale = 1.0 / math.sqrt(d_idx)
        BLOCK_D = 128 if d_idx > 64 else (64 if d_idx > 32 else 32)

        grid = (L, B)
        _fused_score_penalty_kernel[grid](
            q_idx,
            centroids,
            scores,
            q_idx.stride(0),
            q_idx.stride(1),
            q_idx.stride(2),
            centroids.stride(0),
            centroids.stride(1),
            centroids.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            L,
            nb,
            d_idx,
            scale,
            lambda_dist,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q_idx, centroids)
        ctx.scale = scale
        ctx.q_dtype = q_idx.dtype
        ctx.c_dtype = centroids.dtype
        return scores

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q_idx, centroids = ctx.saved_tensors
        scale = ctx.scale

        finite_mask = torch.isfinite(grad_scores)
        grad_s = torch.where(finite_mask, grad_scores, torch.zeros_like(grad_scores)).float()

        grad_q = None
        if ctx.needs_input_grad[0]:
            grad_q = (scale * torch.bmm(grad_s, centroids.float())).to(ctx.q_dtype)

        grad_c = None
        if ctx.needs_input_grad[1]:
            grad_c = (scale * torch.bmm(grad_s.transpose(1, 2), q_idx.float())).to(ctx.c_dtype)

        return grad_q, grad_c, None, None


class _TritonSuperpositionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        o_local: torch.Tensor,
        o_sparse: torch.Tensor,
        o_hca: torch.Tensor,
        gate_logits: Optional[torch.Tensor] = None,
        gate_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, L, D = o_local.shape
        out = torch.empty_like(o_local)

        gate_t = gate_logits if gate_logits is not None else gate_weights
        is_weights = int(gate_weights is not None)

        stride_gb = gate_t.stride(0)
        stride_gh = gate_t.stride(1) if gate_t.dim() == 4 else 0
        stride_gl = gate_t.stride(1) if gate_t.dim() == 3 else gate_t.stride(2)
        stride_gm = gate_t.stride(-1)

        BLOCK_D = 128 if D > 64 else 64
        grid = (B * H * L,)

        _fused_stream_superposition_kernel[grid](
            o_local,
            o_sparse,
            o_hca,
            gate_t,
            out,
            o_local.stride(0),
            o_local.stride(1),
            o_local.stride(2),
            o_local.stride(3),
            o_sparse.stride(0),
            o_sparse.stride(1),
            o_sparse.stride(2),
            o_sparse.stride(3),
            o_hca.stride(0),
            o_hca.stride(1),
            o_hca.stride(2),
            o_hca.stride(3),
            stride_gb,
            stride_gh,
            stride_gl,
            stride_gm,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H,
            L,
            D,
            IS_WEIGHTS=is_weights,
            BLOCK_D=BLOCK_D,
        )

        if is_weights:
            g = gate_weights
        else:
            g = F.softmax(gate_logits.float(), dim=-1).to(o_local.dtype)

        ctx.save_for_backward(o_local, o_sparse, o_hca, g)
        ctx.is_weights = is_weights
        ctx.gate_dim = gate_t.dim()
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        o_local, o_sparse, o_hca, g = ctx.saved_tensors
        is_weights = ctx.is_weights
        gate_dim = ctx.gate_dim

        if gate_dim == 3:
            gl = g[:, :, 0:1].unsqueeze(1)
            gs = g[:, :, 1:2].unsqueeze(1)
            gh = g[:, :, 2:3].unsqueeze(1)
        else:
            gl = g[:, :, :, 0:1]
            gs = g[:, :, :, 1:2]
            gh = g[:, :, :, 2:3]

        grad_ol = (gl * grad_out) if ctx.needs_input_grad[0] else None
        grad_os = (gs * grad_out) if ctx.needs_input_grad[1] else None
        grad_oh = (gh * grad_out) if ctx.needs_input_grad[2] else None

        grad_logits = None
        grad_weights = None

        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            if gate_dim == 3:
                delta_0 = (grad_out * o_local).sum(dim=(1, 3))
                delta_1 = (grad_out * o_sparse).sum(dim=(1, 3))
                delta_2 = (grad_out * o_hca).sum(dim=(1, 3))
                deltas = torch.stack([delta_0, delta_1, delta_2], dim=-1)
                delta_bar = (g * deltas).sum(dim=-1, keepdim=True)
            else:
                delta_0 = (grad_out * o_local).sum(dim=3)
                delta_1 = (grad_out * o_sparse).sum(dim=3)
                delta_2 = (grad_out * o_hca).sum(dim=3)
                deltas = torch.stack([delta_0, delta_1, delta_2], dim=-1)
                delta_bar = (g * deltas).sum(dim=-1, keepdim=True)

            if is_weights:
                grad_weights = deltas.to(g.dtype)
            else:
                grad_logits = (g * (deltas - delta_bar)).to(g.dtype)

        return grad_ol, grad_os, grad_oh, grad_logits, grad_weights


@register_kernel("triton", "compute_centroids")
def triton_compute_centroids(
    k_idx: torch.Tensor,
    block_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    if not (TRITON_AVAILABLE and is_cuda_sm75_available(k_idx.device)):
        return cpu_compute_centroids(k_idx, block_size=block_size, **kwargs)

    validate_indexer_inputs(k_idx=k_idx)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, L, d_idx = k_idx.shape
    if L == 0:
        return torch.empty(B, 0, d_idx, dtype=k_idx.dtype, device=k_idx.device)

    if block_size != 64:
        return cpu_compute_centroids(k_idx, block_size=block_size, **kwargs)

    return _TritonCentroidFunction.apply(k_idx, block_size)


@register_kernel("triton", "index_topk")
def triton_index_topk(
    q_idx: torch.Tensor,
    centroids: torch.Tensor,
    lambda_dist: float = 0.5,
    top_k: int = 32,
    block_size: int = 64,
    return_scores: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not (TRITON_AVAILABLE and is_cuda_sm75_available(q_idx.device)):
        return cpu_index_topk(
            q_idx=q_idx,
            centroids=centroids,
            lambda_dist=lambda_dist,
            top_k=top_k,
            block_size=block_size,
            return_scores=return_scores,
            **kwargs,
        )

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

    if L == 0 or nb == 0:
        ak = min(top_k, nb)
        empty_idx = torch.zeros(B, L, ak, dtype=torch.long, device=q_idx.device)
        if return_scores:
            empty_scores = torch.zeros(B, L, nb, dtype=q_idx.dtype, device=q_idx.device)
            return empty_idx, empty_scores
        return empty_idx

    if return_scores and (q_idx.requires_grad or centroids.requires_grad):
        scores = _TritonScoreFunction.apply(q_idx, centroids, lam, block_size)
    else:
        scores = torch.empty((B, L, nb), dtype=q_idx.dtype, device=q_idx.device)
        scale = 1.0 / math.sqrt(d_idx)
        BLOCK_D = 128 if d_idx > 64 else (64 if d_idx > 32 else 32)
        grid = (L, B)
        _fused_score_penalty_kernel[grid](
            q_idx,
            centroids,
            scores,
            q_idx.stride(0),
            q_idx.stride(1),
            q_idx.stride(2),
            centroids.stride(0),
            centroids.stride(1),
            centroids.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            L,
            nb,
            d_idx,
            scale,
            lam,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
        )

    ak = min(top_k, nb)
    top_indices = torch.topk(scores, k=ak, dim=-1, largest=True, sorted=True).indices

    if return_scores:
        return top_indices.to(torch.long), scores
    return top_indices.to(torch.long)


@register_kernel("triton", "stream_superposition")
def triton_stream_superposition(
    o_local: torch.Tensor,
    o_sparse: torch.Tensor,
    o_hca: torch.Tensor,
    gate_logits: Optional[torch.Tensor] = None,
    gate_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    if not (TRITON_AVAILABLE and is_cuda_sm75_available(o_local.device)):
        return cpu_stream_superposition(
            o_local=o_local,
            o_sparse=o_sparse,
            o_hca=o_hca,
            gate_logits=gate_logits,
            gate_weights=gate_weights,
            **kwargs,
        )

    gate_t = gate_logits if gate_logits is not None else gate_weights
    validate_superposition_inputs(o_local, o_sparse, o_hca, gate_t)

    if o_local.shape[2] == 0:
        return torch.empty_like(o_local)

    return _TritonSuperpositionFunction.apply(
        o_local, o_sparse, o_hca, gate_logits, gate_weights
    )


__all__ = [
    "triton_compute_centroids",
    "triton_index_topk",
    "triton_stream_superposition",
]
