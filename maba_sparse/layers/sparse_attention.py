import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.kernels.dispatcher import dispatch_stream_superposition
from maba_sparse.layers.indexer import DGIndexer


class MabaSparseAttention(nn.Module):
    def __init__(
        self,
        config: Optional[Any] = None,
        dim: int = 640,
        n_heads: int = 10,
        d_head: int = 64,
        d_c: int = 128,
        window_size: int = 128,
        block_size: int = 64,
        top_k: int = 32,
        hca_pool_size: int = 64,
        dist_lambda: float = 0.5,
        d_idx: int = 64,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        if config is not None:
            self.dim = getattr(config, "dim", getattr(config, "d_model", dim))
            self.n_heads = getattr(config, "n_heads", getattr(config, "num_heads", n_heads))
            self.d_head = getattr(config, "d_head", d_head)
            self.d_c = getattr(config, "d_c", d_c)
            self.window_size = getattr(config, "window_size", window_size)
            self.block_size = getattr(config, "block_size", block_size)
            self.top_k = getattr(config, "top_k", top_k)
            self.hca_pool_size = getattr(config, "hca_pool_size", hca_pool_size)
            self.dist_lambda = getattr(config, "dist_lambda", dist_lambda)
            self.n_sink_tokens = getattr(config, "n_sink_tokens", 4)
            d_idx = getattr(config, "d_idx", d_idx)
            self.ablation_mode = ablation_mode if ablation_mode is not None else getattr(config, "ablation_mode", "full")
        else:
            self.dim = dim
            self.n_heads = n_heads
            self.d_head = d_head
            self.d_c = d_c
            self.window_size = window_size
            self.block_size = block_size
            self.top_k = top_k
            self.hca_pool_size = hca_pool_size
            self.dist_lambda = dist_lambda
            self.n_sink_tokens = 4
            self.ablation_mode = ablation_mode

        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.d_head, bias=False)
        self.kv_down_proj = nn.Linear(self.dim, self.d_c, bias=False)
        self.k_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)
        self.v_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.dim, bias=False)
        self.stream_gate = nn.Linear(self.dim, 3, bias=True)
        self.indexer = DGIndexer(
            dim=self.dim,
            d_idx=d_idx,
            block_size=self.block_size,
            top_k=self.top_k,
            dist_lambda=self.dist_lambda,
        )
        self._cached_k: Optional[torch.Tensor] = None
        self._cached_v: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        self.indexer.reset_cache()
        self._cached_k = None
        self._cached_v = None

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.stream_gate(x), dim=-1)

    def _compute_local_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L_full: int
    ) -> torch.Tensor:
        b, h, lq, _ = q.shape
        lkv = k.shape[2]
        if lq == 1 and lkv > (self.window_size + self.n_sink_tokens):
            sink_idx = torch.arange(min(self.n_sink_tokens, lkv), device=q.device)
            win_start = max(min(self.n_sink_tokens, lkv), lkv - self.window_size)
            win_idx = torch.arange(win_start, lkv, device=q.device)
            sel_idx = torch.cat([sink_idx, win_idx])
            k_sel = k[:, :, sel_idx, :]
            v_sel = v[:, :, sel_idx, :]
            a = torch.matmul(q, k_sel.transpose(-1, -2)) * self.scale
            p = F.softmax(a, dim=-1)
            return torch.matmul(p, v_sel)

        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        qo = lkv - lq
        i = (qo + torch.arange(lq, device=q.device)).unsqueeze(1)
        j = torch.arange(lkv, device=q.device).unsqueeze(0)
        c = j <= i
        w = (i - j) < self.window_size
        s = j < min(self.n_sink_tokens, lkv)
        m = c & (w | s)
        a = a.masked_fill(~m.unsqueeze(0).unsqueeze(0), float("-inf"))
        p = torch.nan_to_num(F.softmax(a, dim=-1), nan=0.0)
        return torch.matmul(p, v)

    def _compute_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        top_indices: torch.Tensor,
        L_full: int,
    ) -> torch.Tensor:
        b, h, lq, dh = q.shape
        lkv = k.shape[2]
        nb = (lkv + self.block_size - 1) // self.block_size
        if nb == 0 or lq == 0:
            return torch.zeros(b, h, lq, dh, device=q.device, dtype=q.dtype)

        ak = top_indices.shape[-1]
        pad_len = nb * self.block_size - lkv
        kp = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
        vp = F.pad(v, (0, 0, 0, pad_len)) if pad_len > 0 else v
        kb = kp.view(b, h, nb, self.block_size, dh)
        vb = vp.view(b, h, nb, self.block_size, dh)
        qo = lkv - lq

        if lq == 1:
            bi = top_indices.squeeze(1)
            gather_idx = bi.view(b, 1, ak, 1, 1).expand(b, h, ak, self.block_size, dh)
            k_sel = torch.gather(kb, 2, gather_idx).reshape(b, h, ak * self.block_size, dh)
            v_sel = torch.gather(vb, 2, gather_idx).reshape(b, h, ak * self.block_size, dh)

            a = torch.matmul(q, k_sel.transpose(-1, -2)) * self.scale
            token_offsets = (
                bi.unsqueeze(-1) * self.block_size
                + torch.arange(self.block_size, device=q.device).view(1, 1, self.block_size)
            ).reshape(b, 1, ak * self.block_size)
            valid = (token_offsets < lkv) & (token_offsets <= (lkv - 1))
            a = a.masked_fill(~valid.unsqueeze(1), float("-inf"))
            p = torch.nan_to_num(F.softmax(a, dim=-1), nan=0.0)
            return torch.matmul(p, v_sel)

        os_chunks = []
        chunk_sz = 128
        for q_start in range(0, lq, chunk_sz):
            q_end = min(q_start + chunk_sz, lq)
            c_len = q_end - q_start
            qc = q[:, :, q_start:q_end, :]
            bi = top_indices[:, q_start:q_end, :]

            gather_idx = bi.view(b, 1, c_len, ak, 1, 1).expand(b, h, c_len, ak, self.block_size, dh)
            kb_exp = kb.view(b, h, 1, nb, self.block_size, dh).expand(b, h, c_len, nb, self.block_size, dh)
            vb_exp = vb.view(b, h, 1, nb, self.block_size, dh).expand(b, h, c_len, nb, self.block_size, dh)
            k_sel = torch.gather(kb_exp, 3, gather_idx).reshape(b, h, c_len, ak * self.block_size, dh)
            v_sel = torch.gather(vb_exp, 3, gather_idx).reshape(b, h, c_len, ak * self.block_size, dh)

            ac = (qc.unsqueeze(-2) @ k_sel.transpose(-1, -2)).squeeze(-2) * self.scale
            pos = (qo + torch.arange(q_start, q_end, device=q.device)).view(1, c_len, 1)
            token_offsets = (
                bi.unsqueeze(-1) * self.block_size
                + torch.arange(self.block_size, device=q.device).view(1, 1, 1, self.block_size)
            ).reshape(b, c_len, ak * self.block_size)
            valid = (token_offsets < lkv) & (token_offsets <= pos)
            ac = ac.masked_fill(~valid.unsqueeze(1), float("-inf"))
            pc = torch.nan_to_num(F.softmax(ac, dim=-1), nan=0.0)
            oc = (pc.unsqueeze(-2) @ v_sel).squeeze(-2)
            os_chunks.append(oc)

        return torch.cat(os_chunks, dim=2)

    def _compute_hca_attention(
        self, q: torch.Tensor, c_kv: torch.Tensor, L_full: int
    ) -> torch.Tensor:
        b, h, lq, dh = q.shape
        lkv = c_kv.shape[1]
        r = self.hca_pool_size
        if lkv == 0 or lq == 0:
            return torch.zeros(b, h, lq, dh, device=q.device, dtype=q.dtype)

        qo = lkv - lq
        pos = qo + torch.arange(lq, device=q.device)

        num_complete = lkv // r
        early_mask = pos < (r - 1)
        out = torch.empty(b, h, lq, dh, device=q.device, dtype=q.dtype)

        if early_mask.any():
            early_indices = torch.where(early_mask)[0]
            early_pos = pos[early_mask]
            max_early_pos = early_pos.max().item()
            c_prefix = c_kv[:, : max_early_pos + 1]
            c_cum = torch.cumsum(c_prefix, dim=1)
            denom = torch.arange(1, max_early_pos + 2, device=c_kv.device).view(1, -1, 1)
            c_mean = c_cum / denom
            c_pool_early = c_mean[:, early_pos, :]
            vh_early = self.v_up_proj(c_pool_early).view(b, len(early_pos), h, dh).transpose(1, 2)
            out[:, :, early_indices, :] = vh_early

        if (~early_mask).any() and num_complete > 0:
            late_indices = torch.where(~early_mask)[0]
            late_pos = pos[~early_mask]
            q_late = q[:, :, late_indices, :]

            c_complete = c_kv[:, : num_complete * r].view(b, num_complete, r, self.d_c).mean(dim=2)
            kh = self.k_up_proj(c_complete).view(b, num_complete, h, dh).transpose(1, 2)
            vh = self.v_up_proj(c_complete).view(b, num_complete, h, dh).transpose(1, 2)

            a = torch.matmul(q_late, kh.transpose(-1, -2)) * self.scale
            qi = late_pos.unsqueeze(1)
            bs = torch.arange(num_complete, device=q.device) + 1
            ei = (bs * r - 1).unsqueeze(0)
            c = ei <= qi
            a = a.masked_fill(~c.unsqueeze(0).unsqueeze(0), float("-inf"))
            p = torch.nan_to_num(F.softmax(a, dim=-1), nan=0.0)
            out[:, :, late_indices, :] = torch.matmul(p, vh)

        return out

    def forward(
        self,
        x: torch.Tensor,
        past_c_kv: Optional[torch.Tensor] = None,
        past_centroids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
        if past_c_kv is None or (
            self._cached_k is not None and self._cached_k.shape[2] != past_c_kv.shape[1]
        ) or (
            self.indexer._cached_k_idx is not None and self.indexer._cached_k_idx.shape[1] != past_c_kv.shape[1]
        ):
            self.reset_cache()

        c_kv = self.kv_down_proj(x)
        c_full = torch.cat([past_c_kv, c_kv], dim=1) if past_c_kv is not None else c_kv
        lf = c_full.shape[1]

        q = self.q_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        if (
            not self.training
            and past_c_kv is not None
            and self._cached_k is not None
            and self._cached_k.shape[0] == b
            and self._cached_k.shape[2] == past_c_kv.shape[1]
        ):
            k_new = self.k_up_proj(c_kv).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
            v_new = self.v_up_proj(c_kv).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
            k = torch.cat([self._cached_k, k_new], dim=2)
            v = torch.cat([self._cached_v, v_new], dim=2)
        else:
            k = self.k_up_proj(c_full).view(b, lf, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v_up_proj(c_full).view(b, lf, self.n_heads, self.d_head).transpose(1, 2)

        if not self.training:
            self._cached_k = k
            self._cached_v = v
        else:
            self._cached_k = None
            self._cached_v = None

        ol = self._compute_local_attention(q, k, v, lf)
        idx, _ = self.indexer(
            x, return_scores=False, past_centroids=past_centroids
        )
        os = self._compute_sparse_attention(q, k, v, idx, lf)

        if self.ablation_mode == "no_hca":
            oh = torch.zeros_like(ol)
        else:
            oh = self._compute_hca_attention(q, c_full, lf)

        gate_logits = self.stream_gate(x)
        o = dispatch_stream_superposition(ol, os, oh, gate_logits=gate_logits)
        o = o.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.d_head).to(q.dtype)
        return self.o_proj(o), c_full


MABASALayer = MabaSparseAttention
