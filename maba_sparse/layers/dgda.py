from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.kernels.dispatcher import dispatch_dgda_prefill, dispatch_dgda_step


class ConvState(tuple):
    @property
    def shape(self) -> torch.Size:
        if len(self) == 0:
            return torch.Size([0, 0, 0, 0])
        f = self[0]
        return torch.Size([f.shape[0], len(self), f.shape[1], f.shape[2]])

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(self, dim=1)


class DGDALayer(nn.Module):
    def __init__(
        self,
        config: Optional[Any] = None,
        dim: int = 640,
        n_heads: int = 10,
        d_head: Optional[int] = None,
        kernel_size: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if config is not None:
            dim = getattr(config, "dim", getattr(config, "d_model", dim))
            n_heads = getattr(config, "n_heads", getattr(config, "num_heads", n_heads))
            d_head = getattr(config, "d_head", d_head)
            kernel_size = getattr(
                config, "kernel_size", getattr(config, "conv_kernel_size", kernel_size)
            )
            eps = getattr(config, "eps", getattr(config, "rms_norm_eps", eps))

        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else (dim // n_heads)
        self.d_k = getattr(config, "d_k", self.d_head) if config is not None else self.d_head
        self.d_v = getattr(config, "d_v", self.d_head) if config is not None else self.d_head
        self.kernel_size = kernel_size
        self.k_size = kernel_size
        self.eps = eps
        self.chunk_size = getattr(config, "chunk_size", 16)
        self.inversion_method = getattr(config, "inversion_method", "adaptive")
        self.adaptive_tol = getattr(config, "adaptive_tol", 7e-5)

        qk_dim = self.n_heads * self.d_k
        v_dim = self.n_heads * self.d_v

        self.q_proj = nn.Linear(self.dim, qk_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, qk_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, v_dim, bias=False)

        self.conv_q = nn.Conv1d(
            qk_dim, qk_dim, self.kernel_size, groups=qk_dim, bias=False, padding=0
        )
        self.conv_k = nn.Conv1d(
            qk_dim, qk_dim, self.kernel_size, groups=qk_dim, bias=False, padding=0
        )
        self.conv_v = nn.Conv1d(
            v_dim, v_dim, self.kernel_size, groups=v_dim, bias=False, padding=0
        )

        self.gate_alpha = nn.Linear(self.dim, qk_dim, bias=False)
        self.gate_erase = nn.Linear(self.dim, qk_dim, bias=False)
        self.gate_write = nn.Linear(self.dim, v_dim, bias=False)

        self.o_proj = nn.Linear(v_dim, self.dim, bias=False)

        self._reset_parameters()

    @property
    def alpha_proj(self) -> nn.Linear:
        return self.gate_alpha

    @property
    def b_proj(self) -> nn.Linear:
        return self.gate_erase

    @property
    def w_proj(self) -> nn.Linear:
        return self.gate_write

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.gate_alpha.weight)
        nn.init.xavier_uniform_(self.gate_erase.weight)
        nn.init.xavier_uniform_(self.gate_write.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.normal_(self.conv_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_v.weight, mean=0.0, std=0.02)

    def _apply_conv(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.kernel_size
        xt = x.transpose(1, 2)
        if conv_state is not None:
            if k > 1 and conv_state.shape[-1] != k - 1:
                if conv_state.shape[-1] < k - 1:
                    conv_state = F.pad(conv_state, (k - 1 - conv_state.shape[-1], 0))
                else:
                    conv_state = conv_state[:, :, -(k - 1):]
            p = torch.cat([conv_state, xt], dim=2)
        else:
            p = F.pad(xt, (k - 1, 0)) if k > 1 else xt
        ns = p[:, :, -(k - 1):].contiguous() if k > 1 else p[:, :, :0].contiguous()
        y = F.silu(conv(p)).transpose(1, 2)
        return y, ns

    def _unpack_conv_state(
        self,
        conv_state: Optional[Union[ConvState, Tuple[torch.Tensor, ...], torch.Tensor]],
        b_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if conv_state is None:
            return None, None, None

        def _cast(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device=device, dtype=dtype)

        k_hist = max(self.kernel_size - 1, 0)

        if isinstance(conv_state, (tuple, list)):
            if len(conv_state) == 3:
                return _cast(conv_state[0]), _cast(conv_state[1]), _cast(conv_state[2])
            if len(conv_state) == 1 and isinstance(conv_state[0], (tuple, list)):
                return _cast(conv_state[0][0]), _cast(conv_state[0][1]), _cast(conv_state[0][2])
        if isinstance(conv_state, torch.Tensor):
            if conv_state.dim() == 4 and conv_state.shape[1] == 3:
                return _cast(conv_state[:, 0]), _cast(conv_state[:, 1]), _cast(conv_state[:, 2])
            if conv_state.dim() == 5 and conv_state.shape[1] == 3:
                cs = conv_state.reshape(b_size, 3, conv_state.shape[2], k_hist)
                return _cast(cs[:, 0]), _cast(cs[:, 1]), _cast(cs[:, 2])
            cs = conv_state.reshape(b_size, -1, k_hist)
            tot_ch = cs.shape[1]
            qk_ch = self.n_heads * self.d_k
            v_ch = self.n_heads * self.d_v
            if tot_ch == 2 * qk_ch + v_ch:
                return _cast(cs[:, :qk_ch]), _cast(cs[:, qk_ch : 2 * qk_ch]), _cast(cs[:, 2 * qk_ch :])
            if tot_ch == qk_ch and qk_ch == v_ch:
                return _cast(cs), _cast(cs), _cast(cs)
            if tot_ch == qk_ch:
                zero_v = torch.zeros(b_size, v_ch, k_hist, device=device, dtype=dtype)
                return _cast(cs), _cast(cs), zero_v
            return _cast(cs), _cast(cs), _cast(cs)
        raise ValueError(f"Unsupported conv_state shape or type: {type(conv_state)}")

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        chunk_size: Optional[int] = None,
        inversion_method: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConvState]:
        b, l, d = x.shape
        h, dk, dv = self.n_heads, self.d_k, self.d_v
        qk_dim = h * dk
        v_dim = h * dv

        if l == 0:
            eo = torch.empty(b, 0, d, dtype=x.dtype, device=x.device)
            es = state.clone() if state is not None else torch.zeros(b, h, dk, dv, dtype=x.dtype, device=x.device)
            k_hist = max(self.kernel_size - 1, 0)
            zc_qk = torch.zeros(b, qk_dim, k_hist, dtype=x.dtype, device=x.device)
            zc_v = torch.zeros(b, v_dim, k_hist, dtype=x.dtype, device=x.device)
            return eo, es, ConvState((zc_qk, zc_qk, zc_v))

        cq, ck, cv = self._unpack_conv_state(conv_state, b, x.device, x.dtype)
        q, nq = self._apply_conv(self.q_proj(x), self.conv_q, cq)
        k, nk = self._apply_conv(self.k_proj(x), self.conv_k, ck)
        v, nv = self._apply_conv(self.v_proj(x), self.conv_v, cv)
        ncs = ConvState((nq, nk, nv))

        eb = torch.sigmoid(self.gate_erase(x))
        ew = torch.sigmoid(self.gate_write(x))
        la = -F.softplus(self.gate_alpha(x))
        la = torch.clamp(la, min=-14.0)
        alpha = torch.exp(la)

        q = q.view(b, l, h, dk).transpose(1, 2)
        k = k.view(b, l, h, dk).transpose(1, 2)
        v = v.view(b, l, h, dv).transpose(1, 2)
        eb = eb.view(b, l, h, dk).transpose(1, 2)
        ew = ew.view(b, l, h, dv).transpose(1, 2)
        alpha = alpha.view(b, l, h, dk).transpose(1, 2)
        la = la.view(b, l, h, dk).transpose(1, 2)

        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.eps)

        k = k.to(q.dtype)
        v = v.to(q.dtype)
        eb = eb.to(q.dtype)
        ew = ew.to(q.dtype)
        alpha = alpha.to(q.dtype)
        la = la.to(q.dtype)
        if state is not None:
            state = state.to(q.dtype)

        cs_val = chunk_size if chunk_size is not None else self.chunk_size
        inv_val = inversion_method if inversion_method is not None else self.inversion_method

        o, cs = dispatch_dgda_prefill(
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            b=eb,
            w=ew,
            chunk_size=cs_val,
            initial_state=state,
            inversion_method=inv_val,
            adaptive_tol=self.adaptive_tol,
            log_alpha=la,
        )

        o = o.transpose(1, 2).contiguous().view(b, l, h * dv)
        return self.o_proj(o), cs, ncs

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConvState]:
        b, l = x.shape[0], x.shape[1]
        assert l == 1
        h, dk, dv = self.n_heads, self.d_k, self.d_v

        cq, ck, cv = self._unpack_conv_state(conv_state, b, x.device, x.dtype)
        q, nq = self._apply_conv(self.q_proj(x), self.conv_q, cq)
        k, nk = self._apply_conv(self.k_proj(x), self.conv_k, ck)
        v, nv = self._apply_conv(self.v_proj(x), self.conv_v, cv)
        ncs = ConvState((nq, nk, nv))

        eb = torch.sigmoid(self.gate_erase(x)).view(b, 1, h, dk).transpose(1, 2)
        ew = torch.sigmoid(self.gate_write(x)).view(b, 1, h, dv).transpose(1, 2)
        la = -F.softplus(self.gate_alpha(x)).view(b, 1, h, dk).transpose(1, 2)
        la = torch.clamp(la, min=-14.0)
        alpha = torch.exp(la)

        q = q.view(b, 1, h, dk).transpose(1, 2)
        k = k.view(b, 1, h, dk).transpose(1, 2)
        v = v.view(b, 1, h, dv).transpose(1, 2)

        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.eps)

        k = k.to(q.dtype)
        v = v.to(q.dtype)
        eb = eb.to(q.dtype)
        ew = ew.to(q.dtype)
        alpha = alpha.to(q.dtype)
        la = la.to(q.dtype)
        if state is not None:
            state = state.to(q.dtype)

        oh, st = dispatch_dgda_step(
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            b=eb,
            w=ew,
            state=state,
            log_alpha=la,
        )

        o = self.o_proj(oh.reshape(b, 1, h * dv))
        return o, st, ncs


DecoupledGatedDeltaAttention = DGDALayer
