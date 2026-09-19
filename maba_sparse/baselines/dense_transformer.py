import math
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.model import FactorizedEmbeddings, MabaSparseOutput, RMSNorm, SwiGLUFFN


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    rx = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rx * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        cos = self.cos_cached[:, :, offset : offset + seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, offset : offset + seq_len, :].to(dtype=x.dtype, device=x.device)
        return cos, sin


class DenseAttention(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        n_heads: int = 10,
        d_head: int = 64,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.use_rope = use_rope

        self.q_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, dim, bias=False)

        if self.use_rope:
            self.rope = RotaryEmbedding(d_head, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        if self.use_rope and self.rope is not None:
            cos_q, sin_q = self.rope(q, offset=offset)
            cos_k, sin_k = self.rope(k, offset=offset)
            q = _apply_rotary_emb(q, cos_q, sin_q)
            k = _apply_rotary_emb(k, cos_k, sin_k)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        nkv = (k, v)
        if l > 1:
            if kv_cache is None:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                lkv = k.shape[2]
                mask = (torch.arange(l, device=x.device).unsqueeze(1) + (lkv - l)) >= torch.arange(lkv, device=x.device).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        o = o.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.d_head)
        return self.o_proj(o), nkv


class DenseTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        n_heads: int = 10,
        d_head: int = 64,
        intermediate_size: int = 1728,
        eps: float = 1e-6,
        residual_gate_bias: float = 2.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.mixer = DenseAttention(
            dim, n_heads, d_head, use_rope=use_rope, max_seq_len=max_seq_len
        )
        self.res_gate1 = nn.Parameter(torch.full((dim,), residual_gate_bias))

        self.norm2 = RMSNorm(dim, eps=eps)
        self.ffn = SwiGLUFFN(dim, intermediate_size)
        self.res_gate2 = nn.Parameter(torch.full((dim,), residual_gate_bias))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.norm1(x)
        ao, nkv = self.mixer(h, kv_cache=kv_cache)
        x = x + torch.sigmoid(self.res_gate1) * ao
        x = x + torch.sigmoid(self.res_gate2) * self.ffn(self.norm2(x))
        return x, nkv


class DenseTransformerForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32768,
        d_emb: int = 128,
        dim: int = 640,
        n_layers: int = 20,
        n_heads: int = 10,
        d_head: int = 64,
        intermediate_size: int = 1728,
        eps: float = 1e-6,
        residual_gate_bias: float = 2.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.embeddings = FactorizedEmbeddings(vocab_size, d_emb, dim)
        self.layers = nn.ModuleList([
            DenseTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                d_head=d_head,
                intermediate_size=intermediate_size,
                eps=eps,
                residual_gate_bias=residual_gate_bias,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim, eps=eps)
        self.head_proj = nn.Linear(dim, d_emb, bias=False)
        self.lm_head = nn.Linear(d_emb, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.in_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_states: Optional[List[Any]] = None,
    ) -> MabaSparseOutput:
        if targets is None and labels is not None:
            targets = labels

        x = self.embeddings(input_ids)
        nps = []

        for i, layer in enumerate(self.layers):
            kv = past_states[i] if past_states is not None and i < len(past_states) else None
            x, nkv = layer(x, kv_cache=kv)
            nps.append(nkv)

        xn = self.final_norm(x)
        logits = self.lm_head(self.head_proj(xn))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return MabaSparseOutput(
            logits=logits,
            loss=loss,
            past_states=nps,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            out = self(input_ids)
            past_states = out.past_states
            next_logits = out.logits[:, -1:, :]
            tokens = [input_ids]

            for i in range(max_new_tokens):
                nl = next_logits.squeeze(1)
                if temperature > 0:
                    nl = nl / temperature
                    if top_k is not None:
                        k_val = min(top_k, nl.size(-1))
                        v, _ = torch.topk(nl, k_val)
                        nl = nl.masked_fill(nl < v[:, [-1]], float("-inf"))
                    p = F.softmax(nl, dim=-1)
                    tok = torch.multinomial(p, num_samples=1)
                else:
                    tok = torch.argmax(nl, dim=-1, keepdim=True)
                tokens.append(tok)

                if i < max_new_tokens - 1:
                    out = self(tok, past_states=past_states)
                    past_states = out.past_states
                    next_logits = out.logits[:, -1:, :]

            return torch.cat(tokens, dim=1)
        finally:
            self.train(was_training)


DenseTransformerLM = DenseTransformerForCausalLM
DenseLM = DenseTransformerForCausalLM
