from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.layers.sparse_attention import MabaSparseAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = max(float(eps), 1e-6)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, eps=self.eps)
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        v = xf.pow(2).mean(-1, keepdim=True)
        r = torch.rsqrt(v + self.eps)
        return (xf * r).to(orig_dtype) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, intermediate_size: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_size, bias=False)
        self.w_up = nn.Linear(dim, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FactorizedEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_emb: int, dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.dim = dim
        self.in_emb = nn.Embedding(vocab_size, d_emb)
        self.proj = nn.Linear(d_emb, dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.in_emb(input_ids))


class MTPHead(nn.Module):
    def __init__(self, dim: int, d_emb: int, lm_head: nn.Linear) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, d_emb, bias=False)
        self.lm_head = lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.proj(x))


class MabaBlock(nn.Module):
    def __init__(
        self,
        config: MabaSparseConfig,
        layer_idx: int,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ablation_mode = ablation_mode

        if ablation_mode == "pure_dgda":
            self.is_attention = False
        else:
            self.is_attention = (layer_idx + 1) % 4 == 0

        self.norm1 = RMSNorm(config.dim, eps=config.rms_norm_eps)
        if self.is_attention:
            self.mixer = MabaSparseAttention(config, ablation_mode=ablation_mode)
        else:
            self.mixer = DGDALayer(config)

        self.norm2 = RMSNorm(config.dim, eps=config.rms_norm_eps)
        inter = getattr(config, "intermediate_size", 1728)
        self.ffn = SwiGLUFFN(config.dim, inter)

        b = getattr(config, "residual_gate_bias", 2.0)
        self.res_gate1 = nn.Parameter(torch.full((config.dim,), b))
        self.res_gate2 = nn.Parameter(torch.full((config.dim,), b))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_state: Optional[Any] = None,
        past_c_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any], Optional[torch.Tensor]]:
        h = self.norm1(x)
        if self.is_attention:
            mo, nc = self.mixer(h, past_c_kv=past_c_kv)
            ns, ncv = None, None
        else:
            mo, ns, ncv = self.mixer(h, state=state, conv_state=conv_state)
            nc = None

        x = x + torch.sigmoid(self.res_gate1) * mo
        x = x + torch.sigmoid(self.res_gate2) * self.ffn(self.norm2(x))
        return x, ns, ncv, nc

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_state: Optional[Any] = None,
        past_c_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any], Optional[torch.Tensor]]:
        h = self.norm1(x)
        if self.is_attention:
            mo, nc = self.mixer(h, past_c_kv=past_c_kv)
            ns, ncv = None, None
        else:
            mo, ns, ncv = self.mixer.step(h, state=state, conv_state=conv_state)
            nc = None

        x = x + torch.sigmoid(self.res_gate1) * mo
        x = x + torch.sigmoid(self.res_gate2) * self.ffn(self.norm2(x))
        return x, ns, ncv, nc


class MabaSparseOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        mtp_logits: Optional[torch.Tensor] = None,
        past_states: Optional[List[Any]] = None,
    ) -> None:
        self.logits = logits
        self.loss = loss
        self.mtp_logits = mtp_logits
        self.past_states = past_states

    def __iter__(self):
        return iter((self.logits, self.loss))

    def __getitem__(self, idx: Union[int, str]) -> Any:
        if isinstance(idx, str):
            if hasattr(self, idx):
                return getattr(self, idx)
            raise KeyError(f"No key '{idx}' in MabaSparseOutput")
        return (self.logits, self.loss, self.past_states, self.mtp_logits)[idx]

    def __repr__(self) -> str:
        return (
            f"MabaSparseOutput(logits={tuple(self.logits.shape)}, "
            f"loss={self.loss.item() if self.loss is not None else None}, "
            f"mtp_logits={tuple(self.mtp_logits.shape) if self.mtp_logits is not None else None})"
        )


def get_101m_config(
    intermediate_size: int = 1248,
    vocab_size: int = 32768,
    d_emb: int = 128,
    n_layers: int = 20,
) -> MabaSparseConfig:
    return MabaSparseConfig(
        dim=640,
        n_heads=10,
        d_head=64,
        n_layers=n_layers,
        vocab_size=vocab_size,
        d_emb=d_emb,
        intermediate_size=intermediate_size,
        residual_gate_bias=2.0,
    )


class MabaSparseForCausalLM(nn.Module):
    def __init__(
        self,
        config: Optional[MabaSparseConfig] = None,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        if config is None:
            config = get_101m_config()
        self.config = config
        self.ablation_mode = ablation_mode

        v = getattr(config, "vocab_size", 32768)
        de = getattr(config, "d_emb", 128)
        d = getattr(config, "dim", 640)
        nl = getattr(config, "n_layers", 20)

        self.embeddings = FactorizedEmbeddings(v, de, d)
        self.layers = nn.ModuleList([
            MabaBlock(config, i, ablation_mode=ablation_mode) for i in range(nl)
        ])
        self.final_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.head_proj = nn.Linear(d, de, bias=False)
        self.lm_head = nn.Linear(de, v, bias=False)
        self.lm_head.weight = self.embeddings.in_emb.weight
        self.mtp_head = MTPHead(d, de, self.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_states: Optional[List[Any]] = None,
    ) -> MabaSparseOutput:
        if targets is None and labels is not None:
            targets = labels

        b, l = input_ids.shape
        x = self.embeddings(input_ids)

        nps = []
        for i, layer in enumerate(self.layers):
            ls = past_states[i] if past_states is not None and i < len(past_states) else None
            st = ls[0] if ls else None
            cv = ls[1] if ls else None
            pk = ls[2] if ls else None

            x, nst, ncv, nck = layer(x, state=st, conv_state=cv, past_c_kv=pk)
            nps.append((nst, ncv, nck))

        xn = self.final_norm(x)
        logits = self.lm_head(self.head_proj(xn))

        loss = None
        mtp_logits = None
        if targets is not None:
            v_size = self.config.vocab_size
            loss = F.cross_entropy(logits.view(-1, v_size), targets.view(-1), ignore_index=-100)
            mtp_logits = self.mtp_head(xn[:, :-1, :])
            if mtp_logits.numel() > 0:
                ml = F.cross_entropy(
                    mtp_logits.contiguous().view(-1, v_size),
                    targets[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss = loss + 0.3 * ml
            else:
                loss = loss + 0.0 * mtp_logits.sum()

        return MabaSparseOutput(
            logits=logits,
            loss=loss,
            mtp_logits=mtp_logits,
            past_states=nps,
        )

    def step(
        self,
        input_ids: torch.Tensor,
        past_states: Optional[List[Any]] = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        b, l = input_ids.shape
        assert l == 1, f"step() expects single-token input (L=1), got L={l}"
        x = self.embeddings(input_ids)
        next_states = []
        for i, layer in enumerate(self.layers):
            ls = past_states[i] if past_states is not None and i < len(past_states) else None
            st = ls[0] if ls else None
            cv = ls[1] if ls else None
            pk = ls[2] if ls else None

            x, nst, ncv, nck = layer.step(x, state=st, conv_state=cv, past_c_kv=pk)
            next_states.append((nst, ncv, nck))

        xn = self.final_norm(x)
        logits = self.lm_head(self.head_proj(xn))
        return logits, next_states

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
                    next_logits, past_states = self.step(tok, past_states=past_states)

            return torch.cat(tokens, dim=1)
        finally:
            self.train(was_training)


MabaSparseLM = MabaSparseForCausalLM
MabaLM = MabaSparseForCausalLM
MabaOutput = MabaSparseOutput
