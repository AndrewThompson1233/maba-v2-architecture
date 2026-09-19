import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

from maba_sparse.kernels.dispatcher import dispatch_compute_centroids, dispatch_index_topk


class DGIndexer(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        d_idx: int = 64,
        block_size: int = 64,
        top_k: int = 32,
        dist_lambda: float = 0.5,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            dim = getattr(config, "dim", getattr(config, "d_model", dim))
            d_idx = getattr(config, "d_idx", d_idx)
            block_size = getattr(config, "block_size", block_size)
            top_k = getattr(config, "top_k", top_k)
            dist_lambda = getattr(config, "dist_lambda", dist_lambda)

        self.dim = dim
        self.d_idx = d_idx
        self.block_size = block_size
        self.top_k = top_k
        self.dist_lambda = dist_lambda

        self.q_idx_proj = nn.Linear(dim, d_idx, bias=False)
        self.k_idx_proj = nn.Linear(dim, d_idx, bias=False)
        self.scale = 1.0 / math.sqrt(d_idx)
        self._cached_k_idx: Optional[torch.Tensor] = None
        self._cached_centroids: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        self._cached_k_idx = None
        self._cached_centroids = None

    def forward(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        dist_lambda: Optional[float] = None,
        return_scores: bool = False,
        past_centroids: Optional[torch.Tensor] = None,
        past_k_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        b, l, d = x.shape
        k = self.top_k if top_k is None else top_k
        lam = self.dist_lambda if dist_lambda is None else dist_lambda

        qi = self.q_idx_proj(x)
        ki = self.k_idx_proj(x)

        if past_k_idx is not None:
            ki_full = torch.cat([past_k_idx, ki], dim=1)
            c = dispatch_compute_centroids(ki_full, block_size=self.block_size)
        elif not self.training and l == 1 and self._cached_k_idx is not None and self._cached_k_idx.shape[0] == b:
            self._cached_k_idx = torch.cat([self._cached_k_idx, ki.detach()], dim=1)
            cur_len = self._cached_k_idx.shape[1]
            if cur_len % self.block_size == 0:
                new_c = dispatch_compute_centroids(
                    self._cached_k_idx[:, -self.block_size :], block_size=self.block_size
                )
                if self._cached_centroids is not None:
                    self._cached_centroids = torch.cat([self._cached_centroids, new_c], dim=1)
                else:
                    self._cached_centroids = new_c
                c = self._cached_centroids
            else:
                rem = cur_len % self.block_size
                tail_c = dispatch_compute_centroids(
                    self._cached_k_idx[:, -rem:], block_size=self.block_size
                )
                if self._cached_centroids is not None:
                    c = torch.cat([self._cached_centroids, tail_c], dim=1)
                else:
                    c = tail_c
        elif past_centroids is not None:
            c_curr = dispatch_compute_centroids(ki, block_size=self.block_size)
            c = torch.cat([past_centroids, c_curr], dim=1)
        else:
            c = dispatch_compute_centroids(ki, block_size=self.block_size)
            if not self.training and l > 1:
                self._cached_k_idx = ki.detach()
                num_comp = l // self.block_size
                if num_comp > 0:
                    self._cached_centroids = c[:, :num_comp, :].detach()
                else:
                    self._cached_centroids = None
            else:
                self._cached_k_idx = None
                self._cached_centroids = None

        nb = c.shape[1]

        if l == 1 and (past_centroids is not None or past_k_idx is not None or nb > 1):
            q_blk = nb - 1
            ni = torch.arange(nb, device=c.device)
            dist = (q_blk - ni).clamp(min=0).float()
            pen = lam * torch.log1p(dist)
            dot = torch.matmul(qi * self.scale, c.transpose(-1, -2))
            scores = dot - pen.view(1, 1, nb)
            ak = min(k, nb)
            idx = torch.topk(scores, k=ak, dim=-1, largest=True, sorted=True).indices
            if return_scores:
                return idx, c, scores
            return idx, c

        res = dispatch_index_topk(
            q_idx=qi,
            centroids=c,
            lambda_dist=lam,
            top_k=k,
            block_size=self.block_size,
            return_scores=return_scores,
        )

        if return_scores:
            idx, sc = res
            return idx, c, sc
        return res, c


DeltaGuidedCentroidIndexer = DGIndexer
