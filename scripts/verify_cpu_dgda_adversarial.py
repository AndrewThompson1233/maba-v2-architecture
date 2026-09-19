
import json
import math
import os
import sys
import time
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from maba_sparse.kernels.common import (
    normalize_keys,
    ref_dgda_prefill,
    ref_dgda_step,
)
from maba_sparse.kernels.cpu_dgda import (
    cpu_dgda_prefill,
    cpu_dgda_step,
    cpu_threads,
    get_cpu_num_threads,
)


def run_all_benchmarks():
    results = {}

    decay_metrics = []
    for alpha_val in [0.0, 1e-7, 1e-20, 1e-35, 1.0]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.manual_seed(1001)
            B, H, L, dk, dv = 2, 2, 64, 16, 16
            q = torch.randn(B, H, L, dk, dtype=dtype)
            k = normalize_keys(torch.randn(B, H, L, dk, dtype=dtype))
            v = torch.randn(B, H, L, dv, dtype=dtype)
            alpha = torch.full_like(q, alpha_val)
            b = torch.rand(B, H, L, dk, dtype=dtype)
            w = torch.rand(B, H, L, dv, dtype=dtype)

            out_cpu, state_cpu = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
            out_ref, state_ref = ref_dgda_prefill(
                q.float(), k.float(), v.float(), alpha.float(), b.float(), w.float()
            )

            has_nan = bool(torch.isnan(out_cpu).any() or torch.isnan(state_cpu).any())
            has_inf = bool(torch.isinf(out_cpu).any() or torch.isinf(state_cpu).any())
            diff_out = float((out_cpu.float() - out_ref).abs().max().item())
            diff_state = float((state_cpu.float() - state_ref).abs().max().item())

            decay_metrics.append({
                "alpha": alpha_val,
                "dtype": str(dtype).replace("torch.", ""),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "diff_out": diff_out,
                "diff_state": diff_state,
            })
    results["decay_regimes"] = decay_metrics

    gate_metrics = []
    gate_configs = [
        ("Pure Erase (b=1, w=0)", 1.0, 0.0),
        ("Pure Write (b=0, w=1)", 0.0, 1.0),
        ("Frozen State (b=0, w=0)", 0.0, 0.0),
        ("Full Update (b=1, w=1)", 1.0, 1.0),
    ]
    for name, b_val, w_val in gate_configs:
        torch.manual_seed(2001)
        B, H, L, dk, dv = 2, 2, 64, 16, 16
        q = torch.randn(B, H, L, dk)
        k = normalize_keys(torch.randn(B, H, L, dk))
        v = torch.randn(B, H, L, dv)
        alpha = torch.rand(B, H, L, dk) * 0.9 + 0.05
        b = torch.full_like(k, b_val)
        w = torch.full_like(v, w_val)
        init_state = torch.randn(B, H, dk, dv)

        out_cpu, state_cpu = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state)
        out_ref, state_ref = ref_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state)

        diff_out = float((out_cpu - out_ref).abs().max().item())
        diff_state = float((state_cpu - state_ref).abs().max().item())
        has_nan = bool(torch.isnan(out_cpu).any() or torch.isnan(state_cpu).any())
        has_inf = bool(torch.isinf(out_cpu).any() or torch.isinf(state_cpu).any())

        gate_metrics.append({
            "regime": name,
            "diff_out": diff_out,
            "diff_state": diff_state,
            "has_nan": has_nan,
            "has_inf": has_inf,
        })
    results["gate_regimes"] = gate_metrics

    spectral_metrics = []
    for colinearity in [0.9, 0.99, 1.0]:
        torch.manual_seed(3001)
        B, H, L, dk, dv = 2, 2, 64, 16, 16
        base_key = torch.randn(1, 1, 1, dk)
        base_key = base_key / torch.linalg.norm(base_key, dim=-1, keepdim=True)
        noise = torch.randn(B, H, L, dk) * (1.0 - colinearity)
        k = normalize_keys(base_key.expand(B, H, L, dk) + noise)
        q = torch.randn(B, H, L, dk)
        v = torch.randn(B, H, L, dv)
        alpha = torch.full((B, H, L, dk), 0.9999)
        b = torch.ones(B, H, L, dk)
        w = torch.ones(B, H, L, dv)

        lac = torch.log(alpha[:, :, :16])
        cla = torch.cumsum(lac, dim=-2)
        diff = cla.unsqueeze(-2) - cla.unsqueeze(-3)
        dec = torch.exp(torch.clamp(diff, max=0.0))
        bk = (b[:, :, :16] * k[:, :, :16]).unsqueeze(-2)
        ks = k[:, :, :16].unsqueeze(-3)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)
        spectral_norm = float(torch.linalg.svdvals(l_mat[0, 0])[0].item())

        out_neumann, state_neumann = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, inversion_method="neumann"
        )
        out_adaptive, state_adaptive = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, inversion_method="adaptive"
        )
        out_exact, state_exact = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, inversion_method="exact"
        )
        out_ref, state_ref = ref_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        neumann_diff_vs_ref = float((out_neumann - out_ref).abs().max().item())
        adaptive_diff_vs_ref = float((out_adaptive - out_ref).abs().max().item())
        adaptive_diff_vs_exact = float((out_adaptive - out_exact).abs().max().item())

        spectral_metrics.append({
            "colinearity": colinearity,
            "spectral_norm_L2": spectral_norm,
            "spectral_norm_ge_1": bool(spectral_norm >= 1.0),
            "neumann_diff_vs_ref": neumann_diff_vs_ref,
            "adaptive_diff_vs_ref": adaptive_diff_vs_ref,
            "adaptive_diff_vs_exact": adaptive_diff_vs_exact,
            "adaptive_matches_within_1e4": bool(adaptive_diff_vs_ref < 1e-4),
        })
    results["spectral_stress"] = spectral_metrics

    concurrency_metrics = []
    B, H, L, dk, dv = 4, 8, 128, 32, 32
    torch.manual_seed(4001)
    q = torch.randn(B, H, L, dk)
    k = normalize_keys(torch.randn(B, H, L, dk))
    v = torch.randn(B, H, L, dv)
    alpha = torch.rand(B, H, L, dk) * 0.9 + 0.05
    b = torch.rand(B, H, L, dk)
    w = torch.rand(B, H, L, dv)

    with cpu_threads(1):
        t0 = time.perf_counter()
        out_1, state_1 = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
        lat_1 = (time.perf_counter() - t0) * 1000

    for nth in [1, 2, 4, 8]:
        with cpu_threads(nth):
            t0 = time.perf_counter()
            out_n, state_n = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
            lat_n = (time.perf_counter() - t0) * 1000

        diff_out = float((out_n - out_1).abs().max().item())
        diff_state = float((state_n - state_1).abs().max().item())

        concurrency_metrics.append({
            "threads": nth,
            "latency_ms": round(lat_n, 3),
            "diff_out_vs_1thread": diff_out,
            "diff_state_vs_1thread": diff_state,
            "bitwise_deterministic": bool(diff_out == 0.0),
        })
    results["concurrency"] = concurrency_metrics

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_all_benchmarks()
