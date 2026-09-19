
import os
import sys
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import ConvState, DGDALayer
try:
    from tests.test_dgda import sequential_dgda_reference
except ImportError:
    from test_dgda import sequential_dgda_reference


def chunkwise_exact_solve(
    layer: DGDALayer,
    x: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    conv_state: Optional[Union[ConvState, Tuple[torch.Tensor, ...]]] = None,
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, D = x.shape
    H, d_k, d_v = layer.n_heads, layer.d_head, layer.d_head

    cs_q, cs_k, cs_v = layer._unpack_conv_state(conv_state, B, x.device, x.dtype)
    q, _ = layer._apply_conv(layer.q_proj(x), layer.conv_q, cs_q)
    k, _ = layer._apply_conv(layer.k_proj(x), layer.conv_k, cs_k)
    v, _ = layer._apply_conv(layer.v_proj(x), layer.conv_v, cs_v)

    b = torch.sigmoid(layer.gate_erase(x))
    w = torch.sigmoid(layer.gate_write(x))
    log_alpha = -F.softplus(layer.gate_alpha(x))

    q = q.view(B, L, H, d_k).transpose(1, 2)
    k = k.view(B, L, H, d_k).transpose(1, 2)
    v = v.view(B, L, H, d_v).transpose(1, 2)
    b = b.view(B, L, H, d_k).transpose(1, 2)
    w = w.view(B, L, H, d_v).transpose(1, 2)
    log_alpha = log_alpha.view(B, L, H, d_k).transpose(1, 2)
    k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + layer.eps)

    if state is None:
        curr_S = torch.zeros(B, H, d_k, d_v, dtype=x.dtype, device=x.device)
    else:
        curr_S = state.clone()

    num_full_chunks = L // chunk_size
    remainder = L % chunk_size
    out_chunks = []

    def _solve_chunk(qc, kc, vc, bc, wc, lac, S0):
        T = qc.shape[2]
        cum_log_alpha = torch.cumsum(lac, dim=-2)
        diff = cum_log_alpha.unsqueeze(3) - cum_log_alpha.unsqueeze(2)
        log_decay = torch.clamp(diff, max=0.0)
        decay = torch.exp(log_decay)

        bk = (bc * kc).unsqueeze(3)
        ks = kc.unsqueeze(2)
        L_mat = torch.tril((bk * decay * ks).sum(dim=-1), diagonal=-1)

        Lambda = torch.exp(cum_log_alpha)
        beta_hat = (bc * kc) * Lambda
        v_eff = (wc * vc) - torch.matmul(beta_hat, S0)

        eye_T = torch.eye(T, dtype=qc.dtype, device=qc.device).view(1, 1, T, T)
        I_plus_L = eye_T + L_mat
        u = torch.linalg.solve_triangular(I_plus_L, v_eff, upper=False)

        cum_log_alpha_end = cum_log_alpha[:, :, -1:, :]
        decay_to_end = torch.exp(cum_log_alpha_end - cum_log_alpha)
        k_decayed = kc * decay_to_end
        S_chunk = torch.matmul(k_decayed.transpose(-1, -2), u)
        decay_S0 = torch.exp(cum_log_alpha_end).transpose(-1, -2)
        S_final = decay_S0 * S0 + S_chunk

        O_inter = torch.matmul(qc * Lambda, S0)
        qs = qc.unsqueeze(3)
        A = torch.tril((qs * decay * ks).sum(dim=-1), diagonal=0)
        O_intra = torch.matmul(A, u)
        return O_inter + O_intra, S_final

    for i in range(num_full_chunks):
        s, e = i * chunk_size, (i + 1) * chunk_size
        out_c, curr_S = _solve_chunk(
            q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], b[:, :, s:e], w[:, :, s:e], log_alpha[:, :, s:e], curr_S
        )
        out_chunks.append(out_c)

    if remainder > 0:
        s = num_full_chunks * chunk_size
        out_c, curr_S = _solve_chunk(
            q[:, :, s:], k[:, :, s:], v[:, :, s:], b[:, :, s:], w[:, :, s:], log_alpha[:, :, s:], curr_S
        )
        out_chunks.append(out_c)

    out = torch.cat(out_chunks, dim=2).transpose(1, 2).contiguous().view(B, L, H * d_v)
    return layer.o_proj(out), curr_S


@pytest.mark.parametrize("seq_len", [16, 32, 64, 128, 512, 1024])
def test_random_inputs_large_seq_lens(seq_len):
    torch.manual_seed(42 + seq_len)
    config = MabaSparseConfig(dim=64, n_heads=2, d_head=32, kernel_size=4, chunk_size=16)
    layer = DGDALayer(config).eval()

    x = torch.randn(1, seq_len, config.dim)
    out_seq, state_seq, _ = sequential_dgda_reference(x, layer)
    out_chunk, state_chunk, _ = layer(x, chunk_size=16)

    diff_out = (out_chunk - out_seq).abs().max().item()
    diff_state = (state_chunk - state_seq).abs().max().item()

    assert diff_out < 1e-4, f"Output diff {diff_out:.6e} >= 1e-4 at L={seq_len}"
    assert diff_state < 1e-4, f"State diff {diff_state:.6e} >= 1e-4 at L={seq_len}"


@pytest.mark.parametrize("decay_mode", ["near_zero", "near_one"])
def test_decay_boundary_stability(decay_mode):
    torch.manual_seed(123)
    config = MabaSparseConfig(dim=64, n_heads=2, d_head=32, kernel_size=4, chunk_size=16)
    layer = DGDALayer(config).eval()

    with torch.no_grad():
        if decay_mode == "near_zero":
            layer.gate_alpha.weight.fill_(10.0)
        else:
            layer.gate_alpha.weight.fill_(-10.0)

    x = torch.randn(2, 64, config.dim)
    out_seq, state_seq, _ = sequential_dgda_reference(x, layer)
    out_chunk, state_chunk, _ = layer(x, chunk_size=16)

    diff_out = (out_chunk - out_seq).abs().max().item()
    diff_state = (state_chunk - state_seq).abs().max().item()

    assert diff_out < 1e-4, f"Output diff {diff_out:.6e} >= 1e-4 in mode {decay_mode}"
    assert diff_state < 1e-4, f"State diff {diff_state:.6e} >= 1e-4 in mode {decay_mode}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_precision_chunk_vs_seq(dtype):
    torch.manual_seed(456)
    config = MabaSparseConfig(dim=64, n_heads=2, d_head=32, kernel_size=4, chunk_size=16)
    layer = DGDALayer(config).to(dtype).eval()

    x = torch.randn(2, 32, config.dim, dtype=dtype)
    out_seq, state_seq, _ = sequential_dgda_reference(x, layer)
    out_chunk, state_chunk, _ = layer(x, chunk_size=16)

    diff_out = (out_chunk.float() - out_seq.float()).abs().max().item()
    diff_state = (state_chunk.float() - state_seq.float()).abs().max().item()

    if dtype == torch.bfloat16:
        assert diff_out < 1e-3, f"BF16 output diff {diff_out:.6e} exceeds 1e-3"
        assert diff_state < 2e-3, f"BF16 state diff {diff_state:.6e} exceeds 2e-3"
    else:
        assert diff_out < 1e-4, f"{dtype} output diff {diff_out:.6e} >= 1e-4"
        assert diff_state < 1e-4, f"{dtype} state diff {diff_state:.6e} >= 1e-4"


def test_extreme_inputs_exact_solve_vs_neumann_comparison():
    torch.manual_seed(789)
    config = MabaSparseConfig(dim=64, n_heads=2, d_head=32, kernel_size=4, chunk_size=16)
    layer = DGDALayer(config).eval()

    x = torch.full((1, 32, config.dim), 100.0)

    out_seq, state_seq, _ = sequential_dgda_reference(x, layer)
    out_chunk, state_chunk, _ = layer(x, chunk_size=16, inversion_method="neumann")

    diff_out_neumann = (out_chunk - out_seq).abs().max().item()
    diff_state_neumann = (state_chunk - state_seq).abs().max().item()

    print(f"\n[Neumann-3 Extreme +/-100] Out diff = {diff_out_neumann:.4f}, State diff = {diff_state_neumann:.4f}")
    assert diff_out_neumann > 1.0, f"Expected Neumann series to diverge under extreme input, got diff {diff_out_neumann}"

    out_exact, state_exact = chunkwise_exact_solve(layer, x, chunk_size=16)
    diff_out_exact = (out_exact - out_seq).abs().max().item()
    diff_state_exact = (state_exact - state_seq).abs().max().item()
    print(f"[Exact Solve Extreme +/-100] Out diff = {diff_out_exact:.6e}, State diff = {diff_state_exact:.6e}")
    assert diff_out_exact < 1e-4, f"Exact solve output diff {diff_out_exact:.6e} >= 1e-4"
    assert diff_state_exact < 1e-4, f"Exact solve state diff {diff_state_exact:.6e} >= 1e-4"


def test_101m_scale_extreme_inputs_exact_solve():
    torch.manual_seed(999)
    config = MabaSparseConfig(dim=640, n_heads=10, d_head=64, kernel_size=4, chunk_size=16)
    layer = DGDALayer(config).eval()

    x = torch.full((1, 32, config.dim), 100.0)

    out_seq, state_seq, _ = sequential_dgda_reference(x, layer)
    out_chunk, state_chunk, _ = layer(x, chunk_size=16, inversion_method="neumann")

    diff_out = (out_chunk - out_seq).abs().max().item()
    diff_state = (state_chunk - state_seq).abs().max().item()

    print(f"\n[101M Neumann-3 Explosion] Out diff = {diff_out:.2f}, State diff = {diff_state:.2f}")
    assert diff_state > 1.0, f"Expected state diff divergence at 101M scale, got {diff_state}"

    out_exact, state_exact = chunkwise_exact_solve(layer, x, chunk_size=16)
    diff_exact_state = (state_exact - state_seq).abs().max().item()
    diff_exact_out = (out_exact - out_seq).abs().max().item()
    print(f"[101M Exact Solve] Out diff = {diff_exact_out:.6e}, State diff = {diff_exact_state:.6e}")
    assert diff_exact_state < 1e-4, f"101M exact solve state diff {diff_exact_state:.6e} >= 1e-4"
    assert diff_exact_out < 1e-4, f"101M exact solve out diff {diff_exact_out:.6e} >= 1e-4"


def run_stress_suite():
    print("=" * 80)
    print("EMPIRICAL STRESS TEST SUITE: DGDALayer NUMERICAL ACCURACY & EQUIVALENCE")
    print("=" * 80)

    config_tiny = MabaSparseConfig(dim=64, n_heads=2, d_head=32, kernel_size=4, chunk_size=16)
    layer_tiny = DGDALayer(config_tiny).eval()

    rows = []

    def record(category, test_name, out_diff, state_diff, threshold=1e-4):
        passed = (out_diff < threshold) and (state_diff < threshold)
        rows.append({
            "category": category,
            "name": test_name,
            "out_diff": out_diff,
            "state_diff": state_diff,
            "threshold": threshold,
            "status": "PASS" if passed else "FAIL",
        })

    for L in [16, 32, 64, 128, 512, 1024]:
        torch.manual_seed(100 + L)
        x = torch.randn(1, L, config_tiny.dim)
        out_s, st_s, _ = sequential_dgda_reference(x, layer_tiny)
        out_c, st_c, _ = layer_tiny(x, chunk_size=16)
        record("Random Inputs", f"L={L}", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

    for mag in [10.0, 50.0, 100.0]:
        x_pos = torch.full((1, 32, config_tiny.dim), mag)
        out_s, st_s, _ = sequential_dgda_reference(x_pos, layer_tiny)
        out_c, st_c, _ = layer_tiny(x_pos, chunk_size=16)
        record("Extreme Inputs", f"Uniform +{mag}", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

        x_neg = torch.full((1, 32, config_tiny.dim), -mag)
        out_s, st_s, _ = sequential_dgda_reference(x_neg, layer_tiny)
        out_c, st_c, _ = layer_tiny(x_neg, chunk_size=16)
        record("Extreme Inputs", f"Uniform -{mag}", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

        x_mix = torch.randn(1, 32, config_tiny.dim) * mag
        out_s, st_s, _ = sequential_dgda_reference(x_mix, layer_tiny)
        out_c, st_c, _ = layer_tiny(x_mix, chunk_size=16)
        record("Extreme Inputs", f"Gaussian x{mag}", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        l_dt = DGDALayer(config_tiny).to(dt).eval()
        x = torch.randn(2, 32, config_tiny.dim, dtype=dt)
        out_s, st_s, _ = sequential_dgda_reference(x, l_dt)
        out_c, st_c, _ = l_dt(x, chunk_size=16)
        record(
            "Precision",
            f"{dt}",
            (out_c.float() - out_s.float()).abs().max().item(),
            (st_c.float() - st_s.float()).abs().max().item(),
            threshold=1e-4 if dt != torch.bfloat16 else 2e-3,
        )

    with torch.no_grad():
        layer_tiny.gate_alpha.weight.fill_(10.0)
        x = torch.randn(2, 32, config_tiny.dim)
        out_s, st_s, _ = sequential_dgda_reference(x, layer_tiny)
        out_c, st_c, _ = layer_tiny(x, chunk_size=16)
        record("Decay Boundary", "Near Zero (alpha->0)", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

        layer_tiny.gate_alpha.weight.fill_(-10.0)
        out_s, st_s, _ = sequential_dgda_reference(x, layer_tiny)
        out_c, st_c, _ = layer_tiny(x, chunk_size=16)
        record("Decay Boundary", "Near One (alpha->1)", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

    config_101m = MabaSparseConfig(dim=640, n_heads=10, d_head=64, kernel_size=4, chunk_size=16)
    layer_101m = DGDALayer(config_101m).eval()
    x_101m_rand = torch.randn(1, 128, config_101m.dim)
    out_s, st_s, _ = sequential_dgda_reference(x_101m_rand, layer_101m)
    out_c, st_c, _ = layer_101m(x_101m_rand, chunk_size=16)
    record("101M Scale", "Random L=128", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

    x_101m_ext = torch.full((1, 32, config_101m.dim), 100.0)
    out_s, st_s, _ = sequential_dgda_reference(x_101m_ext, layer_101m)
    out_c, st_c, _ = layer_101m(x_101m_ext, chunk_size=16)
    record("101M Scale", "Neumann-3 Extreme +100", (out_c - out_s).abs().max().item(), (st_c - st_s).abs().max().item())

    out_ex, st_ex = chunkwise_exact_solve(layer_101m, x_101m_ext, chunk_size=16)
    record("101M Scale (Fix)", "Exact Solve Extreme +100", (out_ex - out_s).abs().max().item(), (st_ex - st_s).abs().max().item())

    print(f"\n{'CATEGORY':<18} | {'TEST CASE':<25} | {'OUT DIFF':<12} | {'STATE DIFF':<12} | {'THRESHOLD':<10} | {'STATUS'}")
    print("-" * 92)
    for r in rows:
        print(f"{r['category']:<18} | {r['name']:<25} | {r['out_diff']:<12.4e} | {r['state_diff']:<12.4e} | {r['threshold']:<10.1e} | {r['status']}")

    failures = [r for r in rows if r["status"] == "FAIL"]
    print("\n" + "=" * 92)
    print(f"TOTAL TESTS: {len(rows)} | PASSED: {len(rows) - len(failures)} | FAILED: {len(failures)}")
    print("=" * 92)
    return len(failures)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        num_fail = run_stress_suite()
        sys.exit(num_fail)
    else:
        pytest.main(["-v", "-s", __file__])
