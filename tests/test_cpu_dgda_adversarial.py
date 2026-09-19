
import os
import sys
from typing import Tuple

import pytest
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


def make_tensors(
    B: int = 2,
    H: int = 4,
    L: int = 64,
    dk: int = 32,
    dv: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(B, H, L, dk, dtype=dtype, device=device)
    k = torch.randn(B, H, L, dk, dtype=dtype, device=device)
    k = normalize_keys(k)
    v = torch.randn(B, H, L, dv, dtype=dtype, device=device)
    alpha = torch.rand(B, H, L, dk, dtype=dtype, device=device) * 0.9 + 0.05
    b = torch.rand(B, H, L, dk, dtype=dtype, device=device)
    w = torch.rand(B, H, L, dv, dtype=dtype, device=device)
    return q, k, v, alpha, b, w


class TestExtremeDecayRegimes:

    @pytest.mark.parametrize("alpha_val", [0.0, 1e-7, 1e-20, 1e-35])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_instant_forgetting_alpha_zero(self, alpha_val: float, dtype: torch.dtype):
        B, H, L, dk, dv = 2, 2, 48, 16, 16
        q, k, v, _, b, w = make_tensors(B, H, L, dk, dv, dtype=dtype, seed=101)
        alpha = torch.full_like(q, alpha_val)

        init_state = torch.randn(B, H, dk, dv, dtype=dtype)

        out_cpu, state_cpu = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )

        assert torch.isfinite(out_cpu).all(), f"out_cpu contains non-finite values for alpha={alpha_val}, dtype={dtype}"
        assert torch.isfinite(state_cpu).all(), f"state_cpu contains non-finite values for alpha={alpha_val}, dtype={dtype}"
        assert not torch.isnan(out_cpu).any(), f"NaN in out_cpu for alpha={alpha_val}"
        assert not torch.isnan(state_cpu).any(), f"NaN in state_cpu for alpha={alpha_val}"

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        alpha_f = alpha.float()
        b_f = b.float()
        w_f = w.float()
        init_state_f = init_state.float()

        out_ref, state_ref = ref_dgda_prefill(
            q_f, k_f, v_f, alpha_f, b_f, w_f, initial_state=init_state_f
        )

        tol = 2e-4 if dtype == torch.float32 else (5e-3 if dtype == torch.float16 else 2e-2)
        diff_out = (out_cpu.float() - out_ref).abs().max().item()
        diff_state = (state_cpu.float() - state_ref).abs().max().item()

        assert diff_out < tol, f"Prefill output diff {diff_out:.6e} exceeded tol {tol} for alpha={alpha_val}"
        assert diff_state < tol, f"Prefill state diff {diff_state:.6e} exceeded tol {tol} for alpha={alpha_val}"

    @pytest.mark.parametrize("alpha_val", [1.0, 1.0 - 1e-7])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_no_decay_alpha_one(self, alpha_val: float, dtype: torch.dtype):
        B, H, L, dk, dv = 2, 2, 64, 16, 16
        q, k, v, _, b, w = make_tensors(B, H, L, dk, dv, dtype=dtype, seed=102)
        alpha = torch.full_like(q, alpha_val)

        out_cpu, state_cpu = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        assert torch.isfinite(out_cpu).all(), f"out_cpu non-finite for alpha={alpha_val}"
        assert torch.isfinite(state_cpu).all(), f"state_cpu non-finite for alpha={alpha_val}"

        out_ref, state_ref = ref_dgda_prefill(
            q.float(), k.float(), v.float(), alpha.float(), b.float(), w.float()
        )

        tol = 1e-4 if dtype == torch.float32 else (5e-3 if dtype == torch.float16 else 5e-2)
        diff_out = (out_cpu.float() - out_ref).abs().max().item()
        diff_state = (state_cpu.float() - state_ref).abs().max().item()

        assert diff_out < tol, f"Prefill output diff {diff_out:.6e} exceeded tol {tol}"
        assert diff_state < tol, f"Prefill state diff {diff_state:.6e} exceeded tol {tol}"

    def test_subnormal_precision_safety(self):
        B, H, L, dk, dv = 1, 2, 32, 16, 16
        q, k, v, _, b, w = make_tensors(B, H, L, dk, dv, dtype=torch.float32, seed=103)
        subnormal_val = 1.4e-45
        alpha = torch.full_like(q, subnormal_val)

        out_cpu, state_cpu = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

    def test_step_decode_extreme_decay(self):
        B, H, dk, dv = 2, 4, 32, 32
        for alpha_val in [0.0, 1e-6, 1.0]:
            torch.manual_seed(104)
            q = torch.randn(B, H, dk)
            k = normalize_keys(torch.randn(B, H, dk))
            v = torch.randn(B, H, dv)
            alpha = torch.full_like(q, alpha_val)
            b = torch.rand(B, H, dk)
            w = torch.rand(B, H, dv)
            state = torch.randn(B, H, dk, dv)

            out_cpu, new_state_cpu = cpu_dgda_step(q, k, v, alpha, b, w, state)
            out_ref, new_state_ref = ref_dgda_step(q, k, v, alpha, b, w, state)

            assert torch.isfinite(out_cpu).all()
            assert torch.isfinite(new_state_cpu).all()

            diff_out = (out_cpu - out_ref).abs().max().item()
            diff_state = (new_state_cpu - new_state_ref).abs().max().item()
            assert diff_out < 1e-6, f"Step out diff {diff_out:.6e} for alpha={alpha_val}"
            assert diff_state < 1e-6, f"Step state diff {diff_state:.6e} for alpha={alpha_val}"

    def test_instant_forgetting_wipes_initial_state(self):
        B, H, L, dk, dv = 1, 1, 16, 8, 8
        q, k, v, _, b, w = make_tensors(B, H, L, dk, dv, dtype=torch.float32, seed=105)
        alpha = torch.zeros_like(q)

        state_zeros = torch.zeros(B, H, dk, dv)
        state_random = torch.randn(B, H, dk, dv) * 500.0

        out1, final_s1 = cpu_dgda_prefill(q, k, v, alpha, b, w, initial_state=state_zeros)
        out2, final_s2 = cpu_dgda_prefill(q, k, v, alpha, b, w, initial_state=state_random)

        diff_out = (out1 - out2).abs().max().item()
        diff_state = (final_s1 - final_s2).abs().max().item()
        assert diff_out < 1e-6, f"Initial state leaked into output under alpha=0: diff={diff_out:.6e}"
        assert diff_state < 1e-6, f"Initial state leaked into final state under alpha=0: diff={diff_state:.6e}"


class TestExtremeWriteEraseRegimes:

    @pytest.mark.parametrize("L", [16, 32, 64])
    def test_pure_erase_regime(self, L: int):
        B, H, dk, dv = 2, 2, 16, 16
        q, k, v, alpha, _, _ = make_tensors(B, H, L, dk, dv, seed=201 + L)
        b = torch.ones_like(k)
        w = torch.zeros_like(v)
        init_state = torch.randn(B, H, dk, dv)

        out_cpu, state_cpu = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )
        out_ref, state_ref = ref_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        diff_out = (out_cpu - out_ref).abs().max().item()
        diff_state = (state_cpu - state_ref).abs().max().item()
        assert diff_out < 1e-5, f"Pure erase out diff {diff_out:.6e} at L={L}"
        assert diff_state < 1e-5, f"Pure erase state diff {diff_state:.6e} at L={L}"

        assert torch.linalg.norm(state_cpu) < torch.linalg.norm(init_state) + 1.0

    @pytest.mark.parametrize("L", [16, 32, 64])
    def test_pure_write_regime(self, L: int):
        B, H, dk, dv = 2, 2, 16, 16
        q, k, v, alpha, _, _ = make_tensors(B, H, L, dk, dv, seed=202 + L)
        b = torch.zeros_like(k)
        w = torch.ones_like(v)
        init_state = torch.randn(B, H, dk, dv)

        out_cpu, state_cpu = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )
        out_ref, state_ref = ref_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        diff_out = (out_cpu - out_ref).abs().max().item()
        diff_state = (state_cpu - state_ref).abs().max().item()
        assert diff_out < 1e-5, f"Pure write out diff {diff_out:.6e} at L={L}"
        assert diff_state < 1e-5, f"Pure write state diff {diff_state:.6e} at L={L}"

    def test_zero_write_zero_erase_frozen_state(self):
        B, H, L, dk, dv = 2, 2, 32, 16, 16
        q, k, v, alpha, _, _ = make_tensors(B, H, L, dk, dv, seed=203)
        b = torch.zeros_like(k)
        w = torch.zeros_like(v)
        init_state = torch.randn(B, H, dk, dv)

        out_cpu, state_cpu = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )
        out_ref, state_ref = ref_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        diff_out = (out_cpu - out_ref).abs().max().item()
        diff_state = (state_cpu - state_ref).abs().max().item()
        assert diff_out < 1e-6, f"Frozen state out diff {diff_out:.6e}"
        assert diff_state < 1e-6, f"Frozen state state diff {diff_state:.6e}"

    def test_full_write_full_erase(self):
        B, H, L, dk, dv = 2, 2, 48, 16, 16
        q, k, v, alpha, _, _ = make_tensors(B, H, L, dk, dv, seed=204)
        b = torch.ones_like(k)
        w = torch.ones_like(v)

        out_cpu, state_cpu = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
        out_ref, state_ref = ref_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        diff_out = (out_cpu - out_ref).abs().max().item()
        diff_state = (state_cpu - state_ref).abs().max().item()
        assert diff_out < 1e-5, f"Full write/erase out diff {diff_out:.6e}"
        assert diff_state < 1e-5, f"Full write/erase state diff {diff_state:.6e}"


class TestSpectralNormStress:

    def _build_ill_conditioned_inputs(
        self, B: int, H: int, L: int, dk: int, dv: int, colinearity: float = 1.0
    ):
        torch.manual_seed(301)
        base_key = torch.randn(1, 1, 1, dk)
        base_key = base_key / torch.linalg.norm(base_key, dim=-1, keepdim=True)

        noise = torch.randn(B, H, L, dk) * (1.0 - colinearity)
        k = base_key.expand(B, H, L, dk) + noise
        k = normalize_keys(k)

        q = torch.randn(B, H, L, dk)
        v = torch.randn(B, H, L, dv)
        alpha = torch.full((B, H, L, dk), 0.9999)
        b = torch.ones(B, H, L, dk)
        w = torch.ones(B, H, L, dv)
        return q, k, v, alpha, b, w

    def test_spectral_norm_greater_than_one(self):
        B, H, L, dk, dv = 1, 1, 16, 16, 16
        q, k, v, alpha, b, w = self._build_ill_conditioned_inputs(B, H, L, dk, dv, colinearity=1.0)

        lac = torch.log(alpha)
        cla = torch.cumsum(lac, dim=-2)
        diff = cla.unsqueeze(-2) - cla.unsqueeze(-3)
        dec = torch.exp(torch.clamp(diff, max=0.0))

        bk = (b * k).unsqueeze(-2)
        ks = k.unsqueeze(-3)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

        singular_vals = torch.linalg.svdvals(l_mat[0, 0])
        spectral_norm = singular_vals[0].item()

        assert spectral_norm >= 1.0, f"Expected ||L||_2 >= 1.0, got {spectral_norm:.4f}"
        assert spectral_norm > 5.0, f"Expected strong ill-conditioning, got {spectral_norm:.4f}"

    @pytest.mark.parametrize("chunk_size", [16])
    @pytest.mark.parametrize("L", [16, 32, 64])
    def test_adaptive_fallback_triggers_and_matches_reference(self, chunk_size: int, L: int):
        B, H, dk, dv = 2, 2, 32, 32
        q, k, v, alpha, b, w = self._build_ill_conditioned_inputs(B, H, L, dk, dv, colinearity=1.0)

        out_neumann, state_neumann = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=chunk_size, inversion_method="neumann"
        )

        out_adaptive, state_adaptive = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=chunk_size, inversion_method="adaptive", adaptive_tol=7e-5
        )

        out_exact, state_exact = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=chunk_size, inversion_method="exact"
        )

        out_ref, state_ref = ref_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=chunk_size
        )

        assert torch.isfinite(out_adaptive).all(), "out_adaptive has non-finite entries"
        assert torch.isfinite(state_adaptive).all(), "state_adaptive has non-finite entries"

        neumann_diff = (out_neumann - out_ref).abs().max().item()
        assert neumann_diff > 1e-3, (
            f"Expected Neumann series to exhibit error > 1e-3 under ||L||_2 >= 1, "
            f"got diff={neumann_diff:.6e}"
        )

        diff_adaptive_exact = (out_adaptive - out_exact).abs().max().item()
        assert diff_adaptive_exact < 1e-6, (
            f"Adaptive did not match exact triangular solve: diff={diff_adaptive_exact:.6e}"
        )

        diff_adaptive_ref = (out_adaptive - out_ref).abs().max().item()
        diff_state_ref = (state_adaptive - state_ref).abs().max().item()

        assert diff_adaptive_ref < 1e-4, (
            f"Adaptive output diff vs reference {diff_adaptive_ref:.6e} exceeded 1e-4 tolerance!"
        )
        assert diff_state_ref < 1e-4, (
            f"Adaptive state diff vs reference {diff_state_ref:.6e} exceeded 1e-4 tolerance!"
        )

    def test_spectral_norm_residual_threshold_trigger(self):
        B, H, L, dk, dv = 1, 1, 16, 16, 16
        q, k, v, alpha, b, w = self._build_ill_conditioned_inputs(B, H, L, dk, dv, colinearity=0.99)

        lac = torch.log(alpha)
        cla = torch.cumsum(lac, dim=-2)
        diff = cla.unsqueeze(-2) - cla.unsqueeze(-3)
        dec = torch.exp(torch.clamp(diff, max=0.0))

        bk = (b * k).unsqueeze(-2)
        ks = k.unsqueeze(-3)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

        eye = torch.eye(16).view(1, 1, 16, 16)
        l2 = torch.matmul(l_mat, l_mat)
        l3 = torch.matmul(l2, l_mat)
        inv_l = eye - l_mat + l2 - l3

        wv = w * v
        u_neumann = torch.matmul(inv_l, wv)

        r = wv - u_neumann - torch.matmul(l_mat, u_neumann)
        max_residual = r.abs().max().item()

        adaptive_tol = 7e-5
        assert max_residual > adaptive_tol, (
            f"Expected residual {max_residual:.6e} to exceed adaptive_tol {adaptive_tol}"
        )


class TestOpenMPConcurrency:

    @pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
    def test_deterministic_parity_across_threads(self, num_threads: int):
        B, H, L, dk, dv = 4, 8, 64, 32, 32
        q, k, v, alpha, b, w = make_tensors(B, H, L, dk, dv, seed=401)

        with cpu_threads(1):
            out_base, state_base = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        with cpu_threads(num_threads):
            out_mt, state_mt = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        diff_out = (out_mt - out_base).abs().max().item()
        diff_state = (state_mt - state_base).abs().max().item()

        assert diff_out <= 1e-7, (
            f"Numerical nondeterminism detected at {num_threads} threads: diff_out={diff_out:.6e}"
        )
        assert diff_state <= 1e-7, (
            f"State nondeterminism detected at {num_threads} threads: diff_state={diff_state:.6e}"
        )

    def test_cpu_threads_context_manager_restoration(self):
        initial_threads = get_cpu_num_threads()
        target_threads = 3 if initial_threads != 3 else 2

        with cpu_threads(target_threads):
            assert get_cpu_num_threads() == target_threads

        assert get_cpu_num_threads() == initial_threads

        try:
            with cpu_threads(target_threads):
                assert get_cpu_num_threads() == target_threads
                raise RuntimeError("Intentional error for test")
        except RuntimeError:
            pass

        assert get_cpu_num_threads() == initial_threads

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_concurrency_stress_large_workload(self, dtype: torch.dtype):
        B, H, L, dk, dv = 8, 8, 256, 32, 32
        q, k, v, alpha, b, w = make_tensors(B, H, L, dk, dv, dtype=dtype, seed=402)

        with cpu_threads(1):
            out_1, state_1 = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        with cpu_threads(4):
            out_4, state_4 = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        assert torch.isfinite(out_4).all()
        assert torch.isfinite(state_4).all()

        tol = 1e-6 if dtype == torch.float32 else 1e-3
        diff = (out_4.float() - out_1.float()).abs().max().item()
        assert diff <= tol, f"Large workload thread discrepancy: {diff:.6e}"

    def test_step_decode_thread_determinism(self):
        B, H, dk, dv = 8, 8, 64, 64
        torch.manual_seed(403)
        q = torch.randn(B, H, dk)
        k = normalize_keys(torch.randn(B, H, dk))
        v = torch.randn(B, H, dv)
        alpha = torch.rand(B, H, dk) * 0.9 + 0.05
        b = torch.rand(B, H, dk)
        w = torch.rand(B, H, dv)
        state = torch.randn(B, H, dk, dv)

        with cpu_threads(1):
            out_1, state_1 = cpu_dgda_step(q, k, v, alpha, b, w, state)

        with cpu_threads(4):
            out_4, state_4 = cpu_dgda_step(q, k, v, alpha, b, w, state)

        diff_out = (out_4 - out_1).abs().max().item()
        diff_state = (state_4 - state_1).abs().max().item()
        assert diff_out == 0.0 or diff_out < 1e-7
        assert diff_state == 0.0 or diff_state < 1e-7


class TestBoundaryAndRemainderStress:

    @pytest.mark.parametrize("L", [1, 7, 15, 17, 25, 33, 49])
    @pytest.mark.parametrize("alpha_val", [0.0, 1.0])
    def test_non_multiple_chunk_sizes(self, L: int, alpha_val: float):
        B, H, dk, dv = 2, 2, 16, 16
        q, k, v, _, b, w = make_tensors(B, H, L, dk, dv, seed=501 + L)
        alpha = torch.full_like(q, alpha_val)
        init_state = torch.randn(B, H, dk, dv)

        out_cpu, state_cpu = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )
        out_ref, state_ref = ref_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        diff_out = (out_cpu - out_ref).abs().max().item()
        diff_state = (state_cpu - state_ref).abs().max().item()
        assert diff_out < 1e-4, f"Remainder diff_out={diff_out:.6e} at L={L}, alpha={alpha_val}"
        assert diff_state < 1e-4, f"Remainder diff_state={diff_state:.6e} at L={L}, alpha={alpha_val}"

    def test_single_token_prefill_matches_step(self):
        B, H, dk, dv = 2, 4, 32, 32
        torch.manual_seed(502)
        q = torch.randn(B, H, 1, dk)
        k = normalize_keys(torch.randn(B, H, 1, dk))
        v = torch.randn(B, H, 1, dv)
        alpha = torch.rand(B, H, 1, dk) * 0.9 + 0.05
        b = torch.rand(B, H, 1, dk)
        w = torch.rand(B, H, 1, dv)
        init_state = torch.randn(B, H, dk, dv)

        out_prefill, state_prefill = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=init_state
        )
        out_step, state_step = cpu_dgda_step(
            q.squeeze(2), k.squeeze(2), v.squeeze(2),
            alpha.squeeze(2), b.squeeze(2), w.squeeze(2),
            state=init_state,
        )

        diff_out = (out_prefill.squeeze(2) - out_step).abs().max().item()
        diff_state = (state_prefill - state_step).abs().max().item()
        assert diff_out < 1e-5, f"L=1 prefill vs step out diff: {diff_out:.6e}"
        assert diff_state < 1e-5, f"L=1 prefill vs step state diff: {diff_state:.6e}"


class TestAdversarialAutogradGradients:

    @pytest.mark.parametrize("alpha_val", [0.0, 1.0])
    def test_backward_pass_extreme_decay(self, alpha_val: float):
        B, H, L, dk, dv = 1, 2, 32, 16, 16
        q, k, v, _, b, w = make_tensors(B, H, L, dk, dv, seed=601)
        alpha = torch.full_like(q, alpha_val)

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        b.requires_grad_(True)
        w.requires_grad_(True)

        out, final_state = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
        loss = out.sum() + final_state.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("b", b), ("w", w)]:
            assert t.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(t.grad).all(), f"Gradient for {name} has NaN/Inf"
            assert not (t.grad == 0).all(), f"Gradient for {name} is all zeros"

    def test_backward_pass_spectral_ill_conditioned(self):
        B, H, L, dk, dv = 1, 1, 16, 16, 16
        tester = TestSpectralNormStress()
        q, k, v, alpha, b, w = tester._build_ill_conditioned_inputs(B, H, L, dk, dv, colinearity=1.0)

        q.requires_grad_(True)
        v.requires_grad_(True)

        out, final_state = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16, inversion_method="adaptive")
        loss = out.sum() + final_state.sum()
        loss.backward()

        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()


class TestExtremeMagnitudesAndGates:

    def test_binary_bernoulli_write_erase_gates(self):
        B, H, L, dk, dv = 2, 2, 48, 16, 16
        torch.manual_seed(701)
        q = torch.randn(B, H, L, dk)
        k = normalize_keys(torch.randn(B, H, L, dk))
        v = torch.randn(B, H, L, dv)
        alpha = torch.rand(B, H, L, dk) * 0.9 + 0.05
        b = torch.bernoulli(torch.full_like(k, 0.5))
        w = torch.bernoulli(torch.full_like(v, 0.5))

        out_cpu, state_cpu = cpu_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
        out_ref, state_ref = ref_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        diff_out = (out_cpu - out_ref).abs().max().item()
        diff_state = (state_cpu - state_ref).abs().max().item()
        assert diff_out < 1e-4, f"Bernoulli gates diff_out={diff_out:.6e}"
        assert diff_state < 1e-4, f"Bernoulli gates diff_state={diff_state:.6e}"

    def test_extreme_initial_state_magnitude(self):
        B, H, L, dk, dv = 1, 2, 32, 16, 16
        q, k, v, alpha, b, w = make_tensors(B, H, L, dk, dv, seed=702)
        huge_init_state = torch.randn(B, H, dk, dv) * 10000.0

        out_cpu, state_cpu = cpu_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=huge_init_state
        )
        out_ref, state_ref = ref_dgda_prefill(
            q, k, v, alpha, b, w, chunk_size=16, initial_state=huge_init_state
        )

        assert torch.isfinite(out_cpu).all()
        assert torch.isfinite(state_cpu).all()

        rel_diff = ((out_cpu - out_ref).abs() / (out_ref.abs() + 1e-6)).max().item()
        assert rel_diff < 1e-4, f"Huge initial state rel_diff={rel_diff:.6e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

