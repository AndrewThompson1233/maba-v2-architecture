
import pytest
import torch

from maba_sparse.kernels.common import (
    ref_dgda_prefill,
    normalize_keys,
)
from maba_sparse.kernels.dispatcher import (
    is_cuda_sm75_available,
    is_triton_available,
)
from maba_sparse.kernels.xla_dgda import (
    xla_dgda_prefill,
)

CUDA_ONLINE = torch.cuda.is_available() and is_cuda_sm75_available() and is_triton_available()
DEVICE = "cuda" if CUDA_ONLINE else "cpu"

if CUDA_ONLINE:
    from maba_sparse.kernels.triton_dgda import (
        triton_dgda_prefill,
    )


class TestAlgebraicInverseProperties:

    def test_l_matrix_strict_nilpotency_16(self):
        torch.manual_seed(1234)
        for trial in range(20):
            A = torch.randn(16, 16, dtype=torch.float64)
            L = torch.tril(A, diagonal=-1)

            curr = L.clone()
            for power in range(2, 17):
                curr = curr @ L
                if power < 16:
                    assert not torch.all(curr == 0.0), f"L^{power} is prematurely zero"
                else:
                    max_val = curr.abs().max().item()
                    assert max_val == 0.0, f"L^16 is non-zero: max val = {max_val}"

    def test_horner_p15_exact_inverse(self):
        torch.manual_seed(5678)
        for trial in range(20):
            L = torch.tril(torch.randn(16, 16, dtype=torch.float64) * 2.0, diagonal=-1)
            I = torch.eye(16, dtype=torch.float64)
            M = I + L

            inv_l = I - L
            for _ in range(14):
                inv_l = I - L @ inv_l

            prod = M @ inv_l
            diff = (prod - I).abs().max().item()
            assert diff < 1e-11, f"Horner P15 inverse failed: max residual = {diff}"

    def test_forward_substitution_vs_horner_vs_solve_triangular(self):
        torch.manual_seed(9012)
        for trial in range(20):
            L = torch.tril(torch.rand(16, 16, dtype=torch.float32) * 0.5, diagonal=-1)
            I = torch.eye(16, dtype=torch.float32)
            M = I + L
            ve = torch.randn(16, 64, dtype=torch.float32)

            u_solve = torch.linalg.solve_triangular(M, ve, upper=False, unitriangular=True)

            inv_l = I - L
            for _ in range(14):
                inv_l = I - L @ inv_l
            u_horner = inv_l @ ve

            u_fwd = torch.zeros_like(ve)
            u_fwd[0] = ve[0]
            for i in range(1, 16):
                u_fwd[i] = ve[i] - L[i, :i] @ u_fwd[:i]

            diff_fwd_solve = (u_fwd - u_solve).abs().max().item()
            diff_horner_solve = (u_horner - u_solve).abs().max().item()
            diff_fwd_horner = (u_fwd - u_horner).abs().max().item()

            assert diff_fwd_solve < 1e-5, f"Forward substitution vs solve diff: {diff_fwd_solve}"
            assert diff_horner_solve < 1e-5, f"Horner vs solve diff: {diff_horner_solve}"
            assert diff_fwd_horner < 1e-5, f"Forward substitution vs Horner diff: {diff_fwd_horner}"

        for trial in range(20):
            L = torch.tril(torch.randn(16, 16, dtype=torch.float32) * 1.5, diagonal=-1)
            I = torch.eye(16, dtype=torch.float32)
            M = I + L
            ve = torch.randn(16, 64, dtype=torch.float32)

            u_solve = torch.linalg.solve_triangular(M, ve, upper=False, unitriangular=True)

            inv_l = I - L
            for _ in range(14):
                inv_l = I - L @ inv_l
            u_horner = inv_l @ ve

            u_fwd = torch.zeros_like(ve)
            u_fwd[0] = ve[0]
            for i in range(1, 16):
                u_fwd[i] = ve[i] - L[i, :i] @ u_fwd[:i]

            norm_solve = u_solve.abs().max().item()
            rel_fwd = (u_fwd - u_solve).abs().max().item() / norm_solve
            rel_horner = (u_horner - u_solve).abs().max().item() / norm_solve
            rel_fwd_horner = (u_fwd - u_horner).abs().max().item() / norm_solve

            assert rel_fwd < 1e-5, f"Forward substitution vs solve rel error: {rel_fwd}"
            assert rel_horner < 1e-5, f"Horner vs solve rel error: {rel_horner}"
            assert rel_fwd_horner < 1e-5, f"Forward substitution vs Horner rel error: {rel_fwd_horner}"

    def test_analytical_all_ones_l_matrix(self):
        L = torch.tril(torch.ones(16, 16, dtype=torch.float32), diagonal=-1)
        I = torch.eye(16, dtype=torch.float32)

        inv_exact = torch.eye(16, dtype=torch.float32)
        for i in range(1, 16):
            inv_exact[i, i - 1] = -1.0

        prod = (I + L) @ inv_exact
        assert torch.allclose(prod, I, atol=1e-7), "Analytical inverse theorem check failed"

        inv_horner = I - L
        for _ in range(14):
            inv_horner = I - L @ inv_horner

        diff = (inv_horner - inv_exact).abs().max().item()
        assert diff < 1e-5, f"Horner P15 deviated from analytical inverse on all-ones L: {diff}"


class TestPathologicalGeometryAndDecay:

    @pytest.mark.parametrize("decay_mode", [
        "exact_1",
        "near_1_1e-7",
        "near_0_1e-18",
        "near_0_1e-35",
        "alternating_extreme",
    ])
    def test_xla_extreme_decay_boundaries(self, decay_mode: str):
        B, H, L, dk, dv = 2, 2, 32, 64, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, L, dk)
        k = normalize_keys(torch.randn(B, H, L, dk))
        v = torch.randn(B, H, L, dv)
        b = torch.rand(B, H, L, dk)
        w = torch.rand(B, H, L, dv)
        s0 = torch.randn(B, H, dk, dv)

        if decay_mode == "exact_1":
            alpha = torch.ones(B, H, L, dk)
        elif decay_mode == "near_1_1e-7":
            alpha = 1.0 - torch.rand(B, H, L, dk) * 1e-7
        elif decay_mode == "near_0_1e-18":
            alpha = torch.full((B, H, L, dk), 1e-18)
        elif decay_mode == "near_0_1e-35":
            alpha = torch.full((B, H, L, dk), 1e-35)
        elif decay_mode == "alternating_extreme":
            alpha = torch.ones(B, H, L, dk)
            alpha[:, :, 0::2, :] = 1e-15

        out_xla, s_xla = xla_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert not torch.isnan(out_xla).any(), f"NaN in XLA out under {decay_mode}"
        assert not torch.isnan(s_xla).any(), f"NaN in XLA state under {decay_mode}"
        assert not torch.isinf(out_xla).any(), f"Inf in XLA out under {decay_mode}"
        assert not torch.isinf(s_xla).any(), f"Inf in XLA state under {decay_mode}"

        diff_out = (out_xla - out_ref).abs().max().item()
        diff_state = (s_xla - s_ref).abs().max().item()
        assert diff_out < 1e-4, f"XLA out diff {diff_out:.2e} exceeded 1e-4 under {decay_mode}"
        assert diff_state < 1e-4, f"XLA state diff {diff_state:.2e} exceeded 1e-4 under {decay_mode}"

    @pytest.mark.skipif(not CUDA_ONLINE, reason="CUDA/Triton unavailable")
    @pytest.mark.parametrize("decay_mode", [
        "exact_1",
        "near_1_1e-7",
        "near_0_1e-18",
        "near_0_1e-35",
        "alternating_extreme",
    ])
    def test_triton_extreme_decay_boundaries(self, decay_mode: str):
        B, H, L, dk, dv = 2, 2, 32, 64, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, L, dk, device="cuda")
        k = normalize_keys(torch.randn(B, H, L, dk, device="cuda"))
        v = torch.randn(B, H, L, dv, device="cuda")
        b = torch.rand(B, H, L, dk, device="cuda")
        w = torch.rand(B, H, L, dv, device="cuda")
        s0 = torch.randn(B, H, dk, dv, device="cuda")

        if decay_mode == "exact_1":
            alpha = torch.ones(B, H, L, dk, device="cuda")
        elif decay_mode == "near_1_1e-7":
            alpha = 1.0 - torch.rand(B, H, L, dk, device="cuda") * 1e-7
        elif decay_mode == "near_0_1e-18":
            alpha = torch.full((B, H, L, dk), 1e-18, device="cuda")
        elif decay_mode == "near_0_1e-35":
            alpha = torch.full((B, H, L, dk), 1e-35, device="cuda")
        elif decay_mode == "alternating_extreme":
            alpha = torch.ones(B, H, L, dk, device="cuda")
            alpha[:, :, 0::2, :] = 1e-15

        out_tr, s_tr = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert not torch.isnan(out_tr).any(), f"NaN in Triton out under {decay_mode}"
        assert not torch.isnan(s_tr).any(), f"NaN in Triton state under {decay_mode}"
        assert not torch.isinf(out_tr).any(), f"Inf in Triton out under {decay_mode}"
        assert not torch.isinf(s_tr).any(), f"Inf in Triton state under {decay_mode}"

        diff_out = (out_tr - out_ref).abs().max().item()
        diff_state = (s_tr - s_ref).abs().max().item()
        assert diff_out < 1e-4, f"Triton out diff {diff_out:.2e} exceeded 1e-4 under {decay_mode}"
        assert diff_state < 1e-4, f"Triton state diff {diff_state:.2e} exceeded 1e-4 under {decay_mode}"

    def test_anti_collinear_alternating_keys(self):
        B, H, L, dk, dv = 2, 2, 32, 64, 64
        device = DEVICE
        torch.manual_seed(999)
        k0 = normalize_keys(torch.randn(B, H, 1, dk, device=device))
        signs = torch.tensor([(-1.0) ** t for t in range(L)], device=device).view(1, 1, L, 1)
        k = (k0.expand(B, H, L, dk) * signs).contiguous()

        q = torch.randn(B, H, L, dk, device=device)
        v = torch.randn(B, H, L, dv, device=device)
        alpha = torch.ones(B, H, L, dk, device=device)
        b = torch.ones(B, H, L, dk, device=device)
        w = torch.ones(B, H, L, dv, device=device)
        s0 = torch.randn(B, H, dk, dv, device=device)

        out_xla, s_xla = xla_dgda_prefill(q.cpu(), k.cpu(), v.cpu(), alpha.cpu(), b.cpu(), w.cpu(), initial_state=s0.cpu())
        out_ref, s_ref = ref_dgda_prefill(q.cpu(), k.cpu(), v.cpu(), alpha.cpu(), b.cpu(), w.cpu(), initial_state=s0.cpu())
        diff_xla = (out_xla - out_ref).abs().max().item()
        assert diff_xla < 1e-4, f"Anti-collinear XLA out diff: {diff_xla:.2e}"

        if CUDA_ONLINE:
            out_tr, s_tr = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
            diff_tr = (out_tr - out_ref.cuda()).abs().max().item()
            assert diff_tr < 1e-4, f"Anti-collinear Triton out diff: {diff_tr:.2e}"


class TestVariableHeadDimensions:

    @pytest.mark.parametrize("dk", [32, 64, 128])
    @pytest.mark.parametrize("dv", [32, 64, 128, 256])
    def test_xla_variable_dk_dv(self, dk: int, dv: int):
        B, H, L = 2, 2, 32
        torch.manual_seed(dk * 1000 + dv)
        q = torch.randn(B, H, L, dk)
        k = normalize_keys(torch.randn(B, H, L, dk))
        v = torch.randn(B, H, L, dv)
        alpha = torch.rand(B, H, L, dk) * 0.8 + 0.1
        b = torch.rand(B, H, L, dk)
        w = torch.rand(B, H, L, dv)
        s0 = torch.randn(B, H, dk, dv)

        out_xla, s_xla = xla_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert out_xla.shape == (B, H, L, dv)
        assert s_xla.shape == (B, H, dk, dv)
        diff_out = (out_xla - out_ref).abs().max().item()
        diff_state = (s_xla - s_ref).abs().max().item()
        assert diff_out < 1e-4, f"XLA dk={dk}, dv={dv} out diff: {diff_out:.2e}"
        assert diff_state < 1e-4, f"XLA dk={dk}, dv={dv} state diff: {diff_state:.2e}"

    @pytest.mark.skipif(not CUDA_ONLINE, reason="CUDA/Triton unavailable")
    @pytest.mark.parametrize("dk", [32, 64, 128])
    @pytest.mark.parametrize("dv", [32, 64, 128, 256])
    def test_triton_variable_dk_dv(self, dk: int, dv: int):
        B, H, L = 2, 2, 32
        torch.manual_seed(dk * 1000 + dv)
        q = torch.randn(B, H, L, dk, device="cuda")
        k = normalize_keys(torch.randn(B, H, L, dk, device="cuda"))
        v = torch.randn(B, H, L, dv, device="cuda")
        alpha = torch.rand(B, H, L, dk, device="cuda") * 0.8 + 0.1
        b = torch.rand(B, H, L, dk, device="cuda")
        w = torch.rand(B, H, L, dv, device="cuda")
        s0 = torch.randn(B, H, dk, dv, device="cuda")

        out_tr, s_tr = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert out_tr.shape == (B, H, L, dv)
        assert s_tr.shape == (B, H, dk, dv)
        diff_out = (out_tr - out_ref).abs().max().item()
        diff_state = (s_tr - s_ref).abs().max().item()
        assert diff_out < 1e-4, f"Triton dk={dk}, dv={dv} out diff: {diff_out:.2e}"
        assert diff_state < 1e-4, f"Triton dk={dk}, dv={dv} state diff: {diff_state:.2e}"


class TestSequenceLengthBoundaries:

    @pytest.mark.parametrize("L", [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65])
    def test_xla_boundary_lengths(self, L: int):
        B, H, dk, dv = 2, 2, 64, 64
        torch.manual_seed(100 + L)
        q = torch.randn(B, H, L, dk)
        k = normalize_keys(torch.randn(B, H, L, dk))
        v = torch.randn(B, H, L, dv)
        alpha = torch.rand(B, H, L, dk) * 0.8 + 0.1
        b = torch.rand(B, H, L, dk)
        w = torch.rand(B, H, L, dv)
        s0 = torch.randn(B, H, dk, dv)

        out_xla, s_xla = xla_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert out_xla.shape == (B, H, L, dv)
        assert s_xla.shape == (B, H, dk, dv)
        diff_out = (out_xla - out_ref).abs().max().item()
        diff_state = (s_xla - s_ref).abs().max().item()
        assert diff_out < 1e-4, f"XLA boundary L={L} out diff: {diff_out:.2e}"
        assert diff_state < 1e-4, f"XLA boundary L={L} state diff: {diff_state:.2e}"

    @pytest.mark.skipif(not CUDA_ONLINE, reason="CUDA/Triton unavailable")
    @pytest.mark.parametrize("L", [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65])
    def test_triton_boundary_lengths(self, L: int):
        B, H, dk, dv = 2, 2, 64, 64
        torch.manual_seed(100 + L)
        q = torch.randn(B, H, L, dk, device="cuda")
        k = normalize_keys(torch.randn(B, H, L, dk, device="cuda"))
        v = torch.randn(B, H, L, dv, device="cuda")
        alpha = torch.rand(B, H, L, dk, device="cuda") * 0.8 + 0.1
        b = torch.rand(B, H, L, dk, device="cuda")
        w = torch.rand(B, H, L, dv, device="cuda")
        s0 = torch.randn(B, H, dk, dv, device="cuda")

        out_tr, s_tr = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert out_tr.shape == (B, H, L, dv)
        assert s_tr.shape == (B, H, dk, dv)
        diff_out = (out_tr - out_ref).abs().max().item()
        diff_state = (s_tr - s_ref).abs().max().item()
        assert diff_out < 1e-4, f"Triton boundary L={L} out diff: {diff_out:.2e}"
        assert diff_state < 1e-4, f"Triton boundary L={L} state diff: {diff_state:.2e}"
