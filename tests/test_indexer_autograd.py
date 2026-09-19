
import gc
import pytest
import torch

from maba_sparse.kernels.common import (
    ref_compute_centroids,
    ref_stream_superposition,
)
from maba_sparse.kernels.dispatcher import (
    _KERNEL_REGISTRY,
    clear_fallback_warnings,
    is_cuda_sm75_available,
    is_triton_available,
)
from maba_sparse.kernels.triton_indexer import (
    _TritonCentroidFunction,
    _TritonSuperpositionFunction,
    triton_compute_centroids,
    triton_index_topk,
    triton_stream_superposition,
)

CUDA_AVAILABLE = torch.cuda.is_available() and is_cuda_sm75_available()
TRITON_ACTIVE = CUDA_AVAILABLE and is_triton_available()


@pytest.fixture(autouse=True)
def reset_dispatcher(monkeypatch):
    monkeypatch.delenv("MABA_BACKEND", raising=False)
    monkeypatch.delenv("MABA_STRICT_BACKEND", raising=False)
    if TRITON_ACTIVE:
        _KERNEL_REGISTRY.setdefault("triton", {})["compute_centroids"] = triton_compute_centroids
        _KERNEL_REGISTRY.setdefault("triton", {})["index_topk"] = triton_index_topk
        _KERNEL_REGISTRY.setdefault("triton", {})["stream_superposition"] = triton_stream_superposition
    clear_fallback_warnings()


class TestCentroidAutogradGradientStress:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    @pytest.mark.parametrize("B,L,d_idx", [
        (1, 64, 64),
        (2, 128, 64),
        (4, 256, 64),
        (1, 1, 64),
        (2, 15, 64),
        (2, 63, 64),
        (2, 65, 64),
        (3, 100, 64),
        (2, 127, 64),
        (1, 512, 64),
        (2, 64, 32),
        (2, 64, 128),
    ])
    def test_centroid_gradient_positivity_sum_loss(self, B, L, d_idx):
        block_size = 64
        k = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)

        centroids = triton_compute_centroids(k, block_size=block_size)
        loss = centroids.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()
        assert not torch.isinf(k.grad).any()

        min_bound = 1.0 / (2.0 * block_size) - 1e-6
        assert (k.grad >= min_bound).all(), (
            f"Positivity bound violated: min {k.grad.min().item()} < {min_bound} at shape ({B}, {L}, {d_idx})"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    @pytest.mark.parametrize("loss_type", ["weighted", "l2", "exp"])
    def test_centroid_gradient_under_diverse_losses(self, loss_type):
        B, L, d_idx, block_size = 2, 100, 64, 64
        k = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
        c = triton_compute_centroids(k, block_size=block_size)

        if loss_type == "weighted":
            weights = torch.empty_like(c).uniform_(1.0, 3.0)
            loss = (c * weights).sum()
            loss.backward()
            min_expected = 1.0 / (2.0 * block_size) - 1e-6
            assert (k.grad >= min_expected).all(), f"Weighted loss min grad violated: {k.grad.min().item()}"

        elif loss_type == "l2":
            target = c.detach() - 2.0
            loss = 0.5 * ((c - target) ** 2).sum()
            loss.backward()
            min_expected = 2.0 / (2.0 * block_size) - 1e-6
            assert (k.grad >= min_expected).all(), f"L2 loss min grad violated: {k.grad.min().item()}"

        elif loss_type == "exp":
            loss = torch.exp(c * 0.1).sum()
            loss.backward()
            assert (k.grad > 0.0).all(), f"Exp loss non-positive grad detected: {k.grad.min().item()}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    @pytest.mark.parametrize("L,L_padded", [
        (1, 64),
        (15, 64),
        (63, 64),
        (65, 128),
        (100, 128),
        (200, 256),
    ])
    def test_centroid_padded_positions_exact_zero_gradient(self, L, L_padded):
        B, d_idx, block_size = 2, 64, 64
        k_full = torch.randn(B, L_padded, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)

        k_active = k_full[:, :L, :]
        centroids = triton_compute_centroids(k_active, block_size=block_size)
        loss = centroids.sum()
        loss.backward()

        assert k_full.grad is not None
        active_grad = k_full.grad[:, :L, :]
        padded_grad = k_full.grad[:, L:, :]

        min_bound = 1.0 / (2.0 * block_size) - 1e-6
        assert (active_grad >= min_bound).all(), (
            f"Active tokens violated min bound: min={active_grad.min().item()}"
        )

        assert (padded_grad == 0.0).all(), (
            f"Padded positions received non-zero gradient! Max abs: {padded_grad.abs().max().item()}"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_centroid_all_negative_values_in_partial_block(self):
        B, L, d_idx, block_size = 2, 100, 64, 64
        k = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32)
        k[:, 64:, :] = -5.0
        k.requires_grad = True

        centroids = triton_compute_centroids(k, block_size=block_size)
        loss = centroids.sum()
        loss.backward()

        assert k.grad is not None
        assert not torch.isnan(k.grad).any()
        assert not torch.isinf(k.grad).any()

        b1_grad = k.grad[:, 64:, :]
        expected_grad = 0.5 / block_size
        assert torch.allclose(b1_grad, torch.tensor(expected_grad, device="cuda"), atol=1e-5), (
            f"Block 1 gradient mismatch: expected {expected_grad}, got min={b1_grad.min().item()}, max={b1_grad.max().item()}"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_centroid_autograd_parity_vs_reference(self):
        B, L, d_idx, block_size = 2, 128, 64, 64
        k_ref = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
        k_tri = k_ref.detach().clone().requires_grad_(True)

        c_ref = ref_compute_centroids(k_ref, block_size=block_size)
        c_tri = triton_compute_centroids(k_tri, block_size=block_size)

        grad_out = torch.randn_like(c_ref)
        c_ref.backward(grad_out)
        c_tri.backward(grad_out)

        max_diff = (k_ref.grad - k_tri.grad).abs().max().item()
        assert max_diff < 1e-6, f"Centroid autograd max diff vs ref: {max_diff}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_centroid_gradcheck_analytical(self):
        B, L, d_idx, block_size = 1, 32, 32, 64
        torch.manual_seed(123)
        k = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32)

        argmax_pos = torch.randint(0, L, (B, 1, d_idx), device="cuda")
        k.scatter_add_(1, argmax_pos, torch.full_like(argmax_pos, 2.0, dtype=torch.float32))
        k.requires_grad_(True)

        def func(inp):
            return _TritonCentroidFunction.apply(inp, block_size)

        passed = torch.autograd.gradcheck(func, (k,), eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True)
        assert passed, "torch.autograd.gradcheck failed for _TritonCentroidFunction"


class TestSuperpositionAutogradGradientStress:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    @pytest.mark.parametrize("B,H,L,D", [
        (1, 1, 64, 64),
        (2, 4, 128, 64),
        (4, 8, 64, 64),
        (1, 10, 256, 64),
        (2, 4, 15, 64),
        (2, 4, 64, 32),
        (2, 4, 64, 128),
    ])
    def test_superposition_3d_logits_vjp_parity(self, B, H, L, D):
        ol_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        os_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        oh_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        logits_ref = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)

        ol_tri = ol_ref.detach().clone().requires_grad_(True)
        os_tri = os_ref.detach().clone().requires_grad_(True)
        oh_tri = oh_ref.detach().clone().requires_grad_(True)
        logits_tri = logits_ref.detach().clone().requires_grad_(True)

        out_ref = ref_stream_superposition(ol_ref, os_ref, oh_ref, gate_logits=logits_ref)
        out_tri = triton_stream_superposition(ol_tri, os_tri, oh_tri, gate_logits=logits_tri)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_tri.backward(grad_out)

        assert torch.allclose(ol_tri.grad, ol_ref.grad, atol=1e-5)
        assert torch.allclose(os_tri.grad, os_ref.grad, atol=1e-5)
        assert torch.allclose(oh_tri.grad, oh_ref.grad, atol=1e-5)

        diff_logits = (logits_tri.grad - logits_ref.grad).abs().max().item()
        assert diff_logits < 1e-5, f"3D logits VJP diff {diff_logits} >= 1e-5 at ({B}, {H}, {L}, {D})"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    @pytest.mark.parametrize("B,H,L,D", [
        (1, 2, 64, 64),
        (2, 4, 128, 64),
        (2, 8, 32, 64),
    ])
    def test_superposition_4d_logits_vjp_parity(self, B, H, L, D):
        ol_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        os_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        oh_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        logits_ref = torch.randn(B, H, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)

        ol_tri = ol_ref.detach().clone().requires_grad_(True)
        os_tri = os_ref.detach().clone().requires_grad_(True)
        oh_tri = oh_ref.detach().clone().requires_grad_(True)
        logits_tri = logits_ref.detach().clone().requires_grad_(True)

        out_ref = ref_stream_superposition(ol_ref, os_ref, oh_ref, gate_logits=logits_ref)
        out_tri = triton_stream_superposition(ol_tri, os_tri, oh_tri, gate_logits=logits_tri)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_tri.backward(grad_out)

        assert torch.allclose(ol_tri.grad, ol_ref.grad, atol=1e-5)
        assert torch.allclose(os_tri.grad, os_ref.grad, atol=1e-5)
        assert torch.allclose(oh_tri.grad, oh_ref.grad, atol=1e-5)

        diff_logits = (logits_tri.grad - logits_ref.grad).abs().max().item()
        assert diff_logits < 1e-5, f"4D logits VJP diff {diff_logits} >= 1e-5 at ({B}, {H}, {L}, {D})"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    @pytest.mark.parametrize("grad_scale", [1e2, 1e3, 1e4, 1e5])
    def test_superposition_high_gradient_magnitudes(self, grad_scale):
        B, H, L, D = 2, 4, 64, 64
        ol_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        os_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        oh_ref = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        logits_ref = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)

        ol_tri = ol_ref.detach().clone().requires_grad_(True)
        os_tri = os_ref.detach().clone().requires_grad_(True)
        oh_tri = oh_ref.detach().clone().requires_grad_(True)
        logits_tri = logits_ref.detach().clone().requires_grad_(True)

        out_ref = ref_stream_superposition(ol_ref, os_ref, oh_ref, gate_logits=logits_ref)
        out_tri = triton_stream_superposition(ol_tri, os_tri, oh_tri, gate_logits=logits_tri)

        grad_out = torch.randn_like(out_ref) * grad_scale
        out_ref.backward(grad_out)
        out_tri.backward(grad_out)

        rel_err_logits = (
            (logits_tri.grad - logits_ref.grad).norm() / (logits_ref.grad.norm() + 1e-8)
        ).item()
        assert rel_err_logits < 1e-5, f"High gradient scale {grad_scale}: rel error {rel_err_logits} >= 1e-5"
        assert not torch.isnan(logits_tri.grad).any()
        assert not torch.isinf(logits_tri.grad).any()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_superposition_zero_stream_inputs_gradient_isolation(self):
        B, H, L, D = 2, 4, 32, 64
        ol = torch.zeros(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        os = torch.zeros(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        oh = torch.zeros(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        logits = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)

        out = triton_stream_superposition(ol, os, oh, gate_logits=logits)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert logits.grad is not None
        assert (logits.grad == 0.0).all(), (
            f"Expected zero grad for logits when inputs are 0, got max {logits.grad.abs().max().item()}"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_superposition_gradcheck_analytical(self):
        B, H, L, D = 1, 2, 8, 16
        torch.manual_seed(42)
        ol = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        os = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        oh = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
        logits = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)

        def func(l, s, h, g):
            return _TritonSuperpositionFunction.apply(l, s, h, g, None)

        passed = torch.autograd.gradcheck(func, (ol, os, oh, logits), eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True)
        assert passed, "torch.autograd.gradcheck failed for _TritonSuperpositionFunction"


class TestMemoryLeakAndAllocationInvariance:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_centroid_100_iterations_zero_memory_leak(self):
        B, L, d_idx, block_size = 2, 256, 64, 64
        torch.cuda.empty_cache()
        gc.collect()

        for _ in range(5):
            k = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
            c = triton_compute_centroids(k, block_size=block_size)
            c.sum().backward()

        del k, c
        gc.collect()
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        for _ in range(100):
            k = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
            c = triton_compute_centroids(k, block_size=block_size)
            loss = c.sum()
            loss.backward()

        del k, c, loss
        gc.collect()
        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()

        leak = mem_end - mem_start
        assert leak == 0, f"Memory leak detected in compute_centroids: {leak} bytes retained across 100 iterations"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_index_topk_100_iterations_zero_memory_leak(self):
        B, L, nb, d_idx = 2, 256, 4, 64
        torch.cuda.empty_cache()
        gc.collect()

        for _ in range(5):
            q = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
            c = torch.randn(B, nb, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
            idx, sc = triton_index_topk(q, c, top_k=2, block_size=64, return_scores=True)
            sc[torch.isfinite(sc)].sum().backward()

        del q, c, idx, sc
        gc.collect()
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        for _ in range(100):
            q = torch.randn(B, L, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
            c = torch.randn(B, nb, d_idx, device="cuda", dtype=torch.float32, requires_grad=True)
            idx, sc = triton_index_topk(q, c, top_k=2, block_size=64, return_scores=True)
            finite_sc = sc[torch.isfinite(sc)]
            finite_sc.sum().backward()

        del q, c, idx, sc, finite_sc
        gc.collect()
        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()

        leak = mem_end - mem_start
        assert leak == 0, f"Memory leak detected in index_topk: {leak} bytes retained across 100 iterations"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_superposition_100_iterations_zero_memory_leak(self):
        B, H, L, D = 2, 4, 128, 64
        torch.cuda.empty_cache()
        gc.collect()

        for _ in range(5):
            ol = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            os = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            oh = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            logits = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)
            out = triton_stream_superposition(ol, os, oh, gate_logits=logits)
            out.sum().backward()

        del ol, os, oh, logits, out
        gc.collect()
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        for _ in range(100):
            ol = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            os = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            oh = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            logits = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)
            out = triton_stream_superposition(ol, os, oh, gate_logits=logits)
            loss = out.sum()
            loss.backward()

        del ol, os, oh, logits, out, loss
        gc.collect()
        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()

        leak = mem_end - mem_start
        assert leak == 0, f"Memory leak detected in stream_superposition: {leak} bytes retained across 100 iterations"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+")
    def test_combined_milestone3_pipeline_100_iterations_memory_invariance(self):
        B, H, L, D = 2, 4, 256, 64
        block_size = 64
        torch.cuda.empty_cache()
        gc.collect()

        for _ in range(5):
            k = torch.randn(B, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            q = torch.randn(B, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            c = triton_compute_centroids(k, block_size=block_size)
            idx, sc = triton_index_topk(q, c, top_k=4, block_size=block_size, return_scores=True)
            ol = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            os = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            oh = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            logits = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)
            out = triton_stream_superposition(ol, os, oh, gate_logits=logits)
            total_loss = c.sum() + sc[torch.isfinite(sc)].sum() + out.sum()
            total_loss.backward()

        del k, q, c, idx, sc, ol, os, oh, logits, out, total_loss
        gc.collect()
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        for _ in range(100):
            k = torch.randn(B, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            q = torch.randn(B, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            c = triton_compute_centroids(k, block_size=block_size)
            idx, sc = triton_index_topk(q, c, top_k=4, block_size=block_size, return_scores=True)
            ol = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            os = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            oh = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
            logits = torch.randn(B, L, 3, device="cuda", dtype=torch.float32, requires_grad=True)
            out = triton_stream_superposition(ol, os, oh, gate_logits=logits)
            loss = c.sum() + sc[torch.isfinite(sc)].sum() + out.sum()
            loss.backward()

        del k, q, c, idx, sc, ol, os, oh, logits, out, loss
        gc.collect()
        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()

        leak = mem_end - mem_start
        assert leak == 0, f"Combined pipeline leaked {leak} bytes across 100 iterations"
