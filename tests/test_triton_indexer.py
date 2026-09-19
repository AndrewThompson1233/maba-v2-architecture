
import math
import os
import pytest
import torch
import torch.nn.functional as F

from maba_sparse.kernels.common import (
    ref_compute_centroids,
    ref_index_topk,
    ref_stream_superposition,
    reference_compute_centroids,
    reference_index_topk,
    reference_stream_superposition,
)
from maba_sparse.kernels.cpu_indexer import (
    cpu_compute_centroids,
    cpu_index_topk,
    cpu_stream_superposition,
)
from maba_sparse.kernels.dispatcher import (
    _KERNEL_REGISTRY,
    clear_fallback_warnings,
    dispatch_compute_centroids,
    dispatch_index_topk,
    dispatch_stream_superposition,
    get_backend,
    get_kernel,
    is_cuda_sm75_available,
    is_triton_available,
)
from maba_sparse.kernels.triton_indexer import (
    triton_compute_centroids,
    triton_index_topk,
    triton_stream_superposition,
)
from maba_sparse.kernels.xla_indexer import (
    xla_compute_centroids,
    xla_index_topk,
    xla_stream_superposition,
)

CUDA_AVAILABLE = torch.cuda.is_available() and is_cuda_sm75_available()
TRITON_ACTIVE = CUDA_AVAILABLE and is_triton_available()


@pytest.fixture(autouse=True)
def reset_dispatcher_environment(monkeypatch):
    monkeypatch.delenv("MABA_BACKEND", raising=False)
    monkeypatch.delenv("MABA_STRICT_BACKEND", raising=False)
    _KERNEL_REGISTRY.setdefault("triton", {})["compute_centroids"] = triton_compute_centroids
    _KERNEL_REGISTRY.setdefault("triton", {})["index_topk"] = triton_index_topk
    _KERNEL_REGISTRY.setdefault("triton", {})["stream_superposition"] = triton_stream_superposition
    _KERNEL_REGISTRY.setdefault("cpu", {})["compute_centroids"] = cpu_compute_centroids
    _KERNEL_REGISTRY.setdefault("cpu", {})["index_topk"] = cpu_index_topk
    _KERNEL_REGISTRY.setdefault("cpu", {})["stream_superposition"] = cpu_stream_superposition
    _KERNEL_REGISTRY.setdefault("xla", {})["compute_centroids"] = xla_compute_centroids
    _KERNEL_REGISTRY.setdefault("xla", {})["index_topk"] = xla_index_topk
    _KERNEL_REGISTRY.setdefault("xla", {})["stream_superposition"] = xla_stream_superposition
    clear_fallback_warnings()




class TestIndexerEnvironment:


    def test_environment_hardware_detection(self):
        has_cuda = torch.cuda.is_available()
        sm75 = is_cuda_sm75_available()
        triton_avail = is_triton_available()
        if has_cuda:
            assert sm75, "CUDA is available but sm_75+ check failed"
            assert triton_avail, "CUDA is available but Triton compiler is missing"
        else:
            assert not sm75
            assert not TRITON_ACTIVE

    def test_dispatcher_indexer_registry_entries(self):
        for backend in ("reference", "cpu", "triton", "xla"):
            for op in ("compute_centroids", "index_topk", "stream_superposition"):
                fn = get_kernel(backend, op)
                assert fn is not None, f"Kernel {op} for backend {backend} is None"
                assert callable(fn), f"Kernel {op} for backend {backend} is not callable"

    def test_lazy_loading_indexer_kernels(self):
        fn_triton = get_kernel("triton", "compute_centroids")
        assert fn_triton is triton_compute_centroids

        fn_cpu = get_kernel("cpu", "compute_centroids")
        assert fn_cpu is cpu_compute_centroids

        fn_xla = get_kernel("xla", "compute_centroids")
        assert fn_xla is xla_compute_centroids




class TestTritonCentroidPoolingParity:

    @pytest.mark.parametrize("B,L,d_idx", [
        (1, 64, 64),
        (2, 128, 64),
        (4, 256, 64),
        (7, 65, 64),
        (1, 1, 64),
        (2, 16, 64),
        (2, 63, 64),
        (2, 100, 64),
        (2, 127, 64),
        (1, 512, 64),
        (2, 64, 32),
        (2, 64, 128),
    ])
    def test_centroid_pooling_parity_fp32(self, B, L, d_idx):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        k_idx = torch.randn(B, L, d_idx, device=device, dtype=torch.float32)

        c_disp = dispatch_compute_centroids(k_idx, block_size=64)
        c_ref = ref_compute_centroids(k_idx, block_size=64)
        c_cpu = cpu_compute_centroids(k_idx, block_size=64)
        c_xla = xla_compute_centroids(k_idx, block_size=64, use_bucketing=True)

        assert torch.allclose(c_disp, c_ref, atol=1e-4), f"Dispatch max diff: {(c_disp - c_ref).abs().max()}"
        assert torch.allclose(c_cpu, c_ref, atol=1e-4), f"CPU max diff: {(c_cpu - c_ref).abs().max()}"
        assert torch.allclose(c_xla, c_ref, atol=1e-4), f"XLA max diff: {(c_xla - c_ref).abs().max()}"

        if CUDA_AVAILABLE:
            c_triton = triton_compute_centroids(k_idx, block_size=64)
            assert torch.allclose(c_triton, c_ref, atol=1e-4), f"Triton max diff: {(c_triton - c_ref).abs().max()}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA GPU")
    @pytest.mark.parametrize("dtype,tol", [
        (torch.float16, 5e-3),
        (torch.bfloat16, 1e-2),
    ])
    def test_centroid_pooling_parity_fp16_bf16(self, dtype, tol):
        B, L, d_idx = 2, 128, 64
        k_idx = torch.randn(B, L, d_idx, device="cuda", dtype=dtype)

        c_triton = triton_compute_centroids(k_idx, block_size=64)
        c_ref = ref_compute_centroids(k_idx, block_size=64)

        max_diff = (c_triton.float() - c_ref.float()).abs().max().item()
        assert max_diff <= tol, f"Precision {dtype} exceeded tolerance {tol}: max diff {max_diff}"

    def test_centroid_needle_in_haystack_preservation(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx, block_size = 1, 256, 64, 64
        k_idx = torch.full((B, L, d_idx), 0.01, device=device)
        k_idx[0, 70, :] = 10.0

        c = dispatch_compute_centroids(k_idx, block_size=block_size)
        needle_centroid = c[0, 1, 0].item()

        pure_mean = (63 * 0.01 + 10.0) / 64.0
        assert needle_centroid > 30 * pure_mean, f"Needle dilution detected! Val: {needle_centroid}"
        assert abs(needle_centroid - 0.5 * (pure_mean + 10.0)) < 1e-3

    def test_centroid_boundary_padding_semantics(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx, block_size = 1, 100, 64, 64
        k_idx = torch.full((B, L, d_idx), -2.0, device=device)

        c = dispatch_compute_centroids(k_idx, block_size=block_size)
        assert torch.allclose(c[0, 0, :], torch.tensor(-2.0, device=device), atol=1e-4)
        expected_b1 = 0.5 * ((-72.0 / 64.0) + 0.0)
        assert torch.allclose(c[0, 1, :], torch.tensor(expected_b1, device=device), atol=1e-4)




class TestTritonIndexTopKParity:

    @pytest.mark.parametrize("B,L,top_k,lam", [
        (1, 64, 1, 0.5),
        (2, 128, 2, 0.5),
        (3, 256, 4, 0.5),
        (2, 512, 8, 1.0),
        (1, 100, 2, 0.25),
        (2, 65, 2, 0.0),
        (2, 16, 2, 0.5),
        (1, 512, 32, 0.5),
    ])
    def test_index_topk_parity_exact_discrete_indices(self, B, L, top_k, lam):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        d_idx = 64
        q_idx = torch.randn(B, L, d_idx, device=device)
        k_idx = torch.randn(B, L, d_idx, device=device)
        centroids = dispatch_compute_centroids(k_idx, block_size=64)

        idx_disp = dispatch_index_topk(q_idx, centroids, lambda_dist=lam, top_k=top_k, block_size=64)
        idx_ref = ref_index_topk(q_idx, centroids, lambda_dist=lam, top_k=top_k, block_size=64)
        idx_cpu = cpu_index_topk(q_idx, centroids, lambda_dist=lam, top_k=top_k, block_size=64)
        idx_xla = xla_index_topk(q_idx, centroids, lambda_dist=lam, top_k=top_k, block_size=64, use_bucketing=True)

        assert idx_disp.dtype == torch.long
        assert idx_cpu.dtype == torch.long
        assert idx_xla.dtype == torch.long

        assert torch.equal(idx_disp, idx_ref), "Dispatch index mismatch vs reference"
        assert torch.equal(idx_cpu, idx_ref), "CPU index mismatch vs reference"
        assert torch.equal(idx_xla, idx_ref), "XLA index mismatch vs reference"

        if CUDA_AVAILABLE:
            idx_triton = triton_index_topk(q_idx, centroids, lambda_dist=lam, top_k=top_k, block_size=64)
            assert torch.equal(idx_triton, idx_ref), "Triton index mismatch vs reference"

    def test_index_topk_scores_parity_fp32(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 128, 64
        q_idx = torch.randn(B, L, d_idx, device=device)
        k_idx = torch.randn(B, L, d_idx, device=device)
        centroids = dispatch_compute_centroids(k_idx, block_size=64)

        idx_disp, sc_disp = dispatch_index_topk(q_idx, centroids, top_k=2, block_size=64, return_scores=True)
        idx_ref, sc_ref = ref_index_topk(q_idx, centroids, top_k=2, block_size=64, return_scores=True)

        finite_mask = torch.isfinite(sc_ref)
        assert torch.allclose(sc_disp[finite_mask], sc_ref[finite_mask], atol=1e-4)

    def test_index_topk_strict_causal_masking(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 256, 64
        q_idx = torch.randn(B, L, d_idx, device=device)
        centroids = torch.randn(B, 4, d_idx, device=device)

        indices, scores = dispatch_index_topk(q_idx, centroids, top_k=4, block_size=64, return_scores=True)
        for t in range(L):
            max_allowed_block = t // 64
            for b in range(4):
                if b > max_allowed_block:
                    assert (scores[:, t, b] == float("-inf")).all(), (
                        f"Future block {b} at token {t} has non-inf score: {scores[:, t, b]}"
                    )
                else:
                    assert torch.isfinite(scores[:, t, b]).all()

            available_past_blocks = max_allowed_block + 1
            token_indices = indices[0, t, :available_past_blocks]
            assert (token_indices <= max_allowed_block).all(), (
                f"Causal violation at token {t}: selected {token_indices.tolist()}, "
                f"max allowed {max_allowed_block}"
            )


    def test_index_topk_distance_penalty_monotonicity(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 1, 256, 64
        q_idx = torch.ones(B, L, d_idx, device=device) / math.sqrt(d_idx)
        centroids = torch.ones(B, 4, d_idx, device=device) / math.sqrt(d_idx)

        indices = dispatch_index_topk(q_idx, centroids, lambda_dist=1.0, top_k=4, block_size=64)
        last_token_idx = indices[0, 255, :].tolist()
        assert last_token_idx == [3, 2, 1, 0], f"Expected descending order [3, 2, 1, 0], got {last_token_idx}"

    def test_index_topk_short_sequence_budget_clamping(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 32, 64
        q_idx = torch.randn(B, L, d_idx, device=device)
        centroids = torch.randn(B, 1, d_idx, device=device)

        indices = dispatch_index_topk(q_idx, centroids, top_k=32, block_size=64)
        assert indices.shape == (B, L, 1), f"Expected shape {(B, L, 1)}, got {indices.shape}"
        assert (indices == 0).all()




class TestTritonStreamSuperpositionParity:

    @pytest.mark.parametrize("B,H,L,D", [
        (1, 1, 64, 64),
        (2, 4, 128, 64),
        (4, 8, 64, 64),
        (1, 10, 256, 64),
        (2, 4, 15, 64),
        (1, 1, 1, 64),
        (2, 4, 64, 32),
        (2, 4, 64, 128),
    ])
    def test_stream_superposition_parity_3d_gate_fp32(self, B, H, L, D):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        ol = torch.randn(B, H, L, D, device=device)
        os = torch.randn(B, H, L, D, device=device)
        oh = torch.randn(B, H, L, D, device=device)
        logits = torch.randn(B, L, 3, device=device)

        out_disp = dispatch_stream_superposition(ol, os, oh, gate_logits=logits)
        out_ref = ref_stream_superposition(ol, os, oh, gate_logits=logits)
        out_cpu = cpu_stream_superposition(ol, os, oh, gate_logits=logits)
        out_xla = xla_stream_superposition(ol, os, oh, gate_logits=logits)

        assert torch.allclose(out_disp, out_ref, atol=1e-4)
        assert torch.allclose(out_cpu, out_ref, atol=1e-4)
        assert torch.allclose(out_xla, out_ref, atol=1e-4)

        if CUDA_AVAILABLE:
            out_triton = triton_stream_superposition(ol, os, oh, gate_logits=logits)
            assert torch.allclose(out_triton, out_ref, atol=1e-4)

    def test_stream_superposition_parity_4d_gate_fp32(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 2, 4, 64, 64
        ol = torch.randn(B, H, L, D, device=device)
        os = torch.randn(B, H, L, D, device=device)
        oh = torch.randn(B, H, L, D, device=device)
        logits_4d = torch.randn(B, H, L, 3, device=device)

        out_disp = dispatch_stream_superposition(ol, os, oh, gate_logits=logits_4d)
        out_ref = ref_stream_superposition(ol, os, oh, gate_logits=logits_4d)
        assert torch.allclose(out_disp, out_ref, atol=1e-4)

    def test_stream_superposition_direct_gate_weights(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 2, 4, 32, 64
        ol = torch.randn(B, H, L, D, device=device)
        os = torch.randn(B, H, L, D, device=device)
        oh = torch.randn(B, H, L, D, device=device)
        weights = F.softmax(torch.randn(B, L, 3, device=device), dim=-1)

        out_disp = dispatch_stream_superposition(ol, os, oh, gate_weights=weights)
        out_ref = ref_stream_superposition(ol, os, oh, gate_weights=weights)
        assert torch.allclose(out_disp, out_ref, atol=1e-4)

    def test_stream_superposition_partition_of_unity(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 2, 4, 64, 64
        x = torch.randn(B, H, L, D, device=device)
        logits = torch.randn(B, L, 3, device=device)

        out = dispatch_stream_superposition(x, x, x, gate_logits=logits)
        assert torch.allclose(out, x, atol=1e-5), f"Partition of unity violated! Max diff: {(out - x).abs().max()}"

    def test_stream_superposition_extreme_logits(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 1, 2, 4, 64
        ol = torch.ones(B, H, L, D, device=device) * 1.0
        os = torch.ones(B, H, L, D, device=device) * 2.0
        oh = torch.ones(B, H, L, D, device=device) * 3.0

        logits = torch.tensor([[[1e4, -1e4, -1e4]] * 4], device=device)
        out = dispatch_stream_superposition(ol, os, oh, gate_logits=logits)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert torch.allclose(out, ol, atol=1e-4)




class TestIndexerSuperpositionAutograd:

    def test_autograd_centroid_strictly_positive_gradient(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx, block_size = 2, 100, 64, 64
        k_idx = torch.randn(B, L, d_idx, device=device, requires_grad=True)

        centroids = dispatch_compute_centroids(k_idx, block_size=block_size)
        loss = centroids.sum()
        loss.backward()

        assert k_idx.grad is not None
        assert not torch.isnan(k_idx.grad).any()
        assert not torch.isinf(k_idx.grad).any()

        min_allowed = 1.0 / (2.0 * block_size) - 1e-6
        active_grads = k_idx.grad[:, :L, :]
        assert (active_grads >= min_allowed).all(), (
            f"Gradient positivity violated! Min grad {active_grads.min().item()} < {min_allowed}"
        )

    def test_autograd_centroid_parity_against_reference(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx, block_size = 2, 128, 64, 64

        k1 = torch.randn(B, L, d_idx, device=device, requires_grad=True)
        k2 = k1.detach().clone().requires_grad_(True)

        c_ref = ref_compute_centroids(k1, block_size=block_size)
        grad_out = torch.randn_like(c_ref)
        c_ref.backward(grad_out)

        c_disp = dispatch_compute_centroids(k2, block_size=block_size)
        c_disp.backward(grad_out)

        assert torch.allclose(k1.grad, k2.grad, atol=1e-5), f"Max grad diff: {(k1.grad - k2.grad).abs().max()}"

    def test_autograd_topk_scores_differentiable(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, nb, d_idx = 2, 64, 2, 64
        q_idx = torch.randn(B, L, d_idx, device=device, requires_grad=True)
        centroids = torch.randn(B, nb, d_idx, device=device, requires_grad=True)

        _, scores = dispatch_index_topk(q_idx, centroids, top_k=2, block_size=64, return_scores=True)
        finite_scores = scores[torch.isfinite(scores)]
        finite_scores.sum().backward()

        assert q_idx.grad is not None
        assert centroids.grad is not None
        assert not torch.isnan(q_idx.grad).any()
        assert not torch.isnan(centroids.grad).any()

    def test_autograd_stream_superposition_analytical_vjp(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 2, 4, 32, 64
        ol = torch.randn(B, H, L, D, device=device, requires_grad=True)
        os = torch.randn(B, H, L, D, device=device, requires_grad=True)
        oh = torch.randn(B, H, L, D, device=device, requires_grad=True)
        logits = torch.randn(B, L, 3, device=device, requires_grad=True)

        out = dispatch_stream_superposition(ol, os, oh, gate_logits=logits)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert ol.grad is not None
        assert os.grad is not None
        assert oh.grad is not None
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

        g = F.softmax(logits, dim=-1)
        gl = g[:, :, 0:1].unsqueeze(1)
        gs = g[:, :, 1:2].unsqueeze(1)
        gh = g[:, :, 2:3].unsqueeze(1)

        assert torch.allclose(ol.grad, gl * grad_out, atol=1e-5)
        assert torch.allclose(os.grad, gs * grad_out, atol=1e-5)
        assert torch.allclose(oh.grad, gh * grad_out, atol=1e-5)




class TestDispatcherRoutingAndFallback:

    def test_dispatcher_backend_detection(self):
        if CUDA_AVAILABLE:
            assert get_backend(torch.device("cuda:0")) == "triton"
        assert get_backend(torch.device("cpu")) == "cpu"

    def test_dispatcher_manual_backend_overrides(self):
        old_env = os.environ.get("MABA_BACKEND")
        try:
            for override in ("reference", "cpu", "xla", "triton"):
                os.environ["MABA_BACKEND"] = override
                assert get_backend(torch.device("cpu")) == override
        finally:
            if old_env is not None:
                os.environ["MABA_BACKEND"] = old_env
            else:
                os.environ.pop("MABA_BACKEND", None)

    def test_dispatcher_graceful_fallback_centroids_on_error(self, monkeypatch):
        clear_fallback_warnings()
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        k_idx = torch.randn(2, 64, 64, device=device)

        def faulty_centroids(*args, **kwargs):
            raise RuntimeError("Simulated GPU kernel fault")

        monkeypatch.setitem(_KERNEL_REGISTRY["triton"], "compute_centroids", faulty_centroids)
        monkeypatch.setenv("MABA_BACKEND", "triton")

        with pytest.warns(RuntimeWarning, match="Simulated GPU kernel fault"):
            c = dispatch_compute_centroids(k_idx, block_size=64)
        c_ref = ref_compute_centroids(k_idx, block_size=64)
        assert torch.allclose(c, c_ref, atol=1e-4)

    def test_dispatcher_strict_mode_raises_exception(self, monkeypatch):
        clear_fallback_warnings()
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        k_idx = torch.randn(2, 64, 64, device=device)

        def faulty_centroids(*args, **kwargs):
            raise RuntimeError("Kernel failed in strict mode")

        monkeypatch.setitem(_KERNEL_REGISTRY["triton"], "compute_centroids", faulty_centroids)
        monkeypatch.setenv("MABA_BACKEND", "triton")
        monkeypatch.setenv("MABA_STRICT_BACKEND", "1")

        with pytest.raises(RuntimeError):
            dispatch_compute_centroids(k_idx, block_size=64)




class TestIndexerBoundaryStress:

    def test_adversarial_all_zero_inputs(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 128, 64
        k_zero = torch.zeros(B, L, d_idx, device=device)
        c_zero = dispatch_compute_centroids(k_zero, block_size=64)
        assert (c_zero == 0.0).all()

        q_zero = torch.zeros(B, L, d_idx, device=device)
        idx_zero = dispatch_index_topk(q_zero, c_zero, top_k=2, block_size=64)
        assert idx_zero.shape == (B, L, 2)

    def test_adversarial_empty_sequence_L0(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        k_empty = torch.empty(2, 0, 64, device=device)
        c_empty = dispatch_compute_centroids(k_empty, block_size=64)
        assert c_empty.shape == (2, 0, 64)

        q_empty = torch.empty(2, 0, 64, device=device)
        idx_empty = dispatch_index_topk(q_empty, c_empty, top_k=2, block_size=64)
        assert idx_empty.shape == (2, 0, 0)

    def test_adversarial_lambda_extremes(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 128, 64
        q = torch.randn(B, L, d_idx, device=device)
        c = torch.randn(B, 2, d_idx, device=device)

        idx_0 = dispatch_index_topk(q, c, lambda_dist=0.0, top_k=2, block_size=64)
        idx_1000 = dispatch_index_topk(q, c, lambda_dist=1000.0, top_k=2, block_size=64)
        assert idx_0.shape == (B, L, 2)
        assert idx_1000.shape == (B, L, 2)

    def test_adversarial_collinear_centroids(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 1, 128, 64
        q = torch.randn(B, L, d_idx, device=device)
        c = torch.randn(1, 1, d_idx, device=device).expand(B, 2, d_idx).contiguous()

        idx = dispatch_index_topk(q, c, lambda_dist=0.5, top_k=2, block_size=64)
        assert not torch.isnan(idx).any()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA GPU for AMP autocast")
    def test_adversarial_amp_autocast(self):
        B, L, d_idx = 2, 128, 64
        k = torch.randn(B, L, d_idx, device="cuda")
        q = torch.randn(B, L, d_idx, device="cuda")

        with torch.cuda.amp.autocast(dtype=torch.float16):
            c = dispatch_compute_centroids(k, block_size=64)
            idx = dispatch_index_topk(q, c, top_k=2, block_size=64)
            ol = torch.randn(B, 4, L, d_idx, device="cuda")
            os = torch.randn(B, 4, L, d_idx, device="cuda")
            oh = torch.randn(B, 4, L, d_idx, device="cuda")
            logits = torch.randn(B, L, 3, device="cuda")
            out = dispatch_stream_superposition(ol, os, oh, gate_logits=logits)

        assert not torch.isnan(c).any()
        assert not torch.isnan(out).any()
        assert idx.dtype == torch.long
