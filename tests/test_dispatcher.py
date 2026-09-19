
from unittest.mock import patch
import warnings

import pytest
import torch
import torch.nn.functional as F

from maba_sparse.kernels import (
    clear_fallback_warnings,
    dispatch_compute_centroids,
    dispatch_dgda_prefill,
    dispatch_dgda_step,
    dispatch_index_topk,
    dispatch_stream_superposition,
    get_backend,
    reference_compute_centroids,
    reference_dgda_prefill,
    reference_dgda_step,
    reference_index_topk,
    reference_stream_superposition,
    register_kernel,
)


@pytest.fixture(autouse=True)
def reset_env_and_warnings(monkeypatch):
    monkeypatch.delenv("MABA_BACKEND", raising=False)
    monkeypatch.delenv("MABA_STRICT_BACKEND", raising=False)
    monkeypatch.delenv("MABA_VERBOSE", raising=False)
    clear_fallback_warnings()
    yield
    clear_fallback_warnings()


class TestDispatcherAutoRouting:
    def test_routing_cpu_device_selects_cpu_or_reference(self):
        backend = get_backend(torch.device("cpu"))
        assert backend in ("cpu", "reference")
        assert get_backend("cpu") == backend

    def test_routing_cuda_device_selects_triton_when_sm75_available(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("maba_sparse.kernels.dispatcher.is_cuda_sm75_available", return_value=True), \
             patch("maba_sparse.kernels.dispatcher.is_triton_available", return_value=True):
            assert get_backend(torch.device("cuda:0")) == "triton"
            assert get_backend("cuda") == "triton"

    def test_routing_cuda_device_selects_reference_when_sm_under_75(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("maba_sparse.kernels.dispatcher.is_cuda_sm75_available", return_value=False), \
             patch("maba_sparse.kernels.dispatcher.is_triton_available", return_value=True):
            assert get_backend(torch.device("cuda:0")) == "reference"

    def test_routing_cuda_device_selects_reference_when_triton_missing(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("maba_sparse.kernels.dispatcher.is_cuda_sm75_available", return_value=True), \
             patch("maba_sparse.kernels.dispatcher.is_triton_available", return_value=False):
            assert get_backend(torch.device("cuda:0")) == "reference"

    def test_routing_xla_device_selects_xla_when_available(self):
        with patch("maba_sparse.kernels.dispatcher.is_xla_available", return_value=True):
            assert get_backend(torch.device("xla:0")) == "xla"
            assert get_backend("xla") == "xla"

    def test_routing_unsupported_device_fallback_reference(self):
        assert get_backend(torch.device("meta")) == "reference"
        assert get_backend("meta") == "reference"
        assert get_backend("mps") == "reference"


class TestDispatcherManualOverride:
    def test_maba_backend_env_override_reference(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")
        assert get_backend(torch.device("cpu")) == "reference"
        assert get_backend(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) == "reference"

    def test_maba_backend_env_override_ref_alias(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "ref")
        assert get_backend(torch.device("cpu")) == "reference"
        monkeypatch.setenv("MABA_BACKEND", "pytorch")
        assert get_backend(torch.device("cpu")) == "reference"

    def test_maba_backend_env_override_triton(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")
        assert get_backend(torch.device("cpu")) == "triton"
        monkeypatch.setenv("MABA_BACKEND", "cuda")
        assert get_backend(torch.device("cpu")) == "triton"

    def test_maba_backend_env_override_cpu(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "cpu")
        assert get_backend(torch.device("cpu")) == "cpu"
        monkeypatch.setenv("MABA_BACKEND", "openmp")
        assert get_backend(torch.device("cpu")) == "cpu"

    def test_maba_backend_env_override_case_insensitivity(self, monkeypatch):
        for val, expected in [
            ("TRITON", "triton"),
            ("Triton", "triton"),
            ("CPU", "cpu"),
            ("Cpu", "cpu"),
            ("XLA", "xla"),
            ("REFERENCE", "reference"),
            ("Ref", "reference"),
        ]:
            monkeypatch.setenv("MABA_BACKEND", val)
            assert get_backend(torch.device("cpu")) == expected

    def test_maba_backend_env_override_invalid_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "unsupported_accelerator")
        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            get_backend(torch.device("cpu"))


class TestDispatcherGracefulFallback:
    def test_fallback_on_simulated_triton_kernel_exception_prefill(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        def failing_prefill(**kwargs):
            raise RuntimeError("CUDA out of memory / simulated launch failure")

        register_kernel("triton", "dgda_prefill")(failing_prefill)

        B, H, L, dk, dv = 1, 2, 8, 16, 16
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w)
        assert out.shape == (B, H, L, dv)
        assert state.shape == (B, H, dk, dv)
        assert torch.isfinite(out).all()
        assert torch.isfinite(state).all()

    def test_fallback_on_simulated_triton_kernel_exception_step(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        def failing_step(**kwargs):
            raise NotImplementedError("Kernel not implemented for current shape")

        register_kernel("triton", "dgda_step")(failing_step)

        B, H, dk, dv = 1, 2, 16, 16
        q = torch.randn(B, H, dk)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1)
        v = torch.randn(B, H, dv)
        alpha = torch.sigmoid(torch.randn(B, H, dk)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, dk))
        w = torch.sigmoid(torch.randn(B, H, dv))
        state = torch.zeros(B, H, dk, dv)

        out, new_state = dispatch_dgda_step(q, k, v, alpha, b, w, state)
        assert out.shape == (B, H, dv)
        assert new_state.shape == (B, H, dk, dv)
        assert torch.isfinite(out).all()

    def test_fallback_on_simulated_kernel_exception_centroids(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        def failing_centroids(**kwargs):
            raise RuntimeError("Kernel compilation failure")

        register_kernel("triton", "compute_centroids")(failing_centroids)

        B, L, d_idx = 2, 64, 32
        k_idx = torch.randn(B, L, d_idx)
        c = dispatch_compute_centroids(k_idx, block_size=64)
        ref_c = reference_compute_centroids(k_idx, block_size=64)
        assert torch.equal(c, ref_c)

    def test_fallback_on_simulated_kernel_exception_topk(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        def failing_topk(**kwargs):
            raise RuntimeError("TopK GPU kernel assertion error")

        register_kernel("triton", "index_topk")(failing_topk)

        B, L, nb, d_idx = 1, 64, 2, 16
        q_idx = torch.randn(B, L, d_idx)
        centroids = torch.randn(B, nb, d_idx)
        idx = dispatch_index_topk(q_idx, centroids, top_k=2, block_size=64)
        ref_idx = reference_index_topk(q_idx, centroids, top_k=2, block_size=64)
        assert torch.equal(idx, ref_idx)

    def test_fallback_on_simulated_kernel_exception_superposition(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        def failing_superposition(**kwargs):
            raise RuntimeError("Superposition launch bounds exceeded")

        register_kernel("triton", "stream_superposition")(failing_superposition)

        B, H, L, D = 1, 2, 4, 8
        ol = torch.randn(B, H, L, D)
        os = torch.randn(B, H, L, D)
        oh = torch.randn(B, H, L, D)
        logits = torch.randn(B, L, 3)
        out = dispatch_stream_superposition(ol, os, oh, logits)
        ref_out = reference_stream_superposition(ol, os, oh, logits)
        assert torch.equal(out, ref_out)

    def test_fallback_warning_logged_and_runtime_warning_emitted(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        def failing_prefill(**kwargs):
            raise RuntimeError("Simulated GPU crash")

        register_kernel("triton", "dgda_prefill")(failing_prefill)

        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk))
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            dispatch_dgda_prefill(q, k, v, alpha, b, w)

            fallback_warnings = [
                w for w in recorded_warnings if issubclass(w.category, RuntimeWarning)
            ]
            assert len(fallback_warnings) >= 1
            msg = str(fallback_warnings[0].message)
            assert "triton" in msg
            assert "dgda_prefill" in msg
            assert "falling back to reference" in msg.lower()


class TestDispatcherNumericalParityExactZero:
    def test_numerical_parity_dgda_prefill_fallback_vs_reference_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        B, H, L, dk, dv = 2, 2, 8, 16, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)) * 0.95
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        out_fb, state_fb = dispatch_dgda_prefill(q, k, v, alpha, b, w)
        out_ref, state_ref = reference_dgda_prefill(q, k, v, alpha, b, w)

        assert torch.equal(out_fb, out_ref), f"Max diff out: {(out_fb - out_ref).abs().max()}"
        assert torch.equal(state_fb, state_ref), f"Max diff state: {(state_fb - state_ref).abs().max()}"

    def test_numerical_parity_dgda_step_fallback_vs_reference_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        B, H, dk, dv = 2, 4, 16, 16
        q = torch.randn(B, H, dk)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1)
        v = torch.randn(B, H, dv)
        alpha = torch.sigmoid(torch.randn(B, H, dk)) * 0.95
        b = torch.sigmoid(torch.randn(B, H, dk))
        w = torch.sigmoid(torch.randn(B, H, dv))
        state = torch.randn(B, H, dk, dv)

        out_fb, state_fb = dispatch_dgda_step(q, k, v, alpha, b, w, state)
        out_ref, state_ref = reference_dgda_step(q, k, v, alpha, b, w, state)

        assert torch.equal(out_fb, out_ref)
        assert torch.equal(state_fb, state_ref)

    def test_numerical_parity_compute_centroids_fallback_vs_reference_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        for L in [64, 65, 128, 255]:
            k_idx = torch.randn(2, L, 64)
            c_fb = dispatch_compute_centroids(k_idx, block_size=64)
            c_ref = reference_compute_centroids(k_idx, block_size=64)
            assert torch.equal(c_fb, c_ref), f"Mismatch at L={L}"

    def test_numerical_parity_index_topk_fallback_vs_reference_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        B, L, nb, d_idx = 2, 128, 4, 32
        q_idx = torch.randn(B, L, d_idx)
        centroids = torch.randn(B, nb, d_idx)

        idx_fb = dispatch_index_topk(q_idx, centroids, top_k=2, block_size=64)
        idx_ref = reference_index_topk(q_idx, centroids, top_k=2, block_size=64)
        assert torch.equal(idx_fb, idx_ref)

    def test_numerical_parity_stream_superposition_fallback_vs_reference_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        B, H, L, D = 2, 4, 16, 32
        ol = torch.randn(B, H, L, D)
        os = torch.randn(B, H, L, D)
        oh = torch.randn(B, H, L, D)
        logits = torch.randn(B, L, 3)

        out_fb = dispatch_stream_superposition(ol, os, oh, logits)
        out_ref = reference_stream_superposition(ol, os, oh, logits)
        assert torch.equal(out_fb, out_ref)

    def test_numerical_parity_across_fp32_fp16_bf16_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            B, H, L, dk, dv = 1, 2, 4, 8, 8
            q = torch.randn(B, H, L, dk, dtype=dt)
            k = F.normalize(torch.randn(B, H, L, dk, dtype=dt), p=2, dim=-1)
            v = torch.randn(B, H, L, dv, dtype=dt)
            alpha = torch.sigmoid(torch.randn(B, H, L, dk, dtype=dt)) * 0.9
            b = torch.sigmoid(torch.randn(B, H, L, dk, dtype=dt))
            w = torch.sigmoid(torch.randn(B, H, L, dv, dtype=dt))

            out_fb, state_fb = dispatch_dgda_prefill(q, k, v, alpha, b, w)
            out_ref, state_ref = reference_dgda_prefill(q, k, v, alpha, b, w)
            assert torch.equal(out_fb, out_ref), f"Failed for {dt}"
            assert torch.equal(state_fb, state_ref), f"Failed for {dt}"


class TestDispatcherAutogradGradients:
    def test_autograd_dgda_prefill_finite_nonzero_zero_nan(self):
        B, H, L, dk, dv = 2, 2, 8, 8, 8
        q = torch.randn(B, H, L, dk, requires_grad=True)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, H, L, dv, requires_grad=True)
        alpha = (torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, H, L, dv)).requires_grad_(True)
        init_state = torch.randn(B, H, dk, dv, requires_grad=True)

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w, initial_state=init_state)
        loss = out.sum() + state.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w), ("init_state", init_state)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(t.grad).any(), f"NaN in {name}.grad"
            assert not torch.isinf(t.grad).any(), f"Inf in {name}.grad"
            assert (t.grad.abs() > 0.0).any(), f"{name}.grad is all zeros"
            assert t.grad.shape == t.shape, f"Shape mismatch in {name}.grad"

    def test_autograd_dgda_step_finite_nonzero_zero_nan(self):
        B, H, dk, dv = 1, 2, 8, 8
        q = torch.randn(B, H, dk, requires_grad=True)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, H, dv, requires_grad=True)
        alpha = (torch.sigmoid(torch.randn(B, H, dk)) * 0.9).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, H, dk)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, H, dv)).requires_grad_(True)
        state = torch.randn(B, H, dk, dv, requires_grad=True)

        out, new_state = dispatch_dgda_step(q, k, v, alpha, b, w, state)
        loss = out.sum() + new_state.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w), ("state", state)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(t.grad).any()
            assert not torch.isinf(t.grad).any()
            assert (t.grad.abs() > 0.0).any()

    def test_autograd_compute_centroids_strictly_positive_gradient(self):
        B, L, d_idx = 2, 64, 16
        k_idx = torch.randn(B, L, d_idx, requires_grad=True)
        c = dispatch_compute_centroids(k_idx, block_size=64)
        c.sum().backward()

        assert k_idx.grad is not None
        assert not torch.isnan(k_idx.grad).any()
        assert not torch.isinf(k_idx.grad).any()
        assert (k_idx.grad[:, :L] > 0.0).all()

    def test_autograd_index_topk_discrete_index_compatibility(self):
        B, L, nb, d_idx = 1, 32, 2, 8
        q_idx = torch.randn(B, L, d_idx, requires_grad=True)
        centroids = torch.randn(B, nb, d_idx, requires_grad=True)

        idx = dispatch_index_topk(q_idx, centroids, top_k=2, block_size=16)
        assert idx.dtype == torch.long
        assert not idx.requires_grad

        _, scores = dispatch_index_topk(q_idx, centroids, top_k=2, block_size=16, return_scores=True)
        scores[torch.isfinite(scores)].sum().backward()
        assert q_idx.grad is not None
        assert centroids.grad is not None
        assert not torch.isnan(q_idx.grad).any()

    def test_autograd_stream_superposition_analytical_gate_gradients(self):
        B, H, L, D = 1, 2, 4, 8
        ol = torch.randn(B, H, L, D, requires_grad=True)
        os = torch.randn(B, H, L, D, requires_grad=True)
        oh = torch.randn(B, H, L, D, requires_grad=True)
        logits = torch.randn(B, L, 3, requires_grad=True)

        out = dispatch_stream_superposition(ol, os, oh, logits)
        out.sum().backward()

        for name, t in [("ol", ol), ("os", os), ("oh", oh), ("logits", logits)]:
            assert t.grad is not None
            assert not torch.isnan(t.grad).any()
            assert not torch.isinf(t.grad).any()
            assert (t.grad.abs() > 0.0).any()

        probs = F.softmax(logits.detach(), dim=-1)
        expected_gl = probs[:, :, 0:1].unsqueeze(1).expand_as(ol)
        assert torch.allclose(ol.grad, expected_gl, atol=1e-5)

    def test_autograd_fallback_vs_reference_gradient_parity_exact_zero(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "reference")

        B, H, L, dk, dv = 1, 1, 4, 4, 4
        torch.manual_seed(10)
        q1 = torch.randn(B, H, L, dk, requires_grad=True)
        k1 = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1).requires_grad_(True)
        v1 = torch.randn(B, H, L, dv, requires_grad=True)
        alpha1 = (torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9).requires_grad_(True)
        b1 = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        w1 = torch.sigmoid(torch.randn(B, H, L, dv)).requires_grad_(True)

        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)
        alpha2 = alpha1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)

        out_fb, state_fb = dispatch_dgda_prefill(q1, k1, v1, alpha1, b1, w1)
        (out_fb.sum() + state_fb.sum()).backward()

        out_ref, state_ref = reference_dgda_prefill(q2, k2, v2, alpha2, b2, w2)
        (out_ref.sum() + state_ref.sum()).backward()

        assert torch.equal(q1.grad, q2.grad)
        assert torch.equal(k1.grad, k2.grad)
        assert torch.equal(v1.grad, v2.grad)
        assert torch.equal(alpha1.grad, alpha2.grad)
        assert torch.equal(b1.grad, b2.grad)
        assert torch.equal(w1.grad, w2.grad)


class TestDispatcherInputValidation:
    def test_input_validation_device_mismatch_raises_runtime_error(self):
        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk, device="cpu")
        k = torch.randn(B, H, L, dk, device="meta")
        v = torch.randn(B, H, L, dv, device="cpu")
        alpha = torch.sigmoid(torch.randn(B, H, L, dk, device="cpu"))
        b = torch.sigmoid(torch.randn(B, H, L, dk, device="cpu"))
        w = torch.sigmoid(torch.randn(B, H, L, dv, device="cpu"))

        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            dispatch_dgda_prefill(q, k, v, alpha, b, w)

    def test_input_validation_dtype_mismatch_raises_type_error(self):
        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk, dtype=torch.float32)
        k = torch.randn(B, H, L, dk, dtype=torch.float16)
        v = torch.randn(B, H, L, dv, dtype=torch.float32)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk, dtype=torch.float32))
        b = torch.sigmoid(torch.randn(B, H, L, dk, dtype=torch.float32))
        w = torch.sigmoid(torch.randn(B, H, L, dv, dtype=torch.float32))

        with pytest.raises(TypeError, match="Tensor dtypes must match"):
            dispatch_dgda_prefill(q, k, v, alpha, b, w)

    def test_input_validation_empty_sequence_L0_returns_empty_tensors(self):
        B, H, dk, dv = 2, 4, 16, 16
        q = torch.empty(B, H, 0, dk)
        k = torch.empty(B, H, 0, dk)
        v = torch.empty(B, H, 0, dv)
        alpha = torch.empty(B, H, 0, dk)
        b = torch.empty(B, H, 0, dk)
        w = torch.empty(B, H, 0, dv)
        init_state = torch.ones(B, H, dk, dv)

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w, initial_state=init_state)
        assert out.shape == (B, H, 0, dv)
        assert state.shape == (B, H, dk, dv)
        assert torch.equal(state, init_state)

    def test_input_validation_partial_chunk_L_less_than_C_succeeds(self):
        B, H, L, dk, dv = 2, 2, 7, 16, 16
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk))
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w, chunk_size=16)
        assert out.shape == (B, H, 7, dv)
        assert state.shape == (B, H, dk, dv)
        assert torch.isfinite(out).all()
        assert torch.isfinite(state).all()

    def test_input_validation_uninitialized_state_handled_gracefully(self):
        B, H, dk, dv = 1, 2, 8, 8
        q = torch.randn(B, H, dk)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1)
        v = torch.randn(B, H, dv)
        alpha = torch.sigmoid(torch.randn(B, H, dk))
        b = torch.sigmoid(torch.randn(B, H, dk))
        w = torch.sigmoid(torch.randn(B, H, dv))

        out, new_state = dispatch_dgda_step(q, k, v, alpha, b, w, state=None)
        assert out.shape == (B, H, dv)
        assert new_state.shape == (B, H, dk, dv)
        assert torch.isfinite(out).all()
        assert torch.isfinite(new_state).all()
