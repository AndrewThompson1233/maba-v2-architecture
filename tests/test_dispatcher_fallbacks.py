
import os
import threading
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


class CustomCUDARuntimeError(RuntimeError):
    pass


class KernelCorruptedException(Exception):
    pass


@pytest.fixture(autouse=True)
def isolate_chaos_environment(monkeypatch):
    monkeypatch.delenv("MABA_BACKEND", raising=False)
    monkeypatch.delenv("MABA_STRICT_BACKEND", raising=False)
    monkeypatch.delenv("MABA_VERBOSE", raising=False)
    clear_fallback_warnings()

    from maba_sparse.kernels.cpu_dgda import cpu_dgda_prefill as orig_cpu_prefill, cpu_dgda_step as orig_cpu_step
    register_kernel("cpu", "dgda_prefill")(orig_cpu_prefill)
    register_kernel("cpu", "dgda_step")(orig_cpu_step)
    register_kernel("cpu", "compute_centroids")(reference_compute_centroids)
    register_kernel("cpu", "index_topk")(reference_index_topk)
    register_kernel("cpu", "stream_superposition")(reference_stream_superposition)
    clear_fallback_warnings()
    yield
    register_kernel("cpu", "dgda_prefill")(orig_cpu_prefill)
    register_kernel("cpu", "dgda_step")(orig_cpu_step)
    register_kernel("cpu", "compute_centroids")(reference_compute_centroids)
    register_kernel("cpu", "index_topk")(reference_index_topk)
    register_kernel("cpu", "stream_superposition")(reference_stream_superposition)
    clear_fallback_warnings()


class TestExceptionInjection:

    @pytest.mark.parametrize("exc_class,exc_args", [
        (ZeroDivisionError, ("division by zero in CUDA threadblock",)),
        (MemoryError, ("CUDA out of memory: tried to allocate 16.00 GiB",)),
        (CustomCUDARuntimeError, ("CUDA error: an illegal memory access was encountered",)),
        (KernelCorruptedException, ("Hardware parity fault in SRAM",)),
        (FloatingPointError, ("Denormal/NaN trap triggered in accumulator",)),
    ])
    def test_prefill_exception_injection_fallback_exact_parity(
        self, monkeypatch, exc_class, exc_args
    ):
        monkeypatch.setenv("MABA_BACKEND", "cpu")

        def broken_prefill(**kwargs):
            raise exc_class(*exc_args)

        register_kernel("cpu", "dgda_prefill")(broken_prefill)

        B, H, L, dk, dv = 2, 2, 17, 16, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))
        init_s = torch.randn(B, H, dk, dv)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            out_fb, state_fb = dispatch_dgda_prefill(q, k, v, alpha, b, w, initial_state=init_s)

            fb_warnings = [w for w in recorded_warnings if issubclass(w.category, RuntimeWarning)]
            assert len(fb_warnings) >= 1, f"Expected RuntimeWarning for {exc_class.__name__}"
            warn_msg = str(fb_warnings[0].message)
            assert "dgda_prefill" in warn_msg
            assert "falling back to reference" in warn_msg.lower()
            assert exc_class.__name__ in warn_msg

        out_ref, state_ref = reference_dgda_prefill(q, k, v, alpha, b, w, initial_state=init_s)
        assert torch.equal(out_fb, out_ref), "Fallback prefill output deviated from reference"
        assert torch.equal(state_fb, state_ref), "Fallback prefill state deviated from reference"

    @pytest.mark.parametrize("exc_class", [
        ZeroDivisionError,
        MemoryError,
        CustomCUDARuntimeError,
        KernelCorruptedException,
    ])
    def test_step_exception_injection_fallback_exact_parity(self, monkeypatch, exc_class):
        monkeypatch.setenv("MABA_BACKEND", "cpu")

        def broken_step(**kwargs):
            raise exc_class("Step transition crash")

        register_kernel("cpu", "dgda_step")(broken_step)

        B, H, dk, dv = 2, 4, 16, 16
        q = torch.randn(B, H, dk)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1)
        v = torch.randn(B, H, dv)
        alpha = torch.sigmoid(torch.randn(B, H, dk)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, dk))
        w = torch.sigmoid(torch.randn(B, H, dv))
        s = torch.randn(B, H, dk, dv)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            out_fb, s_fb = dispatch_dgda_step(q, k, v, alpha, b, w, state=s)
            assert any("dgda_step" in str(w.message) for w in recorded_warnings)

        out_ref, s_ref = reference_dgda_step(q, k, v, alpha, b, w, state=s)
        assert torch.equal(out_fb, out_ref)
        assert torch.equal(s_fb, s_ref)

    @pytest.mark.parametrize("exc_class", [ZeroDivisionError, MemoryError, CustomCUDARuntimeError])
    def test_indexer_and_superposition_exception_injection_fallback(self, monkeypatch, exc_class):
        monkeypatch.setenv("MABA_BACKEND", "triton")

        register_kernel("triton", "compute_centroids")(lambda **kw: (_ for _ in ()).throw(exc_class("Centroid crash")))
        register_kernel("triton", "index_topk")(lambda **kw: (_ for _ in ()).throw(exc_class("Topk crash")))
        register_kernel("triton", "stream_superposition")(lambda **kw: (_ for _ in ()).throw(exc_class("Superpos crash")))

        k_idx = torch.randn(2, 65, 32)
        c_fb = dispatch_compute_centroids(k_idx, block_size=64)
        c_ref = reference_compute_centroids(k_idx, block_size=64)
        assert torch.equal(c_fb, c_ref)

        q_idx = torch.randn(2, 65, 32)
        idx_fb = dispatch_index_topk(q_idx, c_fb, top_k=2, block_size=64)
        idx_ref = reference_index_topk(q_idx, c_fb, top_k=2, block_size=64)
        assert torch.equal(idx_fb, idx_ref)

        ol, os_t, oh = torch.randn(2, 2, 8, 16), torch.randn(2, 2, 8, 16), torch.randn(2, 2, 8, 16)
        logits = torch.randn(2, 8, 3)
        sup_fb = dispatch_stream_superposition(ol, os_t, oh, logits)
        sup_ref = reference_stream_superposition(ol, os_t, oh, logits)
        assert torch.equal(sup_fb, sup_ref)

    def test_strict_mode_escalates_fallback_to_runtime_error(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "cpu")
        monkeypatch.setenv("MABA_STRICT_BACKEND", "1")

        register_kernel("cpu", "dgda_step")(lambda **kw: (_ for _ in ()).throw(CustomCUDARuntimeError("Fatal GPU lockup")))

        B, H, dk, dv = 1, 1, 8, 8
        q = torch.randn(B, H, dk)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1)
        v = torch.randn(B, H, dv)
        alpha = torch.sigmoid(torch.randn(B, H, dk))
        b = torch.sigmoid(torch.randn(B, H, dk))
        w = torch.sigmoid(torch.randn(B, H, dv))

        with pytest.raises(RuntimeError, match=r"\[Maba Dispatcher Fallback\].*CustomCUDARuntimeError"):
            dispatch_dgda_step(q, k, v, alpha, b, w)


class TestDynamicBackendSwitching:

    def test_rapid_backend_toggle_loop_100_iterations(self, monkeypatch):
        backends_cycle = ["reference", "cpu", "auto", "ref", "pytorch", "openmp"]

        B, H, L, dk, dv = 1, 2, 8, 16, 16
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        ref_out, ref_state = reference_dgda_prefill(q, k, v, alpha, b, w)

        for i in range(100):
            chosen_backend = backends_cycle[i % len(backends_cycle)]
            monkeypatch.setenv("MABA_BACKEND", chosen_backend)

            resolved = get_backend(q.device)
            if chosen_backend in ("reference", "ref", "pytorch"):
                assert resolved == "reference"
            elif chosen_backend in ("cpu", "openmp"):
                assert resolved == "cpu"
            elif chosen_backend == "auto":
                assert resolved in ("cpu", "reference")

            out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w)
            assert out.shape == ref_out.shape
            assert state.shape == ref_state.shape
            assert torch.isfinite(out).all()
            assert torch.isfinite(state).all()

            diff = (out - ref_out).abs().max().item()
            assert diff < 1e-3, f"Iteration {i} with backend {chosen_backend} diff {diff:.6e} >= 1e-3"

    @pytest.mark.parametrize("invalid_backend", [
        "invalid_accelerator",
        "rocm",
        "mps",
        "none",
        "null",
        "12345",
        "UNKNOWN",
        "tpu_v5",
    ])
    def test_invalid_maba_backend_strictly_rejected(self, monkeypatch, invalid_backend):
        monkeypatch.setenv("MABA_BACKEND", invalid_backend)

        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            get_backend(torch.device("cpu"))

        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk)
        k = torch.randn(B, H, L, dk)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk))
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            dispatch_dgda_prefill(q, k, v, alpha, b, w)

        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            dispatch_dgda_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], alpha[:, :, 0], b[:, :, 0], w[:, :, 0])

        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            dispatch_compute_centroids(torch.randn(1, 16, 8))

        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            dispatch_index_topk(torch.randn(1, 16, 8), torch.randn(1, 2, 8))

        with pytest.raises(ValueError, match="Unsupported MABA_BACKEND"):
            dispatch_stream_superposition(q, q, q, torch.randn(1, 4, 3))

    def test_concurrent_multithreaded_backend_query(self, monkeypatch):
        stop_event = threading.Event()
        errors = []

        def worker_loop(thread_id: int):
            try:
                B, H, dk, dv = 1, 2, 8, 8
                q = torch.randn(B, H, dk)
                k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1)
                v = torch.randn(B, H, dv)
                alpha = torch.sigmoid(torch.randn(B, H, dk))
                b = torch.sigmoid(torch.randn(B, H, dk))
                w = torch.sigmoid(torch.randn(B, H, dv))
                s = torch.zeros(B, H, dk, dv)

                for _ in range(50):
                    if stop_event.is_set():
                        break
                    backend = get_backend(q.device)
                    assert backend in ("reference", "cpu")
                    out, s_next = dispatch_dgda_step(q, k, v, alpha, b, w, state=s)
                    assert out.shape == (B, H, dv)
            except Exception as e:
                errors.append(e)

        orig_backend = os.environ.get("MABA_BACKEND")
        try:
            threads = [threading.Thread(target=worker_loop, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()

            for b in ["reference", "cpu", "auto", "ref"] * 5:
                os.environ["MABA_BACKEND"] = b

            stop_event.set()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Encountered thread execution errors: {errors}"
        finally:
            if orig_backend is None:
                os.environ.pop("MABA_BACKEND", None)
            else:
                os.environ["MABA_BACKEND"] = orig_backend


class TestExtremeShapesAndDtypes:

    @pytest.mark.parametrize("L", [1, 3, 17, 33, 65, 129, 0, 7, 31, 127, 255, 513])
    def test_prefill_odd_and_non_power_of_two_sequence_lengths(self, L: int):
        B, H, dk, dv = 2, 2, 16, 16
        torch.manual_seed(100 + L)
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)) * 0.95
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w)
        assert out.shape == (B, H, L, dv)
        assert state.shape == (B, H, dk, dv)
        assert torch.isfinite(out).all()
        assert torch.isfinite(state).all()

        ref_out, ref_state = reference_dgda_prefill(q, k, v, alpha, b, w)
        if L > 0:
            diff_out = (out - ref_out).abs().max().item()
            diff_state = (state - ref_state).abs().max().item()
            assert diff_out < 1e-3, f"L={L}: prefill out diff {diff_out:.6e} >= 1e-3"
            assert diff_state < 1e-3, f"L={L}: prefill state diff {diff_state:.6e} >= 1e-3"
        else:
            assert out.numel() == 0
            assert torch.equal(state, ref_state)

    @pytest.mark.parametrize("L", [1, 3, 17, 33, 65, 129])
    def test_centroid_and_topk_odd_lengths_with_sub_topk_blocks(self, L: int):
        B, d_idx = 2, 32
        block_size = 64
        top_k = 32

        k_idx = torch.randn(B, L, d_idx)
        q_idx = torch.randn(B, L, d_idx)

        centroids = dispatch_compute_centroids(k_idx, block_size=block_size)
        expected_nb = (L + block_size - 1) // block_size
        assert centroids.shape == (B, expected_nb, d_idx)
        assert torch.isfinite(centroids).all()

        indices = dispatch_index_topk(q_idx, centroids, top_k=top_k, block_size=block_size)
        expected_ak = min(top_k, expected_nb)
        assert indices.shape == (B, L, expected_ak)
        assert indices.dtype == torch.long
        assert (indices >= 0).all()
        assert (indices < expected_nb).all()

    @pytest.mark.parametrize("B,H,dk,dv", [
        (16, 1, 32, 64),
        (32, 2, 64, 32),
        (64, 4, 16, 16),
    ])
    def test_large_batch_and_asymmetric_dimensions(self, B: int, H: int, dk: int, dv: int):
        L = 16
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

    def test_uninitialized_state_strict_equivalence_with_zeros(self):
        B, H, L, dk, dv = 2, 2, 17, 16, 16
        torch.manual_seed(999)
        q = torch.randn(B, H, L, dk)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1)
        v = torch.randn(B, H, L, dv)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, L, dk))
        w = torch.sigmoid(torch.randn(B, H, L, dv))

        zero_state = torch.zeros(B, H, dk, dv)
        out_none, s_none = dispatch_dgda_prefill(q, k, v, alpha, b, w, initial_state=None)
        out_zero, s_zero = dispatch_dgda_prefill(q, k, v, alpha, b, w, initial_state=zero_state)

        assert torch.equal(out_none, out_zero)
        assert torch.equal(s_none, s_zero)

        out_s_none, ns_none = dispatch_dgda_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], alpha[:, :, 0], b[:, :, 0], w[:, :, 0], state=None)
        out_s_zero, ns_zero = dispatch_dgda_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], alpha[:, :, 0], b[:, :, 0], w[:, :, 0], state=zero_state)
        assert torch.equal(out_s_none, out_s_zero)
        assert torch.equal(ns_none, ns_zero)

    @pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
    def test_all_supported_floating_point_dtypes(self, dt: torch.dtype):
        B, H, L, dk, dv = 1, 2, 8, 8, 8
        q = torch.randn(B, H, L, dk, dtype=dt)
        k = F.normalize(torch.randn(B, H, L, dk, dtype=dt), p=2, dim=-1)
        v = torch.randn(B, H, L, dv, dtype=dt)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk, dtype=dt)) * 0.9
        b = torch.sigmoid(torch.randn(B, H, L, dk, dtype=dt))
        w = torch.sigmoid(torch.randn(B, H, L, dv, dtype=dt))

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w)
        assert out.dtype == dt
        assert state.dtype == dt
        assert torch.isfinite(out).all()
        assert torch.isfinite(state).all()

    def test_mixed_dtypes_strictly_rejected_with_type_error(self):
        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk, dtype=torch.float32)
        k_fp16 = torch.randn(B, H, L, dk, dtype=torch.float16)
        v = torch.randn(B, H, L, dv, dtype=torch.float32)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk, dtype=torch.float32))
        b = torch.sigmoid(torch.randn(B, H, L, dk, dtype=torch.float32))
        w = torch.sigmoid(torch.randn(B, H, L, dv, dtype=torch.float32))
        s_bf16 = torch.randn(B, H, dk, dv, dtype=torch.bfloat16)

        with pytest.raises(TypeError, match="Tensor dtypes must match"):
            dispatch_dgda_prefill(q, k_fp16, v, alpha, b, w)

        with pytest.raises(TypeError, match="Tensor dtypes must match"):
            dispatch_dgda_prefill(q, q, v, alpha, b, w, initial_state=s_bf16)

        with pytest.raises(TypeError, match="Tensor dtypes must match"):
            dispatch_dgda_step(q[:, :, 0], q[:, :, 0], v[:, :, 0], alpha[:, :, 0], b[:, :, 0], w[:, :, 0], state=s_bf16)

    def test_non_floating_point_dtypes_rejected(self):
        q_int = torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32)
        k_int = torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32)
        v_int = torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32)
        alpha_int = torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32)
        b_int = torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32)
        w_int = torch.randint(0, 10, (1, 1, 4, 4), dtype=torch.int32)

        with pytest.raises(TypeError, match="Unsupported dtype"):
            dispatch_dgda_prefill(q_int, k_int, v_int, alpha_int, b_int, w_int)


class TestAutogradGraphUnbrokenness:

    @pytest.mark.parametrize("exc_class", [
        ZeroDivisionError,
        MemoryError,
        CustomCUDARuntimeError,
        KernelCorruptedException,
    ])
    def test_prefill_all_7_inputs_receive_unbroken_gradients_under_fallback(
        self, monkeypatch, exc_class
    ):
        monkeypatch.setenv("MABA_BACKEND", "cpu")

        def broken_prefill(q, k, v, alpha, b, w, **kw):
            raise exc_class("Prefill crash during kernel execution")

        register_kernel("cpu", "dgda_prefill")(broken_prefill)

        B, H, L, dk, dv = 2, 2, 8, 8, 8
        torch.manual_seed(777)
        q = torch.randn(B, H, L, dk, requires_grad=True)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, H, L, dv, requires_grad=True)
        alpha = (torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, H, L, dv)).requires_grad_(True)
        init_s = torch.randn(B, H, dk, dv, requires_grad=True)

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w, initial_state=init_s)
        loss = out.sum() + state.sum()
        loss.backward()

        for name, t in [
            ("q", q), ("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w), ("init_s", init_s)
        ]:
            assert t.grad is not None, f"Autograd graph was severed: {name}.grad is None!"
            assert not torch.isnan(t.grad).any(), f"NaN in {name}.grad"
            assert not torch.isinf(t.grad).any(), f"Inf in {name}.grad"
            assert (t.grad.abs() > 0.0).any(), f"Gradient for {name} is all zeros!"
            assert t.grad.shape == t.shape

    def test_prefill_fallback_gradients_match_reference_bitwise(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "cpu")

        register_kernel("cpu", "dgda_prefill")(
            lambda **kw: (_ for _ in ()).throw(CustomCUDARuntimeError("Fail"))
        )

        B, H, L, dk, dv = 1, 1, 4, 4, 4
        torch.manual_seed(888)

        q_fb = torch.randn(B, H, L, dk, requires_grad=True)
        k_fb = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1).requires_grad_(True)
        v_fb = torch.randn(B, H, L, dv, requires_grad=True)
        alpha_fb = (torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9).requires_grad_(True)
        b_fb = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        w_fb = torch.sigmoid(torch.randn(B, H, L, dv)).requires_grad_(True)
        s_fb = torch.randn(B, H, dk, dv, requires_grad=True)

        q_ref = q_fb.detach().clone().requires_grad_(True)
        k_ref = k_fb.detach().clone().requires_grad_(True)
        v_ref = v_fb.detach().clone().requires_grad_(True)
        alpha_ref = alpha_fb.detach().clone().requires_grad_(True)
        b_ref = b_fb.detach().clone().requires_grad_(True)
        w_ref = w_fb.detach().clone().requires_grad_(True)
        s_ref = s_fb.detach().clone().requires_grad_(True)

        out_f, state_f = dispatch_dgda_prefill(q_fb, k_fb, v_fb, alpha_fb, b_fb, w_fb, initial_state=s_fb)
        (out_f.sum() + state_f.sum()).backward()

        out_r, state_r = reference_dgda_prefill(q_ref, k_ref, v_ref, alpha_ref, b_ref, w_ref, initial_state=s_ref)
        (out_r.sum() + state_r.sum()).backward()

        assert torch.equal(q_fb.grad, q_ref.grad)
        assert torch.equal(k_fb.grad, k_ref.grad)
        assert torch.equal(v_fb.grad, v_ref.grad)
        assert torch.equal(alpha_fb.grad, alpha_ref.grad)
        assert torch.equal(b_fb.grad, b_ref.grad)
        assert torch.equal(w_fb.grad, w_ref.grad)
        assert torch.equal(s_fb.grad, s_ref.grad)

    def test_partial_forward_graph_construction_before_crash_resilience(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "cpu")

        def partially_executed_kernel(q, k, v, alpha, b, w, **kwargs):
            dummy_1 = (q * 3.14159).sum()
            dummy_2 = (k.unsqueeze(-1) * v.unsqueeze(-2)).sum()
            dummy_3 = (alpha * b).mean()
            if dummy_1.item() > -1e9:
                raise CustomCUDARuntimeError("Crash after intermediate ops attached to autograd graph")

        register_kernel("cpu", "dgda_prefill")(partially_executed_kernel)

        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk, requires_grad=True)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, H, L, dv, requires_grad=True)
        alpha = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, H, L, dv)).requires_grad_(True)

        out, state = dispatch_dgda_prefill(q, k, v, alpha, b, w)
        loss = out.sum() + state.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w)]:
            assert t.grad is not None
            assert not torch.isnan(t.grad).any()
            assert not torch.isinf(t.grad).any()
            assert (t.grad.abs() > 0.0).any()

    def test_step_decode_all_inputs_unbroken_autograd_under_fallback(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "cpu")
        register_kernel("cpu", "dgda_step")(
            lambda **kw: (_ for _ in ()).throw(ZeroDivisionError("Decode step div0"))
        )

        B, H, dk, dv = 2, 2, 8, 8
        q = torch.randn(B, H, dk, requires_grad=True)
        k = F.normalize(torch.randn(B, H, dk), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, H, dv, requires_grad=True)
        alpha = (torch.sigmoid(torch.randn(B, H, dk)) * 0.9).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, H, dk)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, H, dv)).requires_grad_(True)
        s = torch.randn(B, H, dk, dv, requires_grad=True)

        out, ns = dispatch_dgda_step(q, k, v, alpha, b, w, state=s)
        loss = out.sum() + ns.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w), ("s", s)]:
            assert t.grad is not None
            assert not torch.isnan(t.grad).any()
            assert (t.grad.abs() > 0.0).any()

    def test_double_backward_higher_order_gradient_continuity(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "cpu")
        register_kernel("cpu", "dgda_prefill")(
            lambda **kw: (_ for _ in ()).throw(CustomCUDARuntimeError("Double backward test crash"))
        )

        B, H, L, dk, dv = 1, 1, 4, 4, 4
        q = torch.randn(B, H, L, dk, requires_grad=True)
        k = F.normalize(torch.randn(B, H, L, dk), p=2, dim=-1).requires_grad_(True)
        v = torch.randn(B, H, L, dv, requires_grad=True)
        alpha = (torch.sigmoid(torch.randn(B, H, L, dk)) * 0.9).requires_grad_(True)
        b = torch.sigmoid(torch.randn(B, H, L, dk)).requires_grad_(True)
        w = torch.sigmoid(torch.randn(B, H, L, dv)).requires_grad_(True)

        out, _ = dispatch_dgda_prefill(q, k, v, alpha, b, w)
        loss = (out ** 2).sum()

        grads = torch.autograd.grad(loss, q, create_graph=True)[0]
        assert grads is not None
        assert grads.shape == q.shape

        second_order_loss = grads.sum()
        second_order_loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()
        assert (q.grad.abs() > 0.0).any()
