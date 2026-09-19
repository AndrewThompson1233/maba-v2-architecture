
import pytest
import torch

from maba_sparse.kernels.common import (
    ref_dgda_prefill,
    ref_dgda_step,
)
from maba_sparse.kernels.dispatcher import (
    clear_fallback_warnings,
    dispatch_dgda_prefill,
    is_cuda_sm75_available,
    is_triton_available,
    register_kernel,
)
from maba_sparse.kernels.triton_dgda import (
    TRITON_AVAILABLE,
    triton_dgda_prefill,
    triton_dgda_step,
)

CUDA_AVAILABLE = torch.cuda.is_available() and is_cuda_sm75_available() and is_triton_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def _make_tensors(
    B: int = 2,
    H: int = 4,
    L: int = 32,
    dk: int = 64,
    dv: int = 64,
    device: str = DEVICE,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    s0: bool = False,
):
    torch.manual_seed(42)
    q = torch.randn(B, H, L, dk, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, L, dk, device=device, dtype=dtype)
    k = (k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + 1e-6)).requires_grad_(requires_grad)
    v = torch.randn(B, H, L, dv, device=device, dtype=dtype, requires_grad=requires_grad)
    alpha = (torch.rand(B, H, L, dk, device=device, dtype=dtype) * 0.9 + 0.05).requires_grad_(requires_grad)
    b = torch.rand(B, H, L, dk, device=device, dtype=dtype, requires_grad=requires_grad)
    w = torch.rand(B, H, L, dv, device=device, dtype=dtype, requires_grad=requires_grad)
    init_state = None
    if s0:
        init_state = torch.randn(B, H, dk, dv, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v, alpha, b, w, init_state



class TestTritonEnvironment:

    def test_cuda_sm75_requirement(self):
        if not torch.cuda.is_available():
            assert not is_cuda_sm75_available()
        else:
            cap = torch.cuda.get_device_capability(0)
            assert is_cuda_sm75_available() == (cap >= (7, 5))

    def test_triton_availability_flag(self):
        assert is_triton_available() == TRITON_AVAILABLE



class TestTritonDGDAPrefillNumericalParity:

    @pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
    def test_prefill_fp32_multi_lengths(self, L: int):
        q, k, v, alpha, b, w, s0 = _make_tensors(L=L, s0=(L in (32, 128)))
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        diff_out = (out - ref_out).abs().max().item()
        diff_state = (state - ref_state).abs().max().item()

        assert diff_out < 1e-4, f"Output diff {diff_out} exceeded 1e-4 at L={L}"
        assert diff_state < 1e-4, f"State diff {diff_state} exceeded 1e-4 at L={L}"

    @pytest.mark.parametrize("L", [16, 64, 128])
    def test_prefill_fp16_multi_lengths(self, L: int):
        if not CUDA_AVAILABLE:
            pytest.skip("FP16 hardware verification requires CUDA")
        q, k, v, alpha, b, w, _ = _make_tensors(L=L, dtype=torch.float16)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(
            q.float(), k.float(), v.float(), alpha.float(), b.float(), w.float()
        )

        diff_out = (out.float() - ref_out).abs().max().item()
        diff_state = (state.float() - ref_state).abs().max().item()

        assert diff_out < 5e-3, f"FP16 Output diff {diff_out} exceeded 5e-3 at L={L}"
        assert diff_state < 5e-3, f"FP16 State diff {diff_state} exceeded 5e-3 at L={L}"

    @pytest.mark.parametrize("L", [16, 64])
    def test_prefill_bf16_multi_lengths(self, L: int):
        if not CUDA_AVAILABLE:
            pytest.skip("BF16 hardware verification requires CUDA")
        q, k, v, alpha, b, w, _ = _make_tensors(L=L, dtype=torch.bfloat16)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(
            q.float(), k.float(), v.float(), alpha.float(), b.float(), w.float()
        )

        diff_out = (out.float() - ref_out).abs().max().item()
        diff_state = (state.float() - ref_state).abs().max().item()

        assert diff_out < 2e-2, f"BF16 Output diff {diff_out} exceeded 2e-2 at L={L}"
        assert diff_state < 2e-2, f"BF16 State diff {diff_state} exceeded 2e-2 at L={L}"

    @pytest.mark.parametrize("B,H", [(1, 1), (2, 4), (4, 8)])
    def test_prefill_various_batch_head_configs(self, B: int, H: int):
        q, k, v, alpha, b, w, _ = _make_tensors(B=B, H=H, L=32)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w)

        assert (out - ref_out).abs().max().item() < 1e-4
        assert (state - ref_state).abs().max().item() < 1e-4



class TestTritonDGDARemainderHandling:

    @pytest.mark.parametrize("L", [1, 7, 15, 17, 31, 33, 47, 49, 63, 65, 127])
    def test_prefill_misaligned_lengths(self, L: int):
        q, k, v, alpha, b, w, _ = _make_tensors(L=L)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w)

        diff_out = (out - ref_out).abs().max().item()
        diff_state = (state - ref_state).abs().max().item()

        assert diff_out < 1e-4, f"Misaligned L={L} out diff {diff_out} >= 1e-4"
        assert diff_state < 1e-4, f"Misaligned L={L} state diff {diff_state} >= 1e-4"

    def test_prefill_single_token(self):
        q, k, v, alpha, b, w, _ = _make_tensors(L=1)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w)
        assert (out - ref_out).abs().max().item() < 1e-5
        assert (state - ref_state).abs().max().item() < 1e-5

    def test_empty_sequence_L0(self):
        q, k, v, alpha, b, w, s0 = _make_tensors(L=0, s0=True)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        assert out.shape == (2, 4, 0, 64)
        assert (state - s0).abs().max().item() == 0.0



class TestTritonDGDAInitialState:

    def test_prefill_with_nonzero_initial_state(self):
        q, k, v, alpha, b, w, s0 = _make_tensors(L=32, s0=True)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert (out - ref_out).abs().max().item() < 1e-4
        assert (state - ref_state).abs().max().item() < 1e-4

    def test_prefill_chunk_chaining_equivalence(self):
        q1, k1, v1, a1, b1, w1, s0 = _make_tensors(L=32, s0=True)
        q2, k2, v2, a2, b2, w2, _ = _make_tensors(L=32)

        q_cat = torch.cat([q1, q2], dim=2)
        k_cat = torch.cat([k1, k2], dim=2)
        v_cat = torch.cat([v1, v2], dim=2)
        a_cat = torch.cat([a1, a2], dim=2)
        b_cat = torch.cat([b1, b2], dim=2)
        w_cat = torch.cat([w1, w2], dim=2)
        out_full, state_full = triton_dgda_prefill(q_cat, k_cat, v_cat, a_cat, b_cat, w_cat, initial_state=s0)

        out1, state1 = triton_dgda_prefill(q1, k1, v1, a1, b1, w1, initial_state=s0)
        out2, state2 = triton_dgda_prefill(q2, k2, v2, a2, b2, w2, initial_state=state1)

        out_chained = torch.cat([out1, out2], dim=2)
        assert (out_full - out_chained).abs().max().item() < 1e-4
        assert (state_full - state2).abs().max().item() < 1e-4



class TestTritonDGDAStepDecode:

    def test_step_decode_exact_match_fp32(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=1, s0=True)
        o_step, s_step = triton_dgda_step(q, k, v, a, b, w, state=s0)
        ref_o, ref_s = ref_dgda_step(q, k, v, a, b, w, state=s0)

        assert (o_step - ref_o).abs().max().item() < 1e-5
        assert (s_step - ref_s).abs().max().item() < 1e-5

    def test_step_decode_exact_match_fp16(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=1, dtype=torch.float16, s0=True)
        o_step, s_step = triton_dgda_step(q, k, v, a, b, w, state=s0)
        ref_o, ref_s = ref_dgda_step(
            q.float(), k.float(), v.float(), a.float(), b.float(), w.float(), state=s0.float()
        )

        assert (o_step.float() - ref_o).abs().max().item() < 1e-2
        assert (s_step.float() - ref_s).abs().max().item() < 5e-3

    def test_step_unroll_vs_chunkwise_prefill(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=32, s0=True)
        curr_state = s0.clone()
        step_outs = []
        for t in range(32):
            ot, curr_state = triton_dgda_step(
                q[:, :, t], k[:, :, t], v[:, :, t],
                a[:, :, t], b[:, :, t], w[:, :, t],
                state=curr_state
            )
            step_outs.append(ot)

        out_unrolled = torch.stack(step_outs, dim=2)
        out_fwd, state_fwd = triton_dgda_prefill(q, k, v, a, b, w, initial_state=s0)

        assert (out_unrolled - out_fwd).abs().max().item() < 1e-4
        assert (curr_state - state_fwd).abs().max().item() < 1e-4

    def test_step_o1_memory_invariance(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=1, s0=True)
        state = s0.clone()
        for _ in range(5):
            _, state = triton_dgda_step(q, k, v, a, b, w, state=state)
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
        else:
            mem_before = 0
        for _ in range(50):
            _, state = triton_dgda_step(q, k, v, a, b, w, state=state)
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
        else:
            mem_after = 0
        assert mem_after <= mem_before



class TestTritonDGDAAutogradGradients:

    def test_backward_gradients_finite_and_nonzero(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=32, requires_grad=True, s0=True)
        out, state = triton_dgda_prefill(q, k, v, a, b, w, initial_state=s0)
        loss = (out ** 2).sum() + (state ** 2).sum()
        loss.backward()

        for name, tensor in [("q", q), ("k", k), ("v", v), ("alpha", a), ("b", b), ("w", w), ("s0", s0)]:
            assert tensor.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(tensor.grad).any(), f"Gradient for {name} contains NaN"
            assert not torch.isinf(tensor.grad).any(), f"Gradient for {name} contains Inf"
            assert (tensor.grad.abs() > 0).any(), f"Gradient for {name} is all zeros"

    def test_backward_gradient_relative_error(self):
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = _make_tensors(L=24, requires_grad=True, s0=True)
        q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, s0_tr = (
            q_ref.detach().clone().requires_grad_(),
            k_ref.detach().clone().requires_grad_(),
            v_ref.detach().clone().requires_grad_(),
            a_ref.detach().clone().requires_grad_(),
            b_ref.detach().clone().requires_grad_(),
            w_ref.detach().clone().requires_grad_(),
            s0_ref.detach().clone().requires_grad_(),
        )

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=s0_ref)
        loss_ref = out_ref.sum() + s_ref.sum()
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        loss_tr = out_tr.sum() + s_tr.sum()
        loss_tr.backward()

        tensors = [
            ("q", q_tr.grad, q_ref.grad),
            ("k", k_tr.grad, k_ref.grad),
            ("v", v_tr.grad, v_ref.grad),
            ("alpha", a_tr.grad, a_ref.grad),
            ("b", b_tr.grad, b_ref.grad),
            ("w", w_tr.grad, w_ref.grad),
            ("s0", s0_tr.grad, s0_ref.grad),
        ]

        for name, g_tr, g_ref in tensors:
            rel_err = (torch.linalg.norm(g_tr - g_ref) / (torch.linalg.norm(g_ref) + 1e-6)).item()
            assert rel_err < 1e-2, f"Relative gradient error for {name} was {rel_err:.2e} (>= 1e-2)"

    def test_causal_gradient_isolation(self):
        q, k, v, a, b, w, _ = _make_tensors(L=32, requires_grad=True)
        out, _ = triton_dgda_prefill(q, k, v, a, b, w)
        loss = out[:, :, 12].sum()
        loss.backward()

        for name, tensor in [("q", q), ("k", k), ("v", v), ("alpha", a), ("b", b), ("w", w)]:
            future_grads = tensor.grad[:, :, 13:]
            assert (future_grads == 0.0).all(), f"Causal gradient leakage in {name} for t > 12"



class TestTritonDGDABoundaryRegimes:

    def test_extreme_decay_instant_forgetting(self):
        q, k, v, _, b, w, s0 = _make_tensors(L=32, s0=True)
        alpha = torch.full_like(q, 1e-20)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_o, ref_s = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert not torch.isnan(out).any()
        assert not torch.isnan(state).any()
        assert (out - ref_o).abs().max().item() < 1e-4
        assert (state - ref_s).abs().max().item() < 1e-4

    def test_extreme_decay_no_decay(self):
        q, k, v, _, b, w, s0 = _make_tensors(L=32, s0=True)
        alpha = torch.ones_like(q)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_o, ref_s = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert not torch.isnan(out).any()
        assert not torch.isnan(state).any()
        assert (out - ref_o).abs().max().item() < 5e-3
        assert (state - ref_s).abs().max().item() < 1e-3

    def test_extreme_gates_full_erase(self):
        q, k, v, alpha, _, _, s0 = _make_tensors(L=32, s0=True)
        b = torch.ones_like(q)
        w = torch.zeros_like(v)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_o, ref_s = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert not torch.isnan(out).any()
        assert (out - ref_o).abs().max().item() < 1e-4
        assert (state - ref_s).abs().max().item() < 1e-4

    def test_extreme_gates_full_write(self):
        q, k, v, alpha, _, _, s0 = _make_tensors(L=32, s0=True)
        b = torch.zeros_like(q)
        w = torch.ones_like(v)
        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_o, ref_s = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        assert not torch.isnan(out).any()
        assert (out - ref_o).abs().max().item() < 1e-4
        assert (state - ref_s).abs().max().item() < 1e-4



class TestTritonDispatcherIntegration:

    def test_dispatcher_triton_registration(self):
        from maba_sparse.kernels.dispatcher import get_kernel
        fn_prefill = get_kernel("triton", "dgda_prefill")
        fn_step = get_kernel("triton", "dgda_step")
        assert fn_prefill is not None
        assert fn_step is not None

    def test_dispatcher_routes_cuda_tensor_to_triton(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")
        q, k, v, a, b, w, _ = _make_tensors(L=32)
        out, state = dispatch_dgda_prefill(q, k, v, a, b, w)
        ref_o, ref_s = ref_dgda_prefill(q, k, v, a, b, w)
        assert (out - ref_o).abs().max().item() < 1e-4
        assert (state - ref_s).abs().max().item() < 1e-4

    def test_dispatcher_fallback_on_simulated_exception(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "triton")
        clear_fallback_warnings()

        def failing_kernel(*args, **kwargs):
            raise RuntimeError("Simulated Triton kernel fault")

        register_kernel("triton", "dgda_prefill")(failing_kernel)
        q, k, v, a, b, w, _ = _make_tensors(L=16)
        with pytest.warns(RuntimeWarning, match="Simulated Triton kernel fault"):
            out, state = dispatch_dgda_prefill(q, k, v, a, b, w)

        ref_o, ref_s = ref_dgda_prefill(q, k, v, a, b, w)
        assert (out - ref_o).abs().max().item() < 1e-4

        register_kernel("triton", "dgda_prefill")(triton_dgda_prefill)
