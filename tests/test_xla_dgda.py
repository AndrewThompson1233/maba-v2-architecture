
import pytest
import torch

from maba_sparse.kernels.common import (
    ref_dgda_prefill,
    ref_dgda_step,
)
from maba_sparse.kernels.dispatcher import (
    dispatch_dgda_prefill,
)
from maba_sparse.kernels.xla_dgda import (
    get_static_bucket_length,
    xla_dgda_prefill,
    xla_dgda_step,
)


def _make_tensors(
    B: int = 2,
    H: int = 4,
    L: int = 32,
    dk: int = 64,
    dv: int = 64,
    device: str = "cpu",
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



class TestXLAStaticShapeConformance:

    def test_bucket_length_resolution(self):
        assert get_static_bucket_length(1) == 128
        assert get_static_bucket_length(100) == 128
        assert get_static_bucket_length(128) == 128
        assert get_static_bucket_length(129) == 256
        assert get_static_bucket_length(500) == 512
        assert get_static_bucket_length(512) == 512
        assert get_static_bucket_length(1025) == 2048
        assert get_static_bucket_length(5000) == ((5000 + 15) // 16) * 16

    def test_explicit_static_seq_len_padding(self):
        q, k, v, alpha, b, w, _ = _make_tensors(L=35)
        out, state = xla_dgda_prefill(q, k, v, alpha, b, w, static_seq_len=64)
        assert out.shape == (2, 4, 35, 64)
        assert state.shape == (2, 4, 64, 64)

    def test_static_seq_len_smaller_than_L_raises(self):
        q, k, v, alpha, b, w, _ = _make_tensors(L=50)
        with pytest.raises(ValueError, match="must be >= sequence length L"):
            xla_dgda_prefill(q, k, v, alpha, b, w, static_seq_len=32)



class TestXLADGDAPrefillNumericalParity:

    @pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
    def test_prefill_fp32_multi_lengths(self, L: int):
        q, k, v, alpha, b, w, s0 = _make_tensors(L=L, s0=(L in (32, 128)))
        out, state = xla_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)

        diff_out = (out - ref_out).abs().max().item()
        diff_state = (state - ref_state).abs().max().item()

        assert diff_out < 1e-4, f"Output diff {diff_out} exceeded 1e-4 at L={L}"
        assert diff_state < 1e-4, f"State diff {diff_state} exceeded 1e-4 at L={L}"

    def test_prefill_float16_stability(self):
        q, k, v, alpha, b, w, _ = _make_tensors(L=32, dtype=torch.float16)
        out, state = xla_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w)

        diff_out = (out.float() - ref_out.float()).abs().max().item()
        diff_state = (state.float() - ref_state.float()).abs().max().item()

        assert diff_out < 1e-2, f"FP16 Output diff {diff_out} exceeded 1e-2"
        assert diff_state < 1e-2, f"FP16 State diff {diff_state} exceeded 1e-2"

    def test_prefill_bfloat16_stability(self):
        q, k, v, alpha, b, w, _ = _make_tensors(L=32, dtype=torch.bfloat16)
        out, state = xla_dgda_prefill(q, k, v, alpha, b, w)
        ref_out, ref_state = ref_dgda_prefill(
            q.float(), k.float(), v.float(), alpha.float(), b.float(), w.float()
        )

        diff_out = (out.float() - ref_out).abs().max().item()
        diff_state = (state.float() - ref_state).abs().max().item()

        assert diff_out < 0.02, f"BF16 Output diff {diff_out} exceeded 0.02"
        assert diff_state < 0.02, f"BF16 State diff {diff_state} exceeded 0.02"

    def test_prefill_neumann_order_3_vs_4(self):
        q, k, v, alpha, b, w, _ = _make_tensors(L=48)
        out3, state3 = xla_dgda_prefill(q, k, v, alpha, b, w, neumann_order=3)
        out4, state4 = xla_dgda_prefill(q, k, v, alpha, b, w, neumann_order=4)
        ref_out, ref_state = ref_dgda_prefill(q, k, v, alpha, b, w)

        assert (out3 - ref_out).abs().max().item() < 1.5e-4
        assert (out4 - ref_out).abs().max().item() < 1e-4
        assert (state3 - ref_state).abs().max().item() < 1.5e-4
        assert (state4 - ref_state).abs().max().item() < 1e-4



class TestXLABucketingAndMasking:

    @pytest.mark.parametrize("L", [1, 5, 7, 15, 17, 31, 33, 47, 63, 100])
    def test_identity_padding_bitwise_parity(self, L: int):
        q, k, v, alpha, b, w, _ = _make_tensors(L=L)
        out_pad, state_pad = xla_dgda_prefill(q, k, v, alpha, b, w, use_bucketing=True)
        out_ref, state_ref = ref_dgda_prefill(q, k, v, alpha, b, w)

        assert (out_pad - out_ref).abs().max().item() < 1e-4
        assert (state_pad - state_ref).abs().max().item() < 1e-4

    def test_empty_sequence_L0(self):
        q, k, v, alpha, b, w, s0 = _make_tensors(L=0, s0=True)
        out, state = xla_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        assert out.shape == (2, 4, 0, 64)
        assert (state - s0).abs().max().item() == 0.0



class TestXLAStateProgression:

    def test_chunk_chaining_equivalence(self):
        q1, k1, v1, a1, b1, w1, s0 = _make_tensors(L=32, s0=True)
        q2, k2, v2, a2, b2, w2, _ = _make_tensors(L=32)

        q_cat = torch.cat([q1, q2], dim=2)
        k_cat = torch.cat([k1, k2], dim=2)
        v_cat = torch.cat([v1, v2], dim=2)
        a_cat = torch.cat([a1, a2], dim=2)
        b_cat = torch.cat([b1, b2], dim=2)
        w_cat = torch.cat([w1, w2], dim=2)
        out_full, state_full = xla_dgda_prefill(q_cat, k_cat, v_cat, a_cat, b_cat, w_cat, initial_state=s0)

        out1, state1 = xla_dgda_prefill(q1, k1, v1, a1, b1, w1, initial_state=s0)
        out2, state2 = xla_dgda_prefill(q2, k2, v2, a2, b2, w2, initial_state=state1)

        out_chained = torch.cat([out1, out2], dim=2)
        assert (out_full - out_chained).abs().max().item() < 1e-4
        assert (state_full - state2).abs().max().item() < 1e-4

    def test_step_decode_vs_reference(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=1, s0=True)
        o_step, s_step = xla_dgda_step(q, k, v, a, b, w, state=s0)
        ref_o, ref_s = ref_dgda_step(q, k, v, a, b, w, state=s0)

        assert (o_step - ref_o).abs().max().item() < 1e-5
        assert (s_step - ref_s).abs().max().item() < 1e-5

    def test_step_decode_unroll_vs_prefill(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=32, s0=True)
        curr_state = s0.clone()
        step_outs = []
        for t in range(32):
            ot, curr_state = xla_dgda_step(
                q[:, :, t], k[:, :, t], v[:, :, t],
                a[:, :, t], b[:, :, t], w[:, :, t],
                state=curr_state
            )
            step_outs.append(ot)

        out_unrolled = torch.stack(step_outs, dim=2)
        out_fwd, state_fwd = xla_dgda_prefill(q, k, v, a, b, w, initial_state=s0)

        assert (out_unrolled - out_fwd).abs().max().item() < 1e-4
        assert (curr_state - state_fwd).abs().max().item() < 1e-4



class TestXLAAutogradStaticGraph:

    def test_backward_gradients_finite_and_nonzero(self):
        q, k, v, a, b, w, s0 = _make_tensors(L=32, requires_grad=True, s0=True)
        out, state = xla_dgda_prefill(q, k, v, a, b, w, initial_state=s0)
        loss = (out ** 2).sum() + (state ** 2).sum()
        loss.backward()

        for name, tensor in [("q", q), ("k", k), ("v", v), ("alpha", a), ("b", b), ("w", w), ("s0", s0)]:
            assert tensor.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(tensor.grad).any(), f"Gradient for {name} contains NaN"
            assert not torch.isinf(tensor.grad).any(), f"Gradient for {name} contains Inf"
            assert (tensor.grad.abs() > 0).any(), f"Gradient for {name} is all zeros"

    def test_backward_gradient_relative_error_vs_reference(self):
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = _make_tensors(L=24, requires_grad=True, s0=True)
        q_xla, k_xla, v_xla, a_xla, b_xla, w_xla, s0_xla = (
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

        out_xla, s_xla = xla_dgda_prefill(q_xla, k_xla, v_xla, a_xla, b_xla, w_xla, initial_state=s0_xla)
        loss_xla = out_xla.sum() + s_xla.sum()
        loss_xla.backward()

        tensors = [
            ("q", q_xla.grad, q_ref.grad),
            ("k", k_xla.grad, k_ref.grad),
            ("v", v_xla.grad, v_ref.grad),
            ("alpha", a_xla.grad, a_ref.grad),
            ("b", b_xla.grad, b_ref.grad),
            ("w", w_xla.grad, w_ref.grad),
            ("s0", s0_xla.grad, s0_ref.grad),
        ]

        for name, g_xla, g_ref in tensors:
            rel_err = (torch.linalg.norm(g_xla - g_ref) / (torch.linalg.norm(g_ref) + 1e-6)).item()
            assert rel_err < 1e-2, f"Relative gradient error for {name} was {rel_err:.2e} (>= 1e-2)"

    def test_causal_gradient_isolation(self):
        q, k, v, a, b, w, _ = _make_tensors(L=32, requires_grad=True)
        out, _ = xla_dgda_prefill(q, k, v, a, b, w)
        loss = out[:, :, 10].sum()
        loss.backward()

        for name, tensor in [("q", q), ("k", k), ("v", v), ("alpha", a), ("b", b), ("w", w)]:
            future_grads = tensor.grad[:, :, 11:]
            assert (future_grads == 0.0).all(), f"Future gradient leakage in {name} for t > 10"



class TestXLADispatcherFallback:

    def test_dispatcher_xla_registration(self):
        from maba_sparse.kernels.dispatcher import get_kernel
        fn_prefill = get_kernel("xla", "dgda_prefill")
        fn_step = get_kernel("xla", "dgda_step")
        assert fn_prefill is not None
        assert fn_step is not None

    def test_mock_xla_execution_via_dispatcher(self, monkeypatch):
        monkeypatch.setenv("MABA_BACKEND", "xla")
        q, k, v, a, b, w, _ = _make_tensors(L=32)
        out, state = dispatch_dgda_prefill(q, k, v, a, b, w)
        ref_o, ref_s = ref_dgda_prefill(q, k, v, a, b, w)
        assert (out - ref_o).abs().max().item() < 1e-4
        assert (state - ref_s).abs().max().item() < 1e-4
