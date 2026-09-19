
import time
import pytest
import torch

from maba_sparse.kernels.common import (
    ref_dgda_prefill,
    ref_dgda_step,
    normalize_keys,
)
from maba_sparse.kernels.dispatcher import (
    dispatch_dgda_prefill,
    dispatch_dgda_step,
    is_cuda_sm75_available,
    is_triton_available,
)
from maba_sparse.kernels.triton_dgda import (
    triton_dgda_prefill,
    triton_dgda_step,
)

CUDA_ONLINE = torch.cuda.is_available() and is_cuda_sm75_available() and is_triton_available()
DEVICE = "cuda" if CUDA_ONLINE else "cpu"


def make_dgda_inputs(
    B: int = 2,
    H: int = 4,
    L: int = 32,
    dk: int = 64,
    dv: int = 64,
    device: str = DEVICE,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
    has_s0: bool = True,
    seed: int = 42,
    alpha_range: tuple = (0.1, 0.95),
):
    torch.manual_seed(seed)
    q = torch.randn(B, H, L, dk, device=device, dtype=dtype, requires_grad=requires_grad)
    k_raw = torch.randn(B, H, L, dk, device=device, dtype=dtype)
    k = normalize_keys(k_raw).detach().requires_grad_(requires_grad)
    v = torch.randn(B, H, L, dv, device=device, dtype=dtype, requires_grad=requires_grad)

    a_min, a_max = alpha_range
    alpha = (torch.rand(B, H, L, dk, device=device, dtype=dtype) * (a_max - a_min) + a_min).requires_grad_(requires_grad)
    b = torch.rand(B, H, L, dk, device=device, dtype=dtype, requires_grad=requires_grad)
    w = torch.rand(B, H, L, dv, device=device, dtype=dtype, requires_grad=requires_grad)

    s0 = None
    if has_s0:
        s0 = torch.randn(B, H, dk, dv, device=device, dtype=dtype, requires_grad=requires_grad)

    return q, k, v, alpha, b, w, s0


def compute_rel_err(grad_test: torch.Tensor, grad_ref: torch.Tensor, eps: float = 1e-6) -> float:
    norm_diff = torch.linalg.norm(grad_test.float() - grad_ref.float()).item()
    norm_ref = torch.linalg.norm(grad_ref.float()).item()
    return norm_diff / (norm_ref + eps)


class TestAutogradReverseModeStress:

    @pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
    def test_autograd_all_7_inputs_multi_chunk(self, L: int):
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = make_dgda_inputs(L=L, has_s0=True, seed=100 + L)

        q_tr = q_ref.detach().clone().requires_grad_()
        k_tr = k_ref.detach().clone().requires_grad_()
        v_tr = v_ref.detach().clone().requires_grad_()
        a_tr = a_ref.detach().clone().requires_grad_()
        b_tr = b_ref.detach().clone().requires_grad_()
        w_tr = w_ref.detach().clone().requires_grad_()
        s0_tr = s0_ref.detach().clone().requires_grad_()

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=s0_ref)
        loss_ref = out_ref.sum() + s_ref.sum()
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        loss_tr = out_tr.sum() + s_tr.sum()
        loss_tr.backward()

        inputs = [
            ("q", q_tr, q_ref),
            ("k", k_tr, k_ref),
            ("v", v_tr, v_ref),
            ("alpha", a_tr, a_ref),
            ("b", b_tr, b_ref),
            ("w", w_tr, w_ref),
            ("s0", s0_tr, s0_ref),
        ]

        for name, t_tr, t_ref in inputs:
            assert t_tr.grad is not None, f"Gradient for {name} is None at L={L}"
            assert not torch.isnan(t_tr.grad).any(), f"Gradient for {name} contains NaN at L={L}"
            assert not torch.isinf(t_tr.grad).any(), f"Gradient for {name} contains Inf at L={L}"
            assert (t_tr.grad.abs() > 0).any(), f"Gradient for {name} is all zeros at L={L}"

            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, (
                f"Gradient relative error for {name} was {rel_err:.4e} (> 1e-2) at L={L}"
            )

    @pytest.mark.parametrize("loss_scale", [1e-4, 1e-2, 10.0, 1e3, 1e5])
    def test_autograd_high_loss_scaling(self, loss_scale: float):
        L = 32
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = make_dgda_inputs(L=L, has_s0=True, seed=int(loss_scale * 10) % 9999 + 1)

        q_tr = q_ref.detach().clone().requires_grad_()
        k_tr = k_ref.detach().clone().requires_grad_()
        v_tr = v_ref.detach().clone().requires_grad_()
        a_tr = a_ref.detach().clone().requires_grad_()
        b_tr = b_ref.detach().clone().requires_grad_()
        w_tr = w_ref.detach().clone().requires_grad_()
        s0_tr = s0_ref.detach().clone().requires_grad_()

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=s0_ref)
        loss_ref = (out_ref.sum() + s_ref.sum()) * loss_scale
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        loss_tr = (out_tr.sum() + s_tr.sum()) * loss_scale
        loss_tr.backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q_ref), ("k", k_tr, k_ref), ("v", v_tr, v_ref),
            ("alpha", a_tr, a_ref), ("b", b_tr, b_ref), ("w", w_tr, w_ref), ("s0", s0_tr, s0_ref)
        ]:
            assert not torch.isnan(t_tr.grad).any(), f"NaN in {name} with loss_scale={loss_scale}"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in {name} with loss_scale={loss_scale}"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, (
                f"Gradient relative error for {name} with loss_scale={loss_scale}: {rel_err:.4e}"
            )

    def test_autograd_random_projection_loss(self):
        torch.manual_seed(2026)
        B, H, L, dk, dv = 2, 4, 48, 64, 64
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = make_dgda_inputs(
            B=B, H=H, L=L, dk=dk, dv=dv, has_s0=True, seed=2026
        )

        q_tr = q_ref.detach().clone().requires_grad_()
        k_tr = k_ref.detach().clone().requires_grad_()
        v_tr = v_ref.detach().clone().requires_grad_()
        a_tr = a_ref.detach().clone().requires_grad_()
        b_tr = b_ref.detach().clone().requires_grad_()
        w_tr = w_ref.detach().clone().requires_grad_()
        s0_tr = s0_ref.detach().clone().requires_grad_()

        p_out = torch.randn(B, H, L, dv, device=DEVICE, dtype=torch.float32)
        p_state = torch.randn(B, H, dk, dv, device=DEVICE, dtype=torch.float32)

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=s0_ref)
        loss_ref = (out_ref * p_out).sum() + (s_ref * p_state).sum()
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        loss_tr = (out_tr * p_out).sum() + (s_tr * p_state).sum()
        loss_tr.backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q_ref), ("k", k_tr, k_ref), ("v", v_tr, v_ref),
            ("alpha", a_tr, a_ref), ("b", b_tr, b_ref), ("w", w_tr, w_ref), ("s0", s0_tr, s0_ref)
        ]:
            assert not torch.isnan(t_tr.grad).any(), f"NaN in {name} with projection loss"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in {name} with projection loss"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, (
                f"Gradient relative error for {name} with random projection loss: {rel_err:.4e}"
            )

    @pytest.mark.parametrize("L", [1, 7, 15, 17, 31, 33, 47, 65, 127])
    def test_autograd_misaligned_lengths(self, L: int):
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = make_dgda_inputs(L=L, has_s0=True, seed=3000 + L)

        q_tr = q_ref.detach().clone().requires_grad_()
        k_tr = k_ref.detach().clone().requires_grad_()
        v_tr = v_ref.detach().clone().requires_grad_()
        a_tr = a_ref.detach().clone().requires_grad_()
        b_tr = b_ref.detach().clone().requires_grad_()
        w_tr = w_ref.detach().clone().requires_grad_()
        s0_tr = s0_ref.detach().clone().requires_grad_()

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=s0_ref)
        loss_ref = out_ref.sum() + s_ref.sum()
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        loss_tr = out_tr.sum() + s_tr.sum()
        loss_tr.backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q_ref), ("k", k_tr, k_ref), ("v", v_tr, v_ref),
            ("alpha", a_tr, a_ref), ("b", b_tr, b_ref), ("w", w_tr, w_ref), ("s0", s0_tr, s0_ref)
        ]:
            assert not torch.isnan(t_tr.grad).any(), f"NaN in {name} at misaligned L={L}"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in {name} at misaligned L={L}"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, (
                f"Gradient relative error for {name} at misaligned L={L}: {rel_err:.4e}"
            )

    def test_autograd_without_initial_state(self):
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, _ = make_dgda_inputs(L=32, has_s0=False, seed=4001)

        q_tr = q_ref.detach().clone().requires_grad_()
        k_tr = k_ref.detach().clone().requires_grad_()
        v_tr = v_ref.detach().clone().requires_grad_()
        a_tr = a_ref.detach().clone().requires_grad_()
        b_tr = b_ref.detach().clone().requires_grad_()
        w_tr = w_ref.detach().clone().requires_grad_()

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=None)
        loss_ref = out_ref.sum() + s_ref.sum()
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=None)
        loss_tr = out_tr.sum() + s_tr.sum()
        loss_tr.backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q_ref), ("k", k_tr, k_ref), ("v", v_tr, v_ref),
            ("alpha", a_tr, a_ref), ("b", b_tr, b_ref), ("w", w_tr, w_ref)
        ]:
            assert t_tr.grad is not None, f"Gradient for {name} is None when s0 is None"
            assert not torch.isnan(t_tr.grad).any(), f"NaN in {name} when s0 is None"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in {name} when s0 is None"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, f"Gradient error for {name} when s0 is None: {rel_err:.4e}"

    def test_autograd_single_step_decode_all_7_inputs(self):
        B, H, dk, dv = 2, 4, 64, 64
        q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, s0_ref = make_dgda_inputs(
            B=B, H=H, L=1, dk=dk, dv=dv, has_s0=True, seed=5001
        )

        q_tr = q_ref.squeeze(2).detach().clone().requires_grad_()
        k_tr = k_ref.squeeze(2).detach().clone().requires_grad_()
        v_tr = v_ref.squeeze(2).detach().clone().requires_grad_()
        a_tr = a_ref.squeeze(2).detach().clone().requires_grad_()
        b_tr = b_ref.squeeze(2).detach().clone().requires_grad_()
        w_tr = w_ref.squeeze(2).detach().clone().requires_grad_()
        s0_tr = s0_ref.detach().clone().requires_grad_()

        q_r = q_ref.squeeze(2).detach().clone().requires_grad_()
        k_r = k_ref.squeeze(2).detach().clone().requires_grad_()
        v_r = v_ref.squeeze(2).detach().clone().requires_grad_()
        a_r = a_ref.squeeze(2).detach().clone().requires_grad_()
        b_r = b_ref.squeeze(2).detach().clone().requires_grad_()
        w_r = w_ref.squeeze(2).detach().clone().requires_grad_()
        s0_r = s0_ref.detach().clone().requires_grad_()

        o_ref, s_new_ref = ref_dgda_step(q_r, k_r, v_r, a_r, b_r, w_r, state=s0_r)
        loss_ref = o_ref.sum() + s_new_ref.sum()
        loss_ref.backward()

        o_tr, s_new_tr = triton_dgda_step(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, state=s0_tr)
        loss_tr = o_tr.sum() + s_new_tr.sum()
        loss_tr.backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q_r), ("k", k_tr, k_r), ("v", v_tr, v_r),
            ("alpha", a_tr, a_r), ("b", b_tr, b_r), ("w", w_tr, w_r), ("s0", s0_tr, s0_r)
        ]:
            assert t_tr.grad is not None, f"Decode grad for {name} is None"
            assert not torch.isnan(t_tr.grad).any(), f"NaN in decode grad for {name}"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in decode grad for {name}"
            assert (t_tr.grad.abs() > 0).any(), f"Zero decode grad for {name}"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, f"Decode grad error for {name}: {rel_err:.4e}"


class TestDecodeO1MemoryAndComplexity:

    def test_strict_o1_memory_invariance_100_steps(self):
        B, H, dk, dv = 2, 4, 64, 64
        q, k, v, a, b, w, s0 = make_dgda_inputs(
            B=B, H=H, L=1, dk=dk, dv=dv, has_s0=True, requires_grad=False, seed=6001
        )
        q_step = q.squeeze(2)
        k_step = k.squeeze(2)
        v_step = v.squeeze(2)
        a_step = a.squeeze(2)
        b_step = b.squeeze(2)
        w_step = w.squeeze(2)

        state = s0.clone()

        for _ in range(10):
            _, state = dispatch_dgda_step(q_step, k_step, v_step, a_step, b_step, w_step, state)

        if CUDA_ONLINE:
            torch.cuda.synchronize()
            mem_start = torch.cuda.memory_allocated()
        else:
            mem_start = 0

        memory_snapshots = []
        latencies = []

        for step_idx in range(100):
            t0 = time.perf_counter()
            _, state = dispatch_dgda_step(q_step, k_step, v_step, a_step, b_step, w_step, state)
            if CUDA_ONLINE:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            if CUDA_ONLINE and (step_idx + 1) % 10 == 0:
                current_mem = torch.cuda.memory_allocated()
                memory_snapshots.append((step_idx + 1, current_mem))

        if CUDA_ONLINE:
            torch.cuda.synchronize()
            mem_end = torch.cuda.memory_allocated()
            mem_growth = mem_end - mem_start

            print("\n--- 100-Step Memory Profile (Tesla T4) ---")
            for step_num, mem_val in memory_snapshots:
                print(f"Step {step_num:3d}: {mem_val:,} bytes (delta: {mem_val - mem_start:+d} bytes)")

            assert mem_growth == 0, (
                f"Memory leak detected! GPU memory grew by {mem_growth} bytes across 100 steps."
            )

        early_latency = sum(latencies[10:30]) / 20.0
        late_latency = sum(latencies[80:100]) / 20.0
        ratio = late_latency / (early_latency + 1e-6)

        print(f"\nLatency: early={early_latency:.4f} ms/step, late={late_latency:.4f} ms/step, ratio={ratio:.2f}")
        assert ratio < 2.0, f"Decode step latency degraded significantly over sequence: ratio={ratio:.2f}"


class TestLongSequenceContinuity:

    @pytest.mark.parametrize("L_prefill", [512, 1024])
    def test_prefill_plus_decode_continuity(self, L_prefill: int):
        K_decode = 32
        L_total = L_prefill + K_decode

        q, k, v, a, b, w, s0 = make_dgda_inputs(
            L=L_total, has_s0=True, requires_grad=False, seed=7000 + L_prefill
        )

        out_mono, state_mono = dispatch_dgda_prefill(q, k, v, a, b, w, initial_state=s0)

        q_pref = q[:, :, :L_prefill]
        k_pref = k[:, :, :L_prefill]
        v_pref = v[:, :, :L_prefill]
        a_pref = a[:, :, :L_prefill]
        b_pref = b[:, :, :L_prefill]
        w_pref = w[:, :, :L_prefill]

        out_pref, state_curr = dispatch_dgda_prefill(q_pref, k_pref, v_pref, a_pref, b_pref, w_pref, initial_state=s0)

        pref_diff = (out_pref - out_mono[:, :, :L_prefill]).abs().max().item()
        assert pref_diff < 1.0e-4, f"Prefill prefix diff {pref_diff:.2e} exceeded 1e-4 at L={L_prefill}"

        decode_outs = []
        for step in range(K_decode):
            idx = L_prefill + step
            q_s = q[:, :, idx]
            k_s = k[:, :, idx]
            v_s = v[:, :, idx]
            a_s = a[:, :, idx]
            b_s = b[:, :, idx]
            w_s = w[:, :, idx]

            ot, state_curr = dispatch_dgda_step(q_s, k_s, v_s, a_s, b_s, w_s, state_curr)
            decode_outs.append(ot)

        out_decoded = torch.stack(decode_outs, dim=2)
        out_mono_suffix = out_mono[:, :, L_prefill:]

        diff_decode_out = (out_decoded - out_mono_suffix).abs().max().item()
        diff_final_state = (state_curr - state_mono).abs().max().item()

        print(f"\nContinuity L={L_prefill} + K={K_decode}: max_out_diff={diff_decode_out:.2e}, max_state_diff={diff_final_state:.2e}")
        assert diff_decode_out < 1.0e-4, (
            f"Decoded outputs diverged from prefill: max diff {diff_decode_out:.2e} (>= 1e-4) at L={L_prefill}"
        )
        assert diff_final_state < 1.0e-4, (
            f"Final decoded state diverged from prefill: max diff {diff_final_state:.2e} (>= 1e-4) at L={L_prefill}"
        )

    @pytest.mark.parametrize("L", [512, 1024])
    def test_chunk_chaining_continuity_long_context(self, L: int):
        q, k, v, a, b, w, s0 = make_dgda_inputs(L=L, has_s0=True, requires_grad=False, seed=8000 + L)

        mid = L // 2
        out_mono, state_mono = dispatch_dgda_prefill(q, k, v, a, b, w, initial_state=s0)

        out1, state1 = dispatch_dgda_prefill(
            q[:, :, :mid], k[:, :, :mid], v[:, :, :mid],
            a[:, :, :mid], b[:, :, :mid], w[:, :, :mid],
            initial_state=s0
        )
        out2, state2 = dispatch_dgda_prefill(
            q[:, :, mid:], k[:, :, mid:], v[:, :, mid:],
            a[:, :, mid:], b[:, :, mid:], w[:, :, mid:],
            initial_state=state1
        )

        out_chained = torch.cat([out1, out2], dim=2)
        diff_out = (out_chained - out_mono).abs().max().item()
        diff_state = (state2 - state_mono).abs().max().item()

        print(f"\nChunk Chaining L={L}: diff_out={diff_out:.2e}, diff_state={diff_state:.2e}")
        assert diff_out < 1.0e-4, f"Chained output diff {diff_out:.2e} exceeded 1e-4 at L={L}"
        assert diff_state < 1.0e-4, f"Chained state diff {diff_state:.2e} exceeded 1e-4 at L={L}"


class TestExtremeBoundaryAutograd:

    @pytest.mark.parametrize("L", [512, 1024])
    def test_autograd_long_sequence_backward(self, L: int):
        torch.manual_seed(9000 + L)
        B, H, dk, dv = 1, 2, 64, 64
        q_ref = torch.randn(B, H, L, dk, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k_ref = normalize_keys(torch.randn(B, H, L, dk, device=DEVICE, dtype=torch.float32)).detach().requires_grad_()
        v_ref = torch.randn(B, H, L, dv, device=DEVICE, dtype=torch.float32, requires_grad=True)
        a_ref = (torch.rand(B, H, L, dk, device=DEVICE, dtype=torch.float32) * 0.8 + 0.1).requires_grad_()
        b_ref = torch.rand(B, H, L, dk, device=DEVICE, dtype=torch.float32, requires_grad=True)
        w_ref = torch.rand(B, H, L, dv, device=DEVICE, dtype=torch.float32, requires_grad=True)
        s0_ref = torch.randn(B, H, dk, dv, device=DEVICE, dtype=torch.float32, requires_grad=True)

        q_tr = q_ref.detach().clone().requires_grad_()
        k_tr = k_ref.detach().clone().requires_grad_()
        v_tr = v_ref.detach().clone().requires_grad_()
        a_tr = a_ref.detach().clone().requires_grad_()
        b_tr = b_ref.detach().clone().requires_grad_()
        w_tr = w_ref.detach().clone().requires_grad_()
        s0_tr = s0_ref.detach().clone().requires_grad_()

        out_ref, s_ref = ref_dgda_prefill(q_ref, k_ref, v_ref, a_ref, b_ref, w_ref, initial_state=s0_ref)
        loss_ref = out_ref.sum() + s_ref.sum()
        loss_ref.backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        loss_tr = out_tr.sum() + s_tr.sum()
        loss_tr.backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q_ref), ("k", k_tr, k_ref), ("v", v_tr, v_ref),
            ("alpha", a_tr, a_ref), ("b", b_tr, b_ref), ("w", w_tr, w_ref), ("s0", s0_tr, s0_ref)
        ]:
            assert not torch.isnan(t_tr.grad).any(), f"NaN in {name} at L={L}"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in {name} at L={L}"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, f"Grad rel error for {name} was {rel_err:.2e} at L={L}"

    @pytest.mark.parametrize("scenario", ["alpha_near_1", "alpha_near_0", "b_zero", "b_one", "w_zero", "w_one"])
    def test_singular_parameter_regimes(self, scenario: str):
        torch.manual_seed(9500)
        B, H, L, dk, dv = 2, 2, 32, 64, 64
        q = torch.randn(B, H, L, dk, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = normalize_keys(torch.randn(B, H, L, dk, device=DEVICE, dtype=torch.float32)).detach().requires_grad_()
        v = torch.randn(B, H, L, dv, device=DEVICE, dtype=torch.float32, requires_grad=True)
        s0 = torch.randn(B, H, dk, dv, device=DEVICE, dtype=torch.float32, requires_grad=True)

        if scenario == "alpha_near_1":
            alpha = (torch.ones(B, H, L, dk, device=DEVICE) * 0.9999).requires_grad_()
            b = torch.rand(B, H, L, dk, device=DEVICE).requires_grad_()
            w = torch.rand(B, H, L, dv, device=DEVICE).requires_grad_()
        elif scenario == "alpha_near_0":
            alpha = (torch.ones(B, H, L, dk, device=DEVICE) * 1e-6).requires_grad_()
            b = torch.rand(B, H, L, dk, device=DEVICE).requires_grad_()
            w = torch.rand(B, H, L, dv, device=DEVICE).requires_grad_()
        elif scenario == "b_zero":
            alpha = (torch.rand(B, H, L, dk, device=DEVICE) * 0.8 + 0.1).requires_grad_()
            b = torch.zeros(B, H, L, dk, device=DEVICE).requires_grad_()
            w = torch.rand(B, H, L, dv, device=DEVICE).requires_grad_()
        elif scenario == "b_one":
            alpha = (torch.rand(B, H, L, dk, device=DEVICE) * 0.8 + 0.1).requires_grad_()
            b = torch.ones(B, H, L, dk, device=DEVICE).requires_grad_()
            w = torch.rand(B, H, L, dv, device=DEVICE).requires_grad_()
        elif scenario == "w_zero":
            alpha = (torch.rand(B, H, L, dk, device=DEVICE) * 0.8 + 0.1).requires_grad_()
            b = torch.rand(B, H, L, dk, device=DEVICE).requires_grad_()
            w = torch.zeros(B, H, L, dv, device=DEVICE).requires_grad_()
        elif scenario == "w_one":
            alpha = (torch.rand(B, H, L, dk, device=DEVICE) * 0.8 + 0.1).requires_grad_()
            b = torch.rand(B, H, L, dk, device=DEVICE).requires_grad_()
            w = torch.ones(B, H, L, dv, device=DEVICE).requires_grad_()

        q_tr = q.detach().clone().requires_grad_()
        k_tr = k.detach().clone().requires_grad_()
        v_tr = v.detach().clone().requires_grad_()
        a_tr = alpha.detach().clone().requires_grad_()
        b_tr = b.detach().clone().requires_grad_()
        w_tr = w.detach().clone().requires_grad_()
        s0_tr = s0.detach().clone().requires_grad_()

        out_ref, s_ref = ref_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        (out_ref.sum() + s_ref.sum()).backward()

        out_tr, s_tr = triton_dgda_prefill(q_tr, k_tr, v_tr, a_tr, b_tr, w_tr, initial_state=s0_tr)
        (out_tr.sum() + s_tr.sum()).backward()

        for name, t_tr, t_ref in [
            ("q", q_tr, q), ("k", k_tr, k), ("v", v_tr, v),
            ("alpha", a_tr, alpha), ("b", b_tr, b), ("w", w_tr, w), ("s0", s0_tr, s0)
        ]:
            assert not torch.isnan(t_tr.grad).any(), f"NaN in {name} during {scenario}"
            assert not torch.isinf(t_tr.grad).any(), f"Inf in {name} during {scenario}"
            rel_err = compute_rel_err(t_tr.grad, t_ref.grad)
            assert rel_err <= 1.0e-2, f"Rel error in {name} during {scenario}: {rel_err:.2e}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mixed_precision_autograd(self, dtype):
        if not CUDA_ONLINE:
            pytest.skip("Mixed precision autograd requires CUDA")

        torch.manual_seed(9700)
        B, H, L, dk, dv = 2, 4, 32, 64, 64
        q = torch.randn(B, H, L, dk, device=DEVICE, dtype=dtype, requires_grad=True)
        k = normalize_keys(torch.randn(B, H, L, dk, device=DEVICE, dtype=dtype)).detach().requires_grad_()
        v = torch.randn(B, H, L, dv, device=DEVICE, dtype=dtype, requires_grad=True)
        alpha = (torch.rand(B, H, L, dk, device=DEVICE, dtype=dtype) * 0.8 + 0.1).requires_grad_()
        b = torch.rand(B, H, L, dk, device=DEVICE, dtype=dtype, requires_grad=True)
        w = torch.rand(B, H, L, dv, device=DEVICE, dtype=dtype, requires_grad=True)
        s0 = torch.randn(B, H, dk, dv, device=DEVICE, dtype=dtype, requires_grad=True)

        out, state = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        loss = out.sum() + state.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w), ("s0", s0)]:
            assert t.grad is not None, f"Gradient for {name} is None in {dtype}"
            assert not torch.isnan(t.grad).any(), f"NaN in {name} in {dtype}"
            assert not torch.isinf(t.grad).any(), f"Inf in {name} in {dtype}"
            assert t.grad.dtype == dtype, f"Gradient dtype mismatch for {name}: {t.grad.dtype} != {dtype}"

    def test_selective_subsets_requires_grad(self):
        B, H, L, dk, dv = 2, 2, 32, 64, 64
        q = torch.randn(B, H, L, dk, device=DEVICE, dtype=torch.float32, requires_grad=True)
        k = normalize_keys(torch.randn(B, H, L, dk, device=DEVICE, dtype=torch.float32))
        v = torch.randn(B, H, L, dv, device=DEVICE, dtype=torch.float32, requires_grad=False)
        alpha = (torch.rand(B, H, L, dk, device=DEVICE, dtype=torch.float32) * 0.8 + 0.1).requires_grad_()
        b = torch.rand(B, H, L, dk, device=DEVICE, dtype=torch.float32, requires_grad=False)
        w = torch.rand(B, H, L, dv, device=DEVICE, dtype=torch.float32, requires_grad=True)
        s0 = torch.randn(B, H, dk, dv, device=DEVICE, dtype=torch.float32, requires_grad=False)

        out, s = triton_dgda_prefill(q, k, v, alpha, b, w, initial_state=s0)
        loss = out.sum() + s.sum()
        loss.backward()

        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert alpha.grad is not None and not torch.isnan(alpha.grad).any()
        assert w.grad is not None and not torch.isnan(w.grad).any()
        assert k.grad is None
        assert v.grad is None
        assert b.grad is None
        assert s0.grad is None
