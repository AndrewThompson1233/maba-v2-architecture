
import gc
import time

import pytest
import torch

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer, ConvState
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config
from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.kernels.dispatcher import is_cuda_sm75_available

CUDA_AVAILABLE = torch.cuda.is_available() and is_cuda_sm75_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


class TestDGDAStepMemoryInvarianceAdversarial:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    @pytest.mark.parametrize("history_len", [0, 64, 256, 512, 1024, 2048, 4096])
    def test_dgda_step_memory_invariance_across_histories(self, history_len: int):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=1)
        layer = DGDALayer(cfg).to(DEVICE).eval()

        B, D = 1, cfg.dim
        expected_state_bytes = B * cfg.n_heads * cfg.d_head * cfg.d_head * 4

        if history_len > 0:
            x_hist = torch.randn(B, history_len, D, device=DEVICE)
            with torch.no_grad():
                _, init_state, init_conv = layer(x_hist)
        else:
            init_state, init_conv = None, None

        if init_state is not None:
            actual_bytes = init_state.element_size() * init_state.nelement()
            assert actual_bytes == expected_state_bytes, (
                f"History L={history_len}: Initial state size {actual_bytes} != expected {expected_state_bytes}"
            )
            assert list(init_state.shape) == [B, cfg.n_heads, cfg.d_head, cfg.d_head]

        x_step = torch.randn(B, 1, D, device=DEVICE)

        curr_state = init_state.clone() if init_state is not None else None
        curr_conv = ConvState(t.clone() for t in init_conv) if init_conv is not None else None

        with torch.no_grad():
            for _ in range(5):
                _, curr_state, curr_conv = layer.step(x_step, state=curr_state, conv_state=curr_conv)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        num_steps = 128
        latencies = []
        with torch.no_grad():
            for s in range(num_steps):
                t0 = time.perf_counter()
                _, curr_state, curr_conv = layer.step(x_step, state=curr_state, conv_state=curr_conv)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()
        mem_growth = mem_end - mem_start

        final_state_bytes = curr_state.element_size() * curr_state.nelement()
        assert final_state_bytes == expected_state_bytes, (
            f"History L={history_len}: State size after {num_steps} steps is {final_state_bytes}, expected {expected_state_bytes}"
        )
        assert list(curr_state.shape) == [B, cfg.n_heads, cfg.d_head, cfg.d_head]

        assert mem_growth == 0, (
            f"History L={history_len}: DGDALayer.step leaked {mem_growth} bytes over {num_steps} decode steps!"
        )

        avg_lat = sum(latencies) / len(latencies)
        print(f"\n[PASS] History L={history_len:4d}: State={final_state_bytes} B | "
              f"Steps={num_steps} | Mem Growth={mem_growth} B | Step Latency={avg_lat:.3f} ms")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    def test_dgda_step_extreme_decoding_250_steps(self):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=1)
        layer = DGDALayer(cfg).to(DEVICE).eval()

        B, D = 1, cfg.dim
        x_hist = torch.randn(B, 1024, D, device=DEVICE)
        with torch.no_grad():
            _, state, conv = layer(x_hist)

        x_step = torch.randn(B, 1, D, device=DEVICE)
        with torch.no_grad():
            for _ in range(5):
                _, state, conv = layer.step(x_step, state=state, conv_state=conv)
            torch.cuda.synchronize()

        mem_checkpoints = {}
        check_steps = [10, 50, 100, 150, 200, 250]

        torch.cuda.synchronize()
        mem_initial = torch.cuda.memory_allocated()

        with torch.no_grad():
            for s in range(1, 251):
                _, state, conv = layer.step(x_step, state=state, conv_state=conv)
                if s in check_steps:
                    torch.cuda.synchronize()
                    mem_checkpoints[s] = torch.cuda.memory_allocated()

        for s in check_steps:
            delta = mem_checkpoints[s] - mem_initial
            assert delta == 0, f"Memory drift at step {s}: {delta} bytes (expected strictly 0)"

        print(f"\n[PASS] 250-Step Extreme Decoding: Zero memory drift at checkpoints {check_steps}")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_dgda_step_variable_batch_and_head_dims(self, batch_size: int, head_dim: int):
        torch.cuda.empty_cache()
        gc.collect()

        n_heads = 4
        dim = n_heads * head_dim
        cfg = MabaSparseConfig(
            dim=dim,
            n_heads=n_heads,
            d_head=head_dim,
            n_layers=1,
            vocab_size=1000,
            d_emb=head_dim,
            intermediate_size=dim * 2,
        )
        layer = DGDALayer(cfg).to(DEVICE).eval()

        expected_bytes = batch_size * n_heads * head_dim * head_dim * 4
        x_step = torch.randn(batch_size, 1, dim, device=DEVICE)

        state = None
        conv = None

        with torch.no_grad():
            for _ in range(5):
                _, state, conv = layer.step(x_step, state=state, conv_state=conv)
            torch.cuda.synchronize()

        mem_start = torch.cuda.memory_allocated()
        with torch.no_grad():
            for _ in range(100):
                _, state, conv = layer.step(x_step, state=state, conv_state=conv)
            torch.cuda.synchronize()

        mem_end = torch.cuda.memory_allocated()
        mem_growth = mem_end - mem_start

        actual_state_bytes = state.element_size() * state.nelement()
        assert actual_state_bytes == expected_bytes, (
            f"B={batch_size}, dk={head_dim}: State bytes {actual_state_bytes} != expected {expected_bytes}"
        )
        assert mem_growth == 0, (
            f"B={batch_size}, dk={head_dim}: Leaked {mem_growth} bytes over 100 steps"
        )
        print(f"\n[PASS] B={batch_size}, dk={head_dim}: State={actual_state_bytes} B | Mem Growth={mem_growth} B")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    def test_dgda_step_adversarial_input_patterns(self):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=1)
        layer = DGDALayer(cfg).to(DEVICE).eval()

        B, D = 1, cfg.dim
        state = None
        conv = None

        adversarial_patterns = [
            ("all_zeros", torch.zeros(B, 1, D, device=DEVICE)),
            ("near_zero", torch.full((B, 1, D), 1e-7, device=DEVICE)),
            ("large_values", torch.full((B, 1, D), 50.0, device=DEVICE)),
            ("alternating", torch.tensor([1.0, -1.0] * (D // 2), device=DEVICE).view(B, 1, D)),
            ("random_high_var", torch.randn(B, 1, D, device=DEVICE) * 10.0),
        ]

        with torch.no_grad():
            for _ in range(5):
                _, state, conv = layer.step(torch.randn(B, 1, D, device=DEVICE), state=state, conv_state=conv)
            torch.cuda.synchronize()

        mem_start = torch.cuda.memory_allocated()

        with torch.no_grad():
            for pat_name, pat_tensor in adversarial_patterns:
                for _ in range(25):
                    out, state, conv = layer.step(pat_tensor, state=state, conv_state=conv)
                    assert torch.isfinite(out).all(), f"NaN or Inf in output under pattern {pat_name}!"
                    assert torch.isfinite(state).all(), f"NaN or Inf in recurrent state under pattern {pat_name}!"
                torch.cuda.synchronize()

        mem_end = torch.cuda.memory_allocated()
        assert mem_end - mem_start == 0, f"Memory leak under adversarial input patterns: {mem_end - mem_start} bytes"
        print(f"\n[PASS] Adversarial Inputs: Zero NaNs/Infs and 0 bytes leak across 125 steps")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    def test_dgda_forward_vs_step_decode_consistency(self):
        cfg = get_101m_config(n_layers=1)
        layer = DGDALayer(cfg).to(DEVICE).eval()

        B, D = 1, cfg.dim
        x_prefill = torch.randn(B, 64, D, device=DEVICE)

        with torch.no_grad():
            _, s_init, c_init = layer(x_prefill)

        s_step = s_init.clone()
        c_step = ConvState(t.clone() for t in c_init)
        s_fwd = s_init.clone()
        c_fwd = ConvState(t.clone() for t in c_init)

        torch.manual_seed(42)
        step_tokens = [torch.randn(B, 1, D, device=DEVICE) for _ in range(50)]

        max_out_diff = 0.0
        max_state_diff = 0.0

        with torch.no_grad():
            for tok in step_tokens:
                o_s, s_step, c_step = layer.step(tok, state=s_step, conv_state=c_step)
                o_f, s_fwd, c_fwd = layer(tok, state=s_fwd, conv_state=c_fwd)

                diff_o = (o_s - o_f).abs().max().item()
                diff_s = (s_step - s_fwd).abs().max().item()
                max_out_diff = max(max_out_diff, diff_o)
                max_state_diff = max(max_state_diff, diff_s)

        print(f"\n[PASS] Forward vs Step Consistency over 50 steps: max out diff={max_out_diff:.6e}, max state diff={max_state_diff:.6e}")
        assert max_out_diff < 5e-4, f"Output divergence between forward and step: {max_out_diff}"
        assert max_state_diff < 5e-4, f"State divergence between forward and step: {max_state_diff}"


class TestFullModelDecodeMemoryScalingAdversarial:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    @pytest.mark.parametrize("history_len", [64, 256, 512, 1024, 2048])
    def test_full_model_decode_across_diverse_histories(self, history_len: int):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=20)
        model = MabaSparseForCausalLM(cfg).to(DEVICE).eval()

        B = 1
        prompt = torch.randint(1, cfg.vocab_size, (B, history_len), device=DEVICE)

        with torch.no_grad():
            out_prefill = model(prompt)
            past_states = out_prefill.past_states

        assert len(past_states) == 20
        dgda_layers_count = 0
        attn_layers_count = 0

        dgda_initial_bytes = 0
        for i, s in enumerate(past_states):
            st, cv, pk = s
            if model.layers[i].is_attention:
                attn_layers_count += 1
                assert st is None and cv is None, f"Attention layer {i} should not have DGDA state!"
                assert pk is not None, f"Attention layer {i} must have MLA past_c_kv!"
                assert pk.shape == (B, history_len, cfg.d_c)
            else:
                dgda_layers_count += 1
                assert pk is None, f"DGDA layer {i} should not have MLA past_c_kv!"
                assert st is not None and cv is not None, f"DGDA layer {i} must have state and conv_state!"
                dgda_initial_bytes += st.element_size() * st.nelement()

        assert dgda_layers_count == 15, f"Expected 15 DGDA layers (75%), got {dgda_layers_count}"
        assert attn_layers_count == 5, f"Expected 5 Attention layers (25%), got {attn_layers_count}"
        assert dgda_initial_bytes == 15 * 163840, f"Expected 2,457,600 bytes DGDA state, got {dgda_initial_bytes}"

        step_tok = torch.randint(1, cfg.vocab_size, (B, 1), device=DEVICE)

        curr_states = past_states
        with torch.no_grad():
            for _ in range(3):
                sout = model(step_tok, past_states=curr_states)
                curr_states = sout.past_states
            torch.cuda.synchronize()

        mem_start = torch.cuda.memory_allocated()

        num_decode_steps = 100
        latencies = []
        dgda_state_bytes_history = []

        with torch.no_grad():
            for step in range(1, num_decode_steps + 1):
                t0 = time.perf_counter()
                sout = model(step_tok, past_states=curr_states)
                curr_states = sout.past_states
                step_tok = torch.argmax(sout.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                step_dgda_bytes = 0
                for s in curr_states:
                    st, cv, pk = s
                    if st is not None:
                        step_dgda_bytes += st.element_size() * st.nelement()
                dgda_state_bytes_history.append(step_dgda_bytes)

        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()
        net_mem_growth = mem_end - mem_start

        assert len(set(dgda_state_bytes_history)) == 1, (
            f"DGDA recurrent state bytes varied across decode steps: {set(dgda_state_bytes_history)}"
        )
        assert dgda_state_bytes_history[0] == 15 * 163840, (
            f"DGDA state bytes {dgda_state_bytes_history[0]} != expected {15 * 163840}"
        )

        print(f"\n[PASS] Full Model (L={history_len:4d}, {num_decode_steps} steps): "
              f"DGDA State={dgda_state_bytes_history[0]/1024:.1f} KB (STRICT CONSTANT) | "
              f"Net Mem Growth={net_mem_growth/1024:.1f} KB | Mean Latency={sum(latencies)/len(latencies):.2f} ms/step")
        assert net_mem_growth < 3 * 1024 * 1024, (
            f"Full model memory grew excessively: {net_mem_growth / (1024*1024):.2f} MB"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ GPU")
    def test_full_model_vs_dense_kv_compression_scaling(self):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=20)
        maba_model = MabaSparseForCausalLM(cfg).to(DEVICE).eval()
        dense_model = DenseTransformerForCausalLM().to(DEVICE).eval()

        contexts = [128, 512, 1024, 2048]
        decode_steps = 100

        print("\n--- Cache Compression Scaling Benchmark (Prompt L + 100 Decode Steps) ---")

        for L in contexts:
            prompt = torch.randint(1, cfg.vocab_size, (1, L), device=DEVICE)
            step_tok = torch.randint(1, cfg.vocab_size, (1, 1), device=DEVICE)

            with torch.no_grad():
                out_maba = maba_model(prompt)
                maba_past = out_maba.past_states

                out_dense = dense_model(prompt)
                dense_past = out_dense.past_states

            with torch.no_grad():
                for _ in range(decode_steps):
                    maba_past = maba_model(step_tok, past_states=maba_past).past_states
                    dense_past = dense_model(step_tok, past_states=dense_past).past_states

            maba_dgda_bytes = sum(s[0].element_size() * s[0].nelement() for s in maba_past if s[0] is not None)
            maba_mla_bytes = sum(s[2].element_size() * s[2].nelement() for s in maba_past if s[2] is not None)
            maba_total = maba_dgda_bytes + maba_mla_bytes

            dense_total = 0
            for s in dense_past:
                if isinstance(s, (tuple, list)):
                    for t in s:
                        if isinstance(t, torch.Tensor):
                            dense_total += t.element_size() * t.nelement()

            compression_ratio = dense_total / max(maba_total, 1)

            print(f"Total Seq (L={L}+{decode_steps}={L+decode_steps:4d}) | "
                  f"DGDA Recurrent: {maba_dgda_bytes/1024:6.1f} KB (CONST) | "
                  f"Maba MLA Cache: {maba_mla_bytes/1024:6.1f} KB | "
                  f"Total Maba: {maba_total/1024:7.1f} KB | "
                  f"Dense KV: {dense_total/1024:9.1f} KB | "
                  f"Compression: {compression_ratio:5.1f}x")

            if L >= 1024:
                assert compression_ratio > 15.0, f"Compression ratio {compression_ratio} is below target 15x!"

        print("[PASS] Cache Compression Scaling successfully verified across all contexts.")
