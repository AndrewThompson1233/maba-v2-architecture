
import math
import pytest
import torch

from maba_sparse.layers.sparse_attention import MABASALayer, MabaSparseAttention




class TestMLACompression:

    def test_mla_compression_dimension(self, config):
        layer = MabaSparseAttention(config)
        assert layer.d_c == 128
        assert layer.kv_down_proj.out_features == 128
        assert layer.k_up_proj.in_features == 128
        assert layer.v_up_proj.in_features == 128

        B, L, D = 2, 64, config.dim
        x = torch.randn(B, L, D)
        _, c_kv = layer(x)

        assert c_kv.shape == (B, L, 128), f"Expected c_kv shape (2, 64, 128), got {c_kv.shape}"

    def test_mla_compression_ratio(self, config):
        layer = MabaSparseAttention(config)
        full_kv_dim = layer.n_heads * layer.d_head
        compressed_dim = layer.d_c
        compression_ratio = 1.0 - (compressed_dim / full_kv_dim)

        assert math.isclose(
            compression_ratio, 0.80, abs_tol=1e-5
        ), f"Expected 80% compression, got {compression_ratio * 100:.2f}%"

    def test_strict_nope_attention(self, config):
        layer = MabaSparseAttention(config)

        param_names = [name.lower() for name, _ in layer.named_parameters()]
        for name in param_names:
            assert "rope" not in name, f"Found positional parameter: {name}"
            assert "rotary" not in name, f"Found positional parameter: {name}"
            assert "pos_emb" not in name, f"Found positional parameter: {name}"

        buffer_names = [name.lower() for name, _ in layer.named_buffers()]
        for name in buffer_names:
            assert "rope" not in name, f"Found positional buffer: {name}"
            assert "rotary" not in name, f"Found positional buffer: {name}"




class Test3StreamSuperpositionGates:

    def test_gate_weights_sum_to_one(self, small_config):
        layer = MabaSparseAttention(small_config)
        layer.eval()

        B, L, D = 2, 128, small_config.dim
        x = torch.randn(B, L, D)

        gate_weights = layer.get_gate_weights(x)

        assert gate_weights.shape == (B, L, 3)
        assert (gate_weights >= 0.0).all(), "Gate weights must be non-negative"

        gate_sum = gate_weights.sum(dim=-1)
        torch.testing.assert_close(
            gate_sum,
            torch.ones_like(gate_sum),
            rtol=1e-5,
            atol=1e-6,
            msg="Gate weights must sum strictly to 1.0",
        )

    def test_gate_gradient_flow(self, small_config):
        layer = MabaSparseAttention(small_config)
        layer.train()

        B, L, D = 2, 64, small_config.dim
        x = torch.randn(B, L, D, requires_grad=True)

        out, _ = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.stream_gate.weight.grad is not None, "stream_gate.weight grad is None"
        assert layer.stream_gate.bias.grad is not None, "stream_gate.bias grad is None"
        assert layer.stream_gate.weight.grad.norm().item() > 0.0, "stream_gate.weight grad is zero"
        assert layer.stream_gate.bias.grad.norm().item() > 0.0, "stream_gate.bias grad is zero"

    def test_superposition_individual_streams(self, small_config):
        layer = MabaSparseAttention(small_config)
        layer.eval()

        B, L, D = 1, 128, small_config.dim
        x = torch.randn(B, L, D)

        c_kv = layer.kv_down_proj(x)
        q = layer.q_proj(x).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
        k = layer.k_up_proj(c_kv).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
        v = layer.v_up_proj(c_kv).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)

        out_local = layer._compute_local_attention(q, k, v, L)
        assert out_local.shape == (B, layer.n_heads, L, layer.d_head)
        assert not torch.isnan(out_local).any()

        top_indices, _ = layer.indexer(x)
        out_sparse = layer._compute_sparse_attention(q, k, v, top_indices, L)
        assert out_sparse.shape == (B, layer.n_heads, L, layer.d_head)
        assert not torch.isnan(out_sparse).any()

        out_hca = layer._compute_hca_attention(q, c_kv, L)
        assert out_hca.shape == (B, layer.n_heads, L, layer.d_head)
        assert not torch.isnan(out_hca).any()




class TestHCADocumentCoverage:

    @pytest.mark.parametrize("L", [1, 15, 32, 63, 64, 65, 128, 256, 512, 1024])
    def test_hca_full_context_coverage(self, small_config, L):
        layer = MabaSparseAttention(small_config)
        layer.eval()

        B, D = 1, small_config.dim
        x = torch.randn(B, L, D)

        out, c_kv = layer(x)

        assert out.shape == (B, L, D)
        assert c_kv.shape == (B, L, layer.d_c)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_hca_causal_masking_structure(self, small_config):
        layer = MabaSparseAttention(small_config, hca_pool_size=64)
        layer.eval()

        L = 192
        x = torch.randn(1, L, small_config.dim)

        c_kv = layer.kv_down_proj(x)
        q = layer.q_proj(x).view(1, L, layer.n_heads, layer.d_head).transpose(1, 2)

        out_hca = layer._compute_hca_attention(q[:, :, :10, :], c_kv, L)
        assert out_hca.shape == (1, layer.n_heads, 10, layer.d_head)
        assert not torch.isnan(out_hca).any()




class TestLocalSlidingWindowAndSinks:

    def test_attention_sinks_unconditionally_preserved(self, small_config):
        window_size = 16
        layer = MabaSparseAttention(small_config, window_size=window_size)
        layer.eval()

        L = 64

        q_pos = 50
        i_idx = torch.arange(L).unsqueeze(1)
        j_idx = torch.arange(L).unsqueeze(0)
        causal = j_idx <= i_idx
        in_window = (i_idx - j_idx) < window_size
        in_sinks = j_idx < 4
        allowed = causal & (in_window | in_sinks)

        assert allowed[q_pos, 0].item() is True
        assert allowed[q_pos, 1].item() is True
        assert allowed[q_pos, 2].item() is True
        assert allowed[q_pos, 3].item() is True

        assert allowed[q_pos, 10].item() is False
        assert allowed[q_pos, 20].item() is False

        assert allowed[q_pos, 45].item() is True
        assert allowed[q_pos, 50].item() is True




class TestCausalMaskingIsolation:

    def test_output_causal_invariance(self, small_config):
        layer = MabaSparseAttention(small_config)
        layer.eval()

        B, L, D = 1, 64, small_config.dim
        x1 = torch.randn(B, L, D)
        x2 = x1.clone()

        target_t = 30
        x2[:, target_t + 1:, :] = torch.randn_like(x2[:, target_t + 1:, :]) * 10.0

        with torch.no_grad():
            out1, _ = layer(x1)
            out2, _ = layer(x2)

        torch.testing.assert_close(
            out1[:, :target_t + 1, :],
            out2[:, :target_t + 1, :],
            rtol=1e-5,
            atol=1e-6,
            msg=f"Future tokens corrupted output at or before token {target_t}",
        )

    def test_causal_gradient_jacobian_isolation(self, small_config):
        layer = MabaSparseAttention(small_config)
        layer.eval()

        B, L, D = 1, 64, small_config.dim
        x = torch.randn(B, L, D, requires_grad=True)

        target_t = 25
        out, _ = layer(x)

        loss = out[:, target_t, :].sum()
        loss.backward()

        assert x.grad is not None
        future_grads = x.grad[:, target_t + 1:, :]
        past_and_target_grads = x.grad[:, :target_t + 1, :]

        assert (
            future_grads == 0.0
        ).all(), f"Gradient leaked into future tokens: max |grad| = {future_grads.abs().max().item():.6e}"

        assert past_and_target_grads.abs().max().item() > 0.0, "Target token gradient is zero"




class TestKVCacheAndDecode:

    def test_kv_cache_concatenation_shape(self, small_config):
        layer = MabaSparseAttention(small_config)
        layer.eval()

        B, D = 2, small_config.dim
        L_past = 30
        L_new = 10

        x_past = torch.randn(B, L_past, D)
        _, past_c_kv = layer(x_past)
        assert past_c_kv.shape == (B, L_past, layer.d_c)

        x_new = torch.randn(B, L_new, D)
        out_new, full_c_kv = layer(x_new, past_c_kv=past_c_kv)

        assert out_new.shape == (B, L_new, D)
        assert full_c_kv.shape == (B, L_past + L_new, layer.d_c)

    def test_step_by_step_decode_vs_batched_prefill(self, small_config):
        torch.manual_seed(42)
        layer = MabaSparseAttention(small_config, window_size=64, block_size=16)
        layer.eval()

        B, L, D = 1, 16, small_config.dim
        x = torch.randn(B, L, D)

        with torch.no_grad():
            batched_out, _ = layer(x)

        step_outputs = []
        past_c_kv = None
        with torch.no_grad():
            for t in range(L):
                xt = x[:, t:t + 1, :]
                ot, past_c_kv = layer(xt, past_c_kv=past_c_kv)
                step_outputs.append(ot)

        sequential_out = torch.cat(step_outputs, dim=1)

        torch.testing.assert_close(
            sequential_out,
            batched_out,
            rtol=1e-4,
            atol=1e-4,
            msg="Step-by-step decode diverged from batched prefill",
        )




class TestAttentionAnalyticalGradientContinuity:

    def test_backward_all_parameters_finite_and_nonzero(self, small_config):
        torch.manual_seed(777)
        layer = MabaSparseAttention(small_config)
        layer.train()

        B, L, D = 2, 32, small_config.dim
        x = torch.randn(B, L, D, requires_grad=True)

        out, _ = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input x.grad is None"
        assert not torch.isnan(x.grad).any(), "Input x.grad contains NaNs"
        assert not torch.isinf(x.grad).any(), "Input x.grad contains Infs"
        assert torch.count_nonzero(x.grad) > 0, "Input x.grad is entirely zero"

        core_param_names = [
            "q_proj.weight",
            "kv_down_proj.weight",
            "k_up_proj.weight",
            "v_up_proj.weight",
            "o_proj.weight",
            "stream_gate.weight",
            "stream_gate.bias",
        ]

        for name, param in layer.named_parameters():
            if name in core_param_names:
                assert param.grad is not None, f"Param {name} grad is None"
                assert not torch.isnan(param.grad).any(), f"Param {name} grad has NaNs"
                assert not torch.isinf(param.grad).any(), f"Param {name} grad has Infs"
                grad_norm = param.grad.norm().item()
                assert grad_norm > 0.0, f"Param {name} grad norm is zero ({grad_norm})"
                assert math.isfinite(
                    grad_norm
                ), f"Param {name} grad norm is not finite: {grad_norm}"




class TestAttentionNumericalStabilityAndAliases:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_attention_numerical_stability(self, small_config, dtype):
        layer = MabaSparseAttention(small_config).to(dtype)
        x = torch.randn(1, 64, small_config.dim, dtype=dtype)

        out, c_kv = layer(x)
        assert not torch.isnan(out).any(), f"NaN detected in output for {dtype}"
        assert not torch.isinf(out).any(), f"Inf detected in output for {dtype}"
        assert out.dtype == dtype
        assert c_kv.dtype == dtype

    def test_alias_parity(self):
        assert MABASALayer is MabaSparseAttention
        layer = MABASALayer(dim=64, n_heads=2, d_head=32, d_c=16)
        assert layer.dim == 64
        assert layer.d_c == 16
