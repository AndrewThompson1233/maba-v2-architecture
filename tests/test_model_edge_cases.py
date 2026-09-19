
import pytest
import torch

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.layers.indexer import DGIndexer
from maba_sparse.layers.sparse_attention import MabaSparseAttention
from maba_sparse.model import MabaSparseForCausalLM


@pytest.fixture
def base_config():
    return MabaSparseConfig(
        dim=64,
        n_heads=2,
        d_head=32,
        n_layers=4,
        vocab_size=1000,
        d_emb=32,
        intermediate_size=128,
        window_size=32,
        block_size=16,
        top_k=4,
        hca_pool_size=16,
        chunk_size=16,
        inversion_method="adaptive",
    )


class TestPrefillStepEquivalence:

    @pytest.mark.parametrize("L", [64, 128, 256])
    @pytest.mark.parametrize("chunk_size", [8, 16, 32])
    @pytest.mark.parametrize("has_init_state", [False, True])
    def test_dgda_prefill_vs_step_loop(self, base_config, L, chunk_size, has_init_state):
        torch.manual_seed(1000 + L + chunk_size + (10 if has_init_state else 0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer = DGDALayer(base_config).to(device).eval()

        B = 2
        x = torch.randn(B, L, base_config.dim, device=device)
        init_state = (
            torch.randn(B, base_config.n_heads, base_config.d_head, base_config.d_head, device=device)
            if has_init_state
            else None
        )

        with torch.no_grad():
            out_fwd, state_fwd, conv_fwd = layer.forward(
                x,
                state=init_state.clone() if init_state is not None else None,
                chunk_size=chunk_size,
            )

        curr_state = init_state.clone() if init_state is not None else None
        curr_conv = None
        step_outs = []
        with torch.no_grad():
            for t in range(L):
                xt = x[:, t : t + 1, :]
                ot, curr_state, curr_conv = layer.step(
                    xt, state=curr_state, conv_state=curr_conv
                )
                step_outs.append(ot)

        out_step = torch.cat(step_outs, dim=1)

        diff_out = (out_fwd - out_step).abs().max().item()
        diff_state = (state_fwd - curr_state).abs().max().item()

        tol = 1e-4
        assert (
            diff_out < tol
        ), f"Prefill vs step output difference {diff_out:.8e} exceeds {tol} for L={L}, C={chunk_size}"
        assert (
            diff_state < tol
        ), f"Final state difference {diff_state:.8e} exceeds {tol} for L={L}, C={chunk_size}"

    @pytest.mark.parametrize("odd_L", [3, 17, 33, 67, 129])
    def test_dgda_prefill_vs_step_odd_lengths(self, base_config, odd_L):
        torch.manual_seed(2000 + odd_L)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer = DGDALayer(base_config).to(device).eval()
        B = 2
        x = torch.randn(B, odd_L, base_config.dim, device=device)

        with torch.no_grad():
            out_fwd, state_fwd, _ = layer.forward(x, chunk_size=16)

            curr_state = None
            curr_conv = None
            step_outs = []
            for t in range(odd_L):
                ot, curr_state, curr_conv = layer.step(
                    x[:, t : t + 1, :], state=curr_state, conv_state=curr_conv
                )
                step_outs.append(ot)
            out_step = torch.cat(step_outs, dim=1)

        diff_out = (out_fwd - out_step).abs().max().item()
        diff_state = (state_fwd - curr_state).abs().max().item()
        assert diff_out < 1e-4, f"Odd length L={odd_L} output diff {diff_out:.8e} >= 1e-4"
        assert diff_state < 1e-4, f"Odd length L={odd_L} state diff {diff_state:.8e} >= 1e-4"


class TestCausalIsolationAndMasking:

    def test_dgda_backward_gradient_isolation(self, base_config):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device)

        L = 64
        t_loss = 25
        x = torch.randn(2, L, base_config.dim, device=device, requires_grad=True)

        out, _, _ = layer(x, chunk_size=16)
        loss = out[:, t_loss, :].sum()
        loss.backward()

        future_grads = x.grad[:, t_loss + 1 :, :]
        max_future_leak = future_grads.abs().max().item()
        assert (
            max_future_leak == 0.0
        ), f"DGDALayer future gradient leakage detected: {max_future_leak:.8e} != 0.0"

    def test_sparse_attention_backward_gradient_isolation(self, base_config):
        torch.manual_seed(43)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sa_layer = MabaSparseAttention(base_config).to(device)

        L = 64
        t_loss = 20
        x = torch.randn(2, L, base_config.dim, device=device, requires_grad=True)

        out, _ = sa_layer(x)
        loss = out[:, t_loss, :].sum()
        loss.backward()

        future_grads = x.grad[:, t_loss + 1 :, :]
        max_future_leak = future_grads.abs().max().item()
        assert (
            max_future_leak == 0.0
        ), f"MabaSparseAttention future gradient leakage detected: {max_future_leak:.8e} != 0.0"

    def test_full_model_backward_gradient_isolation(self, base_config):
        torch.manual_seed(44)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MabaSparseForCausalLM(base_config).to(device)

        L = 64
        t_loss = 30
        inp = torch.randint(0, base_config.vocab_size, (2, L), device=device)
        x_emb = model.embeddings(inp).detach().requires_grad_(True)

        h = x_emb
        for layer in model.layers:
            h, _, _, _ = layer(h)
        logits = model.lm_head(model.head_proj(model.final_norm(h)))

        loss = logits[:, t_loss, :].sum()
        loss.backward()

        future_grads = x_emb.grad[:, t_loss + 1 :, :]
        max_future_leak = future_grads.abs().max().item()
        assert (
            max_future_leak == 0.0
        ), f"Full model future gradient leakage detected: {max_future_leak:.8e} != 0.0"

    def test_forward_activation_causal_invariance(self, base_config):
        torch.manual_seed(45)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dgda = DGDALayer(base_config).to(device).eval()
        sa = MabaSparseAttention(base_config).to(device).eval()
        model = MabaSparseForCausalLM(base_config).to(device).eval()

        L = 64
        t_pert = 32

        x = torch.randn(2, L, base_config.dim, device=device)
        x_pert = x.clone()
        x_pert[:, t_pert:, :] += 50.0

        with torch.no_grad():
            o1_d, _, _ = dgda(x, chunk_size=16)
            o2_d, _, _ = dgda(x_pert, chunk_size=16)
            diff_d = (o1_d[:, :t_pert, :] - o2_d[:, :t_pert, :]).abs().max().item()
            assert diff_d == 0.0, f"DGDA forward causal leak: {diff_d:.8e} != 0.0"

            o1_sa, _ = sa(x)
            o2_sa, _ = sa(x_pert)
            diff_sa = (o1_sa[:, :t_pert, :] - o2_sa[:, :t_pert, :]).abs().max().item()
            assert diff_sa == 0.0, f"Sparse Attention forward causal leak: {diff_sa:.8e} != 0.0"

            inp = torch.randint(0, base_config.vocab_size, (2, L), device=device)
            inp_pert = inp.clone()
            inp_pert[:, t_pert:] = torch.randint(0, base_config.vocab_size, (2, L - t_pert), device=device)
            m1 = model(inp)
            m2 = model(inp_pert)
            diff_m = (m1.logits[:, :t_pert, :] - m2.logits[:, :t_pert, :]).abs().max().item()
            assert diff_m == 0.0, f"Model forward causal leak: {diff_m:.8e} != 0.0"


class TestExtremeDecayAndKeyGeometry:

    def test_instant_decay_alpha_zero(self, base_config):
        torch.manual_seed(46)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device).eval()

        with torch.no_grad():
            layer.gate_alpha.weight.fill_(100.0)

        L = 64
        x = torch.randn(2, L, base_config.dim, device=device).abs() + 0.1

        with torch.no_grad():
            ofwd, sfwd, _ = layer.forward(x, chunk_size=16)

            curr_s = None
            curr_c = None
            step_outs = []
            for t in range(L):
                ot, curr_s, curr_c = layer.step(x[:, t : t + 1], state=curr_s, conv_state=curr_c)
                step_outs.append(ot)
            ostep = torch.cat(step_outs, dim=1)

        diff_o = (ofwd - ostep).abs().max().item()
        assert not torch.isnan(ofwd).any(), "NaN detected in alpha -> 0 forward pass"
        assert diff_o < 1e-4, f"alpha -> 0 diff {diff_o:.8e} exceeds 1e-4"

    def test_persistent_decay_alpha_one(self, base_config):
        torch.manual_seed(47)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device).eval()

        with torch.no_grad():
            layer.gate_alpha.weight.fill_(-100.0)

        L = 64
        x = torch.randn(2, L, base_config.dim, device=device).abs() + 0.1

        with torch.no_grad():
            ofwd, sfwd, _ = layer.forward(x, chunk_size=16, inversion_method="adaptive")

            curr_s = None
            curr_c = None
            step_outs = []
            for t in range(L):
                ot, curr_s, curr_c = layer.step(x[:, t : t + 1], state=curr_s, conv_state=curr_c)
                step_outs.append(ot)
            ostep = torch.cat(step_outs, dim=1)

        diff_o = (ofwd - ostep).abs().max().item()
        assert not torch.isnan(ofwd).any(), "NaN detected in alpha -> 1 forward pass"
        assert diff_o < 1e-4, f"alpha -> 1 diff {diff_o:.8e} exceeds 1e-4"

    def test_collinear_adversarial_keys(self, base_config):
        torch.manual_seed(48)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device).eval()

        L = 64
        x_base = torch.randn(2, 1, base_config.dim, device=device)
        x = x_base.repeat(1, L, 1)

        with torch.no_grad():
            ofwd, sfwd, _ = layer.forward(x, chunk_size=16, inversion_method="adaptive")

            curr_s = None
            curr_c = None
            step_outs = []
            for t in range(L):
                ot, curr_s, curr_c = layer.step(x[:, t : t + 1], state=curr_s, conv_state=curr_c)
                step_outs.append(ot)
            ostep = torch.cat(step_outs, dim=1)

        diff_o = (ofwd - ostep).abs().max().item()
        assert not torch.isnan(ofwd).any(), "NaN detected under collinear keys"
        assert diff_o < 1e-4, f"Collinear key diff {diff_o:.8e} exceeds 1e-4"

    def test_extreme_gates_zero_one(self, base_config):
        torch.manual_seed(49)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device).eval()

        L = 32
        x = torch.randn(1, L, base_config.dim, device=device)

        with torch.no_grad():
            layer.gate_erase.weight.fill_(50.0)
            layer.gate_write.weight.fill_(-50.0)
            ofwd, sfwd, _ = layer.forward(x, chunk_size=16)

        assert not torch.isnan(ofwd).any()
        assert not torch.isinf(ofwd).any()


class TestDegenerateAndBoundaryInputs:

    def test_empty_sequence_length_zero(self, base_config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dgda = DGDALayer(base_config).to(device).eval()
        sa = MabaSparseAttention(base_config).to(device).eval()
        idx = DGIndexer(dim=base_config.dim, block_size=16, top_k=4).to(device).eval()
        model = MabaSparseForCausalLM(base_config).to(device).eval()

        x0 = torch.empty(2, 0, base_config.dim, device=device)
        inp0 = torch.empty(2, 0, dtype=torch.long, device=device)

        with torch.no_grad():
            o_d, s_d, _ = dgda(x0)
            assert o_d.shape == (2, 0, base_config.dim)
            assert s_d.shape == (2, base_config.n_heads, base_config.d_head, base_config.d_head)

            o_sa, _ = sa(x0)
            assert o_sa.shape == (2, 0, base_config.dim)

            indices, centroids = idx(x0)
            assert indices.shape == (2, 0, 0)

            out_m = model(inp0)
            assert out_m.logits.shape == (2, 0, base_config.vocab_size)

    def test_single_token_prefill_vs_step(self, base_config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device).eval()

        x1 = torch.randn(2, 1, base_config.dim, device=device)
        with torch.no_grad():
            o_fwd, s_fwd, _ = layer.forward(x1)
            o_step, s_step, _ = layer.step(x1)

        diff_o = (o_fwd - o_step).abs().max().item()
        diff_s = (s_fwd - s_step).abs().max().item()
        assert diff_o < 1e-6, f"L=1 out diff {diff_o:.8e} >= 1e-6"
        assert diff_s < 1e-6, f"L=1 state diff {diff_s:.8e} >= 1e-6"

    def test_all_zeros_and_large_scale_inputs(self, base_config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = DGDALayer(base_config).to(device).eval()
        sa = MabaSparseAttention(base_config).to(device).eval()

        x_zero = torch.zeros(2, 32, base_config.dim, device=device)
        with torch.no_grad():
            oz_d, _, _ = layer(x_zero)
            assert not torch.isnan(oz_d).any() and not torch.isinf(oz_d).any()
            oz_sa, _ = sa(x_zero)
            assert not torch.isnan(oz_sa).any() and not torch.isinf(oz_sa).any()

        x_huge = torch.randn(2, 32, base_config.dim, device=device) * 1000.0
        with torch.no_grad():
            oh_d, _, _ = layer(x_huge)
            assert not torch.isnan(oh_d).any() and not torch.isinf(oh_d).any()
            oh_sa, _ = sa(x_huge)
            assert not torch.isnan(oh_sa).any() and not torch.isinf(oh_sa).any()
