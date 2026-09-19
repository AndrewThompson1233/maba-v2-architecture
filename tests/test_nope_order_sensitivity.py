
import pytest
import torch
import torch.nn.functional as F

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.layers.sparse_attention import MabaSparseAttention
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config


class TestNoPEOrderSensitivity:

    @pytest.fixture
    def small_model(self):
        torch.manual_seed(42)
        cfg = MabaSparseConfig(
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
        )
        model = MabaSparseForCausalLM(cfg)
        model.eval()
        return model

    def test_token_permutation_order_sensitivity(self, small_model):
        s1 = torch.tensor([[100, 200, 300, 400]])
        s2 = torch.tensor([[200, 100, 300, 400]])

        with torch.no_grad():
            out1 = small_model(s1)
            out2 = small_model(s2)

        logits1 = out1.logits
        logits2 = out2.logits

        diff_norm = torch.norm(logits1[:, -1, :] - logits2[:, -1, :], p=2).item()
        assert (
            diff_norm > 1e-2
        ), f"Expected order sensitivity ||L(S1) - L(S2)||_2 > 0.01, got {diff_norm:.6f}"

    def test_semantic_sentence_permutation(self, small_model):
        cat_chased_mouse = torch.tensor([[10, 25, 50, 10, 75]])
        mouse_chased_cat = torch.tensor([[10, 75, 50, 10, 25]])

        with torch.no_grad():
            out_cat = small_model(cat_chased_mouse)
            out_mouse = small_model(mouse_chased_cat)

        diff = torch.norm(out_cat.logits - out_mouse.logits, p=2).item()
        assert (
            diff > 0.1
        ), f"Sentences must produce distinct representations, got diff={diff:.6f}"

    def test_dgda_decay_is_order_mechanism(self):
        torch.manual_seed(42)
        cfg = MabaSparseConfig(dim=64, n_heads=2, d_head=32)
        layer = DGDALayer(cfg)
        layer.eval()

        x1 = torch.randn(1, 8, 64)
        x2 = torch.flip(x1, dims=[1])

        with torch.no_grad():
            out1, state1, _ = layer(x1)
            out2, state2, _ = layer(x2)

        state_diff = torch.norm(state1 - state2).item()
        out_diff = torch.norm(out1 - torch.flip(out2, dims=[1])).item()

        assert state_diff > 1e-2, f"DGDA recurrent state must depend on token order: {state_diff}"
        assert out_diff > 1e-3, f"DGDA outputs must depend on token order: {out_diff}"

    def test_attention_layers_have_no_positional_embeddings(self):
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        for i, layer in enumerate(model.layers):
            if layer.is_attention:
                attn = layer.mixer
                for name, param in attn.named_parameters():
                    param_name_lower = name.lower()
                    assert "pos" not in param_name_lower, f"Found pos in {name}"
                    assert "rope" not in param_name_lower, f"Found rope in {name}"
                    assert "rotary" not in param_name_lower, f"Found rotary in {name}"
                    assert "alibi" not in param_name_lower, f"Found alibi in {name}"
