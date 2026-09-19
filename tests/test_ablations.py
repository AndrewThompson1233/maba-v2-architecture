
import pytest
import torch

from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.model import MabaSparseForCausalLM


class TestArchitecturalAblations:

    @pytest.fixture
    def small_config(self):
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
        )

    def test_ablation_a_dense_baseline(self):
        model = DenseTransformerForCausalLM(
            vocab_size=1000,
            d_emb=32,
            dim=64,
            n_layers=4,
            n_heads=2,
            d_head=32,
            intermediate_size=128,
        )
        assert len(model.layers) == 4
        x = torch.randint(0, 1000, (2, 16))
        targets = torch.randint(0, 1000, (2, 16))

        out = model(x, targets=targets)
        assert out.logits.shape == (2, 16, 1000)
        assert out.loss is not None
        assert torch.isfinite(out.loss)

    def test_ablation_b_pure_dgda(self, small_config):
        model = MabaSparseForCausalLM(small_config, ablation_mode="pure_dgda")

        for i, layer in enumerate(model.layers):
            assert isinstance(
                layer.mixer, DGDALayer
            ), f"Layer {i} should be DGDALayer in pure_dgda, got {type(layer.mixer)}"

        x = torch.randint(0, 1000, (2, 16))
        targets = torch.randint(0, 1000, (2, 16))
        out = model(x, targets=targets)

        assert out.logits.shape == (2, 16, 1000)
        assert torch.isfinite(out.loss)

    def test_ablation_c_dgda_plus_sparse(self, small_config):
        model = MabaSparseForCausalLM(small_config, ablation_mode="no_hca")
        assert model.ablation_mode == "no_hca"

        x = torch.randint(0, 1000, (2, 16))
        targets = torch.randint(0, 1000, (2, 16))
        out = model(x, targets=targets)

        assert out.logits.shape == (2, 16, 1000)
        assert torch.isfinite(out.loss)

    def test_ablation_d_full_maba_sparse(self, small_config):
        model = MabaSparseForCausalLM(small_config, ablation_mode="full")
        assert model.ablation_mode == "full"

        x = torch.randint(0, 1000, (2, 16))
        targets = torch.randint(0, 1000, (2, 16))
        out = model(x, targets=targets)

        assert out.logits.shape == (2, 16, 1000)
        assert torch.isfinite(out.loss)
        assert out.mtp_logits is not None

    def test_cross_ablation_divergence(self, small_config):
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (1, 16))

        dense_model = DenseTransformerForCausalLM(
            vocab_size=1000, d_emb=32, dim=64, n_layers=4, n_heads=2, d_head=32, intermediate_size=128
        )
        dense_model.eval()

        pure_dgda = MabaSparseForCausalLM(small_config, ablation_mode="pure_dgda")
        pure_dgda.eval()

        dgda_sparse = MabaSparseForCausalLM(small_config, ablation_mode="no_hca")
        dgda_sparse.eval()

        full_maba = MabaSparseForCausalLM(small_config, ablation_mode="full")
        full_maba.eval()

        with torch.no_grad():
            out_dense = dense_model(x).logits
            out_pure = pure_dgda(x).logits
            out_full = full_maba(x).logits

        assert not torch.allclose(out_dense, out_pure, atol=1e-3)
        assert not torch.allclose(out_pure, out_full, atol=1e-3)
