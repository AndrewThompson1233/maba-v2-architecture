
import pytest
import torch

from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.layers.sparse_attention import MabaSparseAttention
from maba_sparse.model import (
    MabaSparseForCausalLM,
    get_101m_config,
)


class TestModelParameterCount:

    def test_maba_sparse_101m_parameter_count(self):
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)
        unique_params = sum(p.numel() for p in set(model.parameters()))

        assert (
            95_000_000 <= unique_params <= 105_000_000
        ), f"Expected Maba-Sparse params in [95M, 105M], got {unique_params:,} ({unique_params/1e6:.2f}M)"

    def test_dense_transformer_101m_parameter_count(self):
        dense_model = DenseTransformerForCausalLM()
        unique_params = sum(p.numel() for p in set(dense_model.parameters()))

        assert (
            95_000_000 <= unique_params <= 105_000_000
        ), f"Expected Dense Transformer params in [95M, 105M], got {unique_params:,} ({unique_params/1e6:.2f}M)"

    def test_parameter_budget_alignment(self):
        maba_model = MabaSparseForCausalLM(get_101m_config())
        dense_model = DenseTransformerForCausalLM()

        maba_params = sum(p.numel() for p in set(maba_model.parameters()))
        dense_params = sum(p.numel() for p in set(dense_model.parameters()))

        relative_diff = abs(maba_params - dense_params) / dense_params
        assert (
            relative_diff < 0.05
        ), f"Expected <5% param difference, got {relative_diff*100:.2f}% (Maba: {maba_params:,}, Dense: {dense_params:,})"


class TestModelArchitectureTopology:

    def test_cyclic_3_to_1_macro_stack(self):
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        assert len(model.layers) == 20
        dgda_count = 0
        maba_sa_count = 0

        for i, layer in enumerate(model.layers):
            if (i + 1) % 4 == 0:
                assert isinstance(
                    layer.mixer, MabaSparseAttention
                ), f"Layer {i} should be MabaSparseAttention, got {type(layer.mixer)}"
                maba_sa_count += 1
            else:
                assert isinstance(
                    layer.mixer, DGDALayer
                ), f"Layer {i} should be DGDALayer, got {type(layer.mixer)}"
                dgda_count += 1

        assert dgda_count == 15, f"Expected 15 DGDA layers, got {dgda_count}"
        assert maba_sa_count == 5, f"Expected 5 MABA-SA layers, got {maba_sa_count}"

    def test_factorized_embedding_dimensions(self):
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        assert model.embeddings.vocab_size == 32768
        assert model.embeddings.d_emb == 128
        assert model.embeddings.dim == 640
        assert model.embeddings.in_emb.weight.shape == (32768, 128)
        assert model.embeddings.proj.weight.shape == (640, 128)

    def test_weight_tying(self):
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        assert (
            model.lm_head.weight is model.embeddings.in_emb.weight
        ), "lm_head.weight must be tied to embeddings.in_emb.weight"

    def test_residual_gate_initial_bias(self):
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        for i, layer in enumerate(model.layers):
            assert torch.allclose(
                layer.res_gate1, torch.full_like(layer.res_gate1, 2.0)
            ), f"Layer {i} res_gate1 bias != 2.0"
            assert torch.allclose(
                layer.res_gate2, torch.full_like(layer.res_gate2, 2.0)
            ), f"Layer {i} res_gate2 bias != 2.0"




class TestModelForwardBackward:

    @pytest.fixture
    def small_model(self):
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
        return MabaSparseForCausalLM(cfg)

    def test_forward_output_shape(self, small_model):
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        out = small_model(x)

        assert out.logits.shape == (B, L, 1000)
        assert out.loss is None
        logits, loss = out
        assert logits.shape == (B, L, 1000)
        assert loss is None

    def test_forward_with_targets(self, small_model):
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))
        out = small_model(x, targets=targets)

        assert out.loss is not None
        assert torch.isfinite(out.loss)
        assert out.loss.item() > 0.0

    def test_backward_gradient_continuity(self, small_model):
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))

        out = small_model(x, targets=targets)
        out.loss.backward()

        for name, p in small_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"

    def test_mtp_auxiliary_loss(self, small_model):
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))

        out = small_model(x, targets=targets)
        assert out.mtp_logits is not None
        assert out.mtp_logits.shape == (B, L - 1, 1000)

        out.loss.backward()
        assert small_model.mtp_head.proj.weight.grad is not None
        assert not torch.isnan(small_model.mtp_head.proj.weight.grad).any()




class TestDenseTransformerBaseline:

    @pytest.fixture
    def small_dense_model(self):
        return DenseTransformerForCausalLM(
            vocab_size=1000,
            d_emb=32,
            dim=64,
            n_layers=4,
            n_heads=2,
            d_head=32,
            intermediate_size=128,
        )

    def test_dense_forward_backward(self, small_dense_model):
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))

        out = small_dense_model(x, targets=targets)
        assert out.logits.shape == (B, L, 1000)
        assert out.loss is not None
        assert torch.isfinite(out.loss)

        out.loss.backward()
        for name, p in small_dense_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in dense {name}"

    def test_autoregressive_generation(self, small_dense_model):
        x = torch.randint(0, 1000, (1, 4))
        gen = small_dense_model.generate(x, max_new_tokens=6, temperature=0.0)
        assert gen.shape == (1, 10)
        assert torch.equal(gen[:, :4], x)
