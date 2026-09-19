
import math
import pytest
import torch

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.indexer import DGIndexer, DeltaGuidedCentroidIndexer


class TestHybridPoolingNeedleProtection:

    def test_hybrid_pooling_needle_protection(self):
        torch.manual_seed(42)
        dim = 640
        d_idx = 64
        block_size = 64
        top_k = 32
        L = 4096
        B = 1

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=top_k,
            dist_lambda=0.1,
        )
        indexer.eval()

        needle_pos = 500
        expected_needle_block = needle_pos // block_size
        query_pos = 2048

        x = torch.randn(B, L, dim) * 0.05

        needle_pattern = torch.randn(dim) * 20.0
        x[0, needle_pos] = needle_pattern

        x[0, query_pos] = needle_pattern * 0.5

        with torch.no_grad():
            top_indices, centroids = indexer(x)

        num_blocks = (L + block_size - 1) // block_size
        assert centroids.shape == (B, num_blocks, d_idx)
        assert top_indices.shape == (B, L, top_k)

        query_top_blocks = top_indices[0, query_pos].tolist()

        assert (
            expected_needle_block in query_top_blocks
        ), f"Needle block {expected_needle_block} not found in top-{top_k} blocks: {query_top_blocks}"

    def test_needle_dilution_mitigation_vs_pure_mean(self):
        block_size = 64
        d_idx = 64
        k_blocks = torch.randn(1, 1, block_size, d_idx) * 0.01

        needle_idx = 37
        k_blocks[0, 0, needle_idx, 0] = 10.0

        mean_centroid = k_blocks.mean(dim=2)
        max_centroid, _ = k_blocks.max(dim=2)
        hybrid_centroid = 0.5 * (mean_centroid + max_centroid)

        assert mean_centroid[0, 0, 0].item() < 0.25
        assert max_centroid[0, 0, 0].item() >= 9.9
        assert hybrid_centroid[0, 0, 0].item() > 4.5
        assert hybrid_centroid[0, 0, 0].item() > 25.0 * mean_centroid[0, 0, 0].item()

    def test_multiple_needles_retrieval(self):
        torch.manual_seed(123)
        dim = 128
        d_idx = 32
        block_size = 64
        top_k = 16
        L = 1024
        B = 1

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=top_k,
            dist_lambda=0.05,
        )
        indexer.eval()

        needle_positions = [100, 300, 600]
        expected_blocks = [p // block_size for p in needle_positions]

        x = torch.randn(B, L, dim) * 0.02
        probe_vector = torch.randn(dim) * 15.0

        for pos in needle_positions:
            x[0, pos] = probe_vector

        q_pos = 900
        x[0, q_pos] = probe_vector

        with torch.no_grad():
            top_indices, _ = indexer(x)

        query_selected = top_indices[0, q_pos].tolist()
        for blk in expected_blocks:
            assert blk in query_selected, f"Needle block {blk} missing from retrieved blocks: {query_selected}"


class TestLogDistancePenalty:

    def test_log_distance_decay_monotonicity(self):
        dim = 64
        d_idx = 32
        block_size = 64
        num_blocks = 16
        L = num_blocks * block_size
        B = 1

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=num_blocks,
            dist_lambda=0.5,
        )
        indexer.eval()

        with torch.no_grad():
            indexer.q_idx_proj.weight.zero_()
            indexer.k_idx_proj.weight.zero_()

        x = torch.randn(B, L, dim)
        _, _, scores = indexer(x, return_scores=True)

        q_token = L - 1
        q_scores = scores[0, q_token, :num_blocks]

        for i in range(num_blocks - 1):
            assert (
                q_scores[i] < q_scores[i + 1]
            ), f"Score for block {i} ({q_scores[i]}) should be strictly lower than block {i+1} ({q_scores[i+1]})"

    def test_log_distance_exact_formula(self):
        dim = 64
        d_idx = 32
        block_size = 64
        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            dist_lambda=0.75,
        )
        indexer.eval()

        with torch.no_grad():
            indexer.q_idx_proj.weight.zero_()
            indexer.k_idx_proj.weight.zero_()

        x = torch.zeros(1, 256, dim)
        _, _, scores = indexer(x, return_scores=True)

        expected_penalty = 0.75 * math.log(3.0)
        actual_score = scores[0, 192, 1].item()
        assert math.isclose(-expected_penalty, actual_score, rel_tol=1e-5)


class TestCausalBlockMasking:

    def test_causal_block_isolation(self):
        dim = 64
        d_idx = 32
        block_size = 64
        num_blocks = 8
        L = num_blocks * block_size

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=num_blocks,
            dist_lambda=0.5,
        )
        indexer.eval()

        x = torch.randn(1, L, dim)
        top_indices, _, scores = indexer(x, return_scores=True)

        for t in [0, 31, 63, 64, 127, 250, L - 1]:
            q_block = t // block_size

            for b in range(num_blocks):
                if b > q_block:
                    assert (
                        scores[0, t, b].item() == float("-inf")
                    ), f"Future block {b} has non-inf score {scores[0, t, b]} for query at token {t} (block {q_block})"
                else:
                    assert math.isfinite(
                        scores[0, t, b].item()
                    ), f"Past/current block {b} has non-finite score {scores[0, t, b]}"

            available_past_blocks = q_block + 1
            selected = top_indices[0, t].tolist()
            for rank_idx, b in enumerate(selected[:available_past_blocks]):
                assert (
                    b <= q_block
                ), f"Query at token {t} (block {q_block}) selected future block {b} at rank {rank_idx}"


class TestDynamicTopKSelection:

    @pytest.mark.parametrize("L,expected_blocks,expected_k", [
        (32, 1, 1),
        (64, 1, 1),
        (65, 2, 2),
        (128, 2, 2),
        (512, 8, 8),
        (2048, 32, 32),
        (4096, 64, 32),
    ])
    def test_adaptive_ceiling(self, L, expected_blocks, expected_k):
        dim = 64
        d_idx = 32
        block_size = 64
        default_top_k = 32

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=default_top_k,
        )
        indexer.eval()

        x = torch.randn(2, L, dim)
        top_indices, centroids = indexer(x)

        assert centroids.shape == (2, expected_blocks, d_idx)
        assert top_indices.shape == (2, L, expected_k)

    def test_custom_top_k_override(self):
        dim = 64
        d_idx = 32
        indexer = DGIndexer(dim=dim, d_idx=d_idx, block_size=64, top_k=32)

        x = torch.randn(1, 1024, dim)
        top_indices_8, _ = indexer(x, top_k=8)
        assert top_indices_8.shape == (1, 1024, 8)

        top_indices_16, _ = indexer(x, top_k=16)
        assert top_indices_16.shape == (1, 1024, 16)


class TestIndexerAnalyticalGradientContinuity:

    def test_backward_indexer_parameters_finite_and_nonzero(self):
        torch.manual_seed(999)
        dim = 128
        d_idx = 32
        block_size = 64
        indexer = DGIndexer(dim=dim, d_idx=d_idx, block_size=block_size, top_k=8)
        indexer.train()

        B, L = 2, 256
        x = torch.randn(B, L, dim, requires_grad=True)

        top_indices, centroids, scores = indexer(x, return_scores=True)

        loss = centroids.sum() + scores[scores != float("-inf")].sum()
        loss.backward()

        assert x.grad is not None, "Input x.grad is None"
        assert not torch.isnan(x.grad).any(), "Input x.grad contains NaNs"
        assert not torch.isinf(x.grad).any(), "Input x.grad contains Infs"
        assert torch.count_nonzero(x.grad) > 0, "Input x.grad is entirely zero"

        for name, param in indexer.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"Param {name} grad is None"
            assert not torch.isnan(param.grad).any(), f"Param {name} grad has NaNs"
            assert not torch.isinf(param.grad).any(), f"Param {name} grad has Infs"
            norm = param.grad.norm().item()
            assert norm > 0.0, f"Param {name} grad norm is zero"
            assert math.isfinite(norm), f"Param {name} grad norm is not finite: {norm}"

    def test_centroid_pooling_gradient_flow(self):
        dim = 64
        d_idx = 16
        block_size = 32
        indexer = DGIndexer(dim=dim, d_idx=d_idx, block_size=block_size)
        indexer.train()

        x = torch.randn(1, 64, dim, requires_grad=True)
        _, centroids = indexer(x)
        centroids.sum().backward()

        assert indexer.k_idx_proj.weight.grad is not None
        assert indexer.k_idx_proj.weight.grad.norm().item() > 0.0


class TestIndexerDtypesAndAliases:

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_indexer_numerical_stability(self, dtype):
        indexer = DGIndexer(dim=64, d_idx=32, block_size=64, top_k=4).to(dtype)
        x = torch.randn(1, 128, 64, dtype=dtype)

        top_indices, centroids = indexer(x)
        assert not torch.isnan(centroids).any()
        assert not torch.isinf(centroids).any()
        assert top_indices.dtype == torch.int64

    def test_alias_parity(self):
        assert DeltaGuidedCentroidIndexer is DGIndexer
        config = MabaSparseConfig(dim=128, d_idx=32, block_size=32, top_k=8)
        indexer = DeltaGuidedCentroidIndexer(config=config)
        assert indexer.dim == 128
        assert indexer.d_idx == 32
