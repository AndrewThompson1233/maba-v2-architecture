
import pytest
import torch

from maba_sparse.kernels.common import (
    ref_compute_centroids,
    ref_index_topk,
    ref_stream_superposition,
)
from maba_sparse.kernels.cpu_indexer import (
    cpu_compute_centroids,
    cpu_index_topk,
    cpu_stream_superposition,
)
from maba_sparse.kernels.dispatcher import (
    _KERNEL_REGISTRY,
    clear_fallback_warnings,
    is_cuda_sm75_available,
    is_triton_available,
)
from maba_sparse.kernels.triton_indexer import (
    triton_compute_centroids,
    triton_index_topk,
    triton_stream_superposition,
)
from maba_sparse.kernels.xla_indexer import (
    xla_compute_centroids,
    xla_index_topk,
    xla_stream_superposition,
)

CUDA_AVAILABLE = torch.cuda.is_available() and is_cuda_sm75_available()
TRITON_ACTIVE = CUDA_AVAILABLE and is_triton_available()


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.delenv("MABA_BACKEND", raising=False)
    monkeypatch.delenv("MABA_STRICT_BACKEND", raising=False)
    if TRITON_ACTIVE:
        _KERNEL_REGISTRY.setdefault("triton", {})["compute_centroids"] = triton_compute_centroids
        _KERNEL_REGISTRY.setdefault("triton", {})["index_topk"] = triton_index_topk
        _KERNEL_REGISTRY.setdefault("triton", {})["stream_superposition"] = triton_stream_superposition
    _KERNEL_REGISTRY.setdefault("cpu", {})["compute_centroids"] = cpu_compute_centroids
    _KERNEL_REGISTRY.setdefault("cpu", {})["index_topk"] = cpu_index_topk
    _KERNEL_REGISTRY.setdefault("cpu", {})["stream_superposition"] = cpu_stream_superposition
    _KERNEL_REGISTRY.setdefault("xla", {})["compute_centroids"] = xla_compute_centroids
    _KERNEL_REGISTRY.setdefault("xla", {})["index_topk"] = xla_index_topk
    _KERNEL_REGISTRY.setdefault("xla", {})["stream_superposition"] = xla_stream_superposition
    clear_fallback_warnings()


class TestNumericalSingularities:

    @pytest.mark.parametrize(
        "logits_pattern",
        [
            [-10000.0, 10000.0, 0.0],
            [10000.0, -10000.0, -10000.0],
            [-10000.0, -10000.0, -10000.0],
            [5000.0, 5000.0, 5000.0],
        ],
    )
    def test_extreme_logits_superposition(self, logits_pattern):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 2, 4, 128, 64
        ol = torch.randn(B, H, L, D, device=device)
        os_stream = torch.randn(B, H, L, D, device=device)
        oh = torch.randn(B, H, L, D, device=device)

        logits = torch.tensor([[logits_pattern]], device=device).expand(B, L, 3).clone()

        out_ref = ref_stream_superposition(ol, os_stream, oh, gate_logits=logits)
        out_cpu = cpu_stream_superposition(ol, os_stream, oh, gate_logits=logits)

        assert torch.isfinite(out_ref).all(), "Reference output contains non-finite values"
        assert torch.isfinite(out_cpu).all(), "CPU output contains non-finite values"
        assert torch.allclose(out_cpu, out_ref, atol=1e-5), "CPU vs Ref mismatch under extreme logits"

        if TRITON_ACTIVE:
            out_tri = triton_stream_superposition(ol, os_stream, oh, gate_logits=logits)
            assert torch.isfinite(out_tri).all(), "Triton output contains non-finite values"
            assert torch.allclose(out_tri, out_ref, atol=1e-5), "Triton vs Ref mismatch under extreme logits"

    def test_extreme_logits_superposition_autograd(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, H, L, D = 2, 2, 32, 64
        ol = torch.randn(B, H, L, D, device=device, requires_grad=True)
        os_stream = torch.randn(B, H, L, D, device=device, requires_grad=True)
        oh = torch.randn(B, H, L, D, device=device, requires_grad=True)
        logits = torch.tensor([[[-10000.0, 10000.0, 0.0]]], device=device).expand(B, L, 3).clone().requires_grad_(True)

        if TRITON_ACTIVE:
            out = triton_stream_superposition(ol, os_stream, oh, gate_logits=logits)
        else:
            out = cpu_stream_superposition(ol, os_stream, oh, gate_logits=logits)

        loss = out.sum()
        loss.backward()

        assert torch.isfinite(ol.grad).all(), "ol.grad contains NaNs/Infs under extreme logits"
        assert torch.isfinite(os_stream.grad).all(), "os.grad contains NaNs/Infs under extreme logits"
        assert torch.isfinite(oh.grad).all(), "oh.grad contains NaNs/Infs under extreme logits"
        assert torch.isfinite(logits.grad).all(), "logits.grad contains NaNs/Infs under extreme logits"

    def test_uniform_zero_keys(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 256, 64
        k = torch.zeros(B, L, d_idx, device=device)
        q = torch.randn(B, L, d_idx, device=device)

        c_ref = ref_compute_centroids(k, block_size=64)
        c_cpu = cpu_compute_centroids(k, block_size=64)
        assert (c_ref == 0.0).all()
        assert (c_cpu == 0.0).all()

        idx_ref = ref_index_topk(q, c_ref, lambda_dist=0.5, top_k=4, block_size=64)
        idx_cpu = cpu_index_topk(q, c_cpu, lambda_dist=0.5, top_k=4, block_size=64)
        assert (idx_cpu == idx_ref).all(), "CPU vs Ref index mismatch on uniform zero keys"

        if TRITON_ACTIVE:
            c_tri = triton_compute_centroids(k, block_size=64)
            assert (c_tri == 0.0).all()
            idx_tri = triton_index_topk(q, c_tri, lambda_dist=0.5, top_k=4, block_size=64)
            assert (idx_tri == idx_ref).all(), "Triton vs Ref index mismatch on uniform zero keys"

    def test_all_negative_partial_block_behavior(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 1, 65, 64
        k = torch.full((B, L, d_idx), -5.0, device=device)

        c_ref = ref_compute_centroids(k, block_size=64)
        c_cpu = cpu_compute_centroids(k, block_size=64)

        assert torch.allclose(c_cpu, c_ref, atol=1e-5), "CPU vs Ref centroid mismatch on all-negative keys"

        assert torch.allclose(c_ref[0, 0], torch.full((d_idx,), -5.0, device=device), atol=1e-5)

        expected_c1 = 0.5 * (-5.0 / 64.0 + 0.0)
        assert torch.allclose(c_ref[0, 1], torch.full((d_idx,), expected_c1, device=device), atol=1e-5)

        if TRITON_ACTIVE:
            c_tri = triton_compute_centroids(k, block_size=64)
            assert torch.allclose(c_tri, c_ref, atol=1e-5), "Triton vs Ref mismatch on all-negative keys"
            assert torch.allclose(c_tri[0, 1], torch.full((d_idx,), expected_c1, device=device), atol=1e-5)

    def test_lambda_zero_pure_content_retrieval(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 256, 64
        q = torch.randn(B, L, d_idx, device=device)
        c = torch.randn(B, 4, d_idx, device=device)

        idx_ref, sc_ref = ref_index_topk(q, c, lambda_dist=0.0, top_k=4, block_size=64, return_scores=True)
        idx_cpu, sc_cpu = cpu_index_topk(q, c, lambda_dist=0.0, top_k=4, block_size=64, return_scores=True)

        assert torch.allclose(sc_cpu, sc_ref, atol=1e-5), "Score mismatch for lambda=0"
        assert (idx_cpu == idx_ref).all(), "Index mismatch for lambda=0 on CPU"

        if TRITON_ACTIVE:
            idx_tri, sc_tri = triton_index_topk(q, c, lambda_dist=0.0, top_k=4, block_size=64, return_scores=True)
            assert torch.allclose(sc_tri, sc_ref, atol=1e-4), "Score mismatch for lambda=0 on Triton"
            assert (idx_tri == idx_ref).all(), "Index mismatch for lambda=0 on Triton"

    def test_lambda_huge_extreme_recency_prior(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 2, 512, 64
        nb = 8
        q = torch.randn(B, L, d_idx, device=device)
        c = torch.randn(B, nb, d_idx, device=device)

        idx_ref = ref_index_topk(q, c, lambda_dist=1000.0, top_k=4, block_size=64)
        idx_cpu = cpu_index_topk(q, c, lambda_dist=1000.0, top_k=4, block_size=64)
        assert (idx_cpu == idx_ref).all()

        assert idx_cpu[0, 500].tolist() == [7, 6, 5, 4]

        if TRITON_ACTIVE:
            idx_tri = triton_index_topk(q, c, lambda_dist=1000.0, top_k=4, block_size=64)
            assert (idx_tri == idx_ref).all()
            assert idx_tri[0, 500].tolist() == [7, 6, 5, 4]


class TestOutlierNeedleInAHaystack:

    @pytest.mark.parametrize("L", [1024, 2048, 4096])
    def test_needle_detection_across_long_contexts(self, L):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        nb = L // 64
        B, d_idx = 1, 64
        torch.manual_seed(42)

        k = torch.randn(B, L, d_idx, device=device)

        needle_block = 5
        needle_pos = needle_block * 64 + 10
        k[:, needle_pos, 0] = 25.0

        q = torch.randn(B, L, d_idx, device=device) * 0.1
        q[:, -1, 0] = 5.0

        if TRITON_ACTIVE:
            c_hybrid = triton_compute_centroids(k, block_size=64)
            idx_hybrid = triton_index_topk(q, c_hybrid, lambda_dist=0.5, top_k=32, block_size=64)
        else:
            c_hybrid = cpu_compute_centroids(k, block_size=64)
            idx_hybrid = cpu_index_topk(q, c_hybrid, lambda_dist=0.5, top_k=32, block_size=64)

        kb = k.view(B, nb, 64, d_idx)
        c_mean = kb.mean(dim=2)
        idx_mean = ref_index_topk(q, c_mean, lambda_dist=0.5, top_k=32, block_size=64)

        last_query_hybrid = idx_hybrid[0, -1].tolist()
        last_query_mean = idx_mean[0, -1].tolist()

        assert needle_block in last_query_hybrid, f"Hybrid pooling failed to select needle block {needle_block} at L={L}"
        assert last_query_hybrid[0] == needle_block, (
            f"Hybrid pooling did not rank needle block {needle_block} at rank 0: got {last_query_hybrid[:4]}"
        )

        mean_found = needle_block in last_query_mean
        mean_rank = last_query_mean.index(needle_block) if mean_found else -1
        if L == 4096:
            assert mean_rank != 0, "Mean pooling unexpectedly ranked needle at rank 0"

    def test_hybrid_vs_mean_needle_signal_amplification(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L, d_idx = 1, 64, 64
        torch.manual_seed(999)
        k = torch.randn(B, L, d_idx, device=device) * 0.1
        k[0, 17, 0] = 20.0

        if TRITON_ACTIVE:
            c_hybrid = triton_compute_centroids(k, block_size=64)
        else:
            c_hybrid = cpu_compute_centroids(k, block_size=64)
        c_mean = k.mean(dim=1, keepdim=True)

        hybrid_signal = c_hybrid[0, 0, 0].item()
        mean_signal = c_mean[0, 0, 0].item()

        ratio = hybrid_signal / max(mean_signal, 1e-6)
        assert ratio >= 15.0, f"Expected hybrid signal amplification >= 15x, got {ratio:.2f}x"


MISALIGNED_LENGTHS = [1, 2, 63, 65, 127, 129, 255, 257, 511, 513]
ODD_BATCHES = [3, 7]


class TestBoundaryAndSequenceMisalignment:

    @pytest.mark.parametrize("B", ODD_BATCHES)
    @pytest.mark.parametrize("L", MISALIGNED_LENGTHS)
    def test_misaligned_sequence_parity_cpu_and_xla(self, B, L):
        d_idx = 64
        torch.manual_seed(B * 1000 + L)
        k = torch.randn(B, L, d_idx)
        q = torch.randn(B, L, d_idx)

        c_ref = ref_compute_centroids(k, block_size=64)
        c_cpu = cpu_compute_centroids(k, block_size=64)
        c_xla = xla_compute_centroids(k, block_size=64)

        assert torch.allclose(c_cpu, c_ref, atol=1e-5), f"CPU centroid mismatch at B={B}, L={L}"
        assert torch.allclose(c_xla, c_ref, atol=1e-5), f"XLA centroid mismatch at B={B}, L={L}"

        nb = c_ref.shape[1]
        top_k = min(32, nb)
        idx_ref = ref_index_topk(q, c_ref, lambda_dist=0.5, top_k=top_k, block_size=64)
        idx_cpu = cpu_index_topk(q, c_cpu, lambda_dist=0.5, top_k=top_k, block_size=64)
        idx_xla = xla_index_topk(q, c_xla, lambda_dist=0.5, top_k=top_k, block_size=64)

        assert (idx_cpu == idx_ref).all(), f"CPU index mismatch at B={B}, L={L}"
        assert (idx_xla == idx_ref).all(), f"XLA index mismatch at B={B}, L={L}"

        H = 2
        ol = torch.randn(B, H, L, d_idx)
        os_stream = torch.randn(B, H, L, d_idx)
        oh = torch.randn(B, H, L, d_idx)
        logits = torch.randn(B, L, 3)

        sup_ref = ref_stream_superposition(ol, os_stream, oh, gate_logits=logits)
        sup_cpu = cpu_stream_superposition(ol, os_stream, oh, gate_logits=logits)
        sup_xla = xla_stream_superposition(ol, os_stream, oh, gate_logits=logits)

        assert torch.allclose(sup_cpu, sup_ref, atol=1e-5), f"CPU superposition mismatch at B={B}, L={L}"
        assert torch.allclose(sup_xla, sup_ref, atol=1e-5), f"XLA superposition mismatch at B={B}, L={L}"

    @pytest.mark.skipif(not TRITON_ACTIVE, reason="Requires CUDA GPU with Triton")
    @pytest.mark.parametrize("B", ODD_BATCHES)
    @pytest.mark.parametrize("L", MISALIGNED_LENGTHS)
    def test_misaligned_sequence_parity_triton(self, B, L):
        d_idx = 64
        torch.manual_seed(B * 1000 + L)
        k = torch.randn(B, L, d_idx, device="cuda")
        q = torch.randn(B, L, d_idx, device="cuda")

        c_ref = ref_compute_centroids(k, block_size=64)
        c_tri = triton_compute_centroids(k, block_size=64)
        assert torch.allclose(c_tri, c_ref, atol=1e-4), f"Triton centroid mismatch at B={B}, L={L}"

        nb = c_ref.shape[1]
        top_k = min(32, nb)
        idx_ref = ref_index_topk(q, c_ref, lambda_dist=0.5, top_k=top_k, block_size=64)
        idx_tri = triton_index_topk(q, c_tri, lambda_dist=0.5, top_k=top_k, block_size=64)
        assert (idx_tri == idx_ref).all(), f"Triton index mismatch at B={B}, L={L}"

        H = 2
        ol = torch.randn(B, H, L, d_idx, device="cuda")
        os_stream = torch.randn(B, H, L, d_idx, device="cuda")
        oh = torch.randn(B, H, L, d_idx, device="cuda")
        logits = torch.randn(B, L, 3, device="cuda")

        sup_ref = ref_stream_superposition(ol, os_stream, oh, gate_logits=logits)
        sup_tri = triton_stream_superposition(ol, os_stream, oh, gate_logits=logits)
        assert torch.allclose(sup_tri, sup_ref, atol=1e-4), f"Triton superposition mismatch at B={B}, L={L}"


class TestDiscreteTopKCausality:

    def test_strict_causality_for_mature_tokens(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        L = 2048
        top_k = 16
        B, d_idx = 2, 64
        q = torch.randn(B, L, d_idx, device=device)
        c = torch.randn(B, L // 64, d_idx, device=device)

        if TRITON_ACTIVE:
            idx = triton_index_topk(q, c, lambda_dist=0.5, top_k=top_k, block_size=64)
        else:
            idx = cpu_index_topk(q, c, lambda_dist=0.5, top_k=top_k, block_size=64)

        t_idx = torch.arange(L, device=device).unsqueeze(0).unsqueeze(-1) // 64
        q_block = torch.arange(L, device=device) // 64

        mature_mask = q_block >= (top_k - 1)
        idx_mature = idx[:, mature_mask, :]
        t_mature = t_idx[:, mature_mask, :]

        violations = (idx_mature > t_mature).any().item()
        assert not violations, "Causal violation found for mature tokens: future blocks were selected!"

    def test_early_tokens_score_invariance(self):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        L = 256
        top_k = 4
        B, d_idx = 1, 64
        q = torch.randn(B, L, d_idx, device=device)
        c = torch.randn(B, 4, d_idx, device=device)

        if TRITON_ACTIVE:
            idx, scores = triton_index_topk(q, c, lambda_dist=0.5, top_k=top_k, block_size=64, return_scores=True)
        else:
            idx, scores = cpu_index_topk(q, c, lambda_dist=0.5, top_k=top_k, block_size=64, return_scores=True)

        assert scores[0, 0, 0] > -1e8, "Block 0 score should be finite"
        assert scores[0, 0, 1] == float("-inf"), "Future block 1 must be -inf"
        assert scores[0, 0, 2] == float("-inf"), "Future block 2 must be -inf"
        assert scores[0, 0, 3] == float("-inf"), "Future block 3 must be -inf"


class TestFeatureDimensionBoundaries:

    @pytest.mark.parametrize("d_idx", [32, 64, 128])
    def test_native_supported_dimensions(self, d_idx):
        device = "cuda" if CUDA_AVAILABLE else "cpu"
        B, L = 2, 128
        k = torch.randn(B, L, d_idx, device=device)
        q = torch.randn(B, L, d_idx, device=device)

        c_ref = ref_compute_centroids(k, block_size=64)
        c_cpu = cpu_compute_centroids(k, block_size=64)
        assert torch.allclose(c_cpu, c_ref, atol=1e-5)

        idx_ref = ref_index_topk(q, c_ref, lambda_dist=0.5, top_k=2, block_size=64)
        idx_cpu = cpu_index_topk(q, c_cpu, lambda_dist=0.5, top_k=2, block_size=64)
        assert (idx_cpu == idx_ref).all()

        if TRITON_ACTIVE:
            c_tri = triton_compute_centroids(k, block_size=64)
            assert torch.allclose(c_tri, c_ref, atol=1e-4)
            idx_tri = triton_index_topk(q, c_tri, lambda_dist=0.5, top_k=2, block_size=64)
            assert (idx_tri == idx_ref).all()

    def test_d_idx_exceeding_128_documented_limitation(self):
        d_idx = 256
        B, L = 1, 64
        k = torch.randn(B, L, d_idx)
        c_ref = ref_compute_centroids(k, block_size=64)
        c_cpu = cpu_compute_centroids(k, block_size=64)
        assert torch.allclose(c_cpu, c_ref, atol=1e-5), "CPU correctly handles d_idx=256"

        if TRITON_ACTIVE:
            k_cuda = k.cuda()
            c_tri = triton_compute_centroids(k_cuda, block_size=64)
            diff_upper = (c_tri[0, 0, 128:] - c_ref[0, 0, 128:].cuda()).abs().max().item()
            print(f"Adversarial finding: Triton d_idx=256 upper dimension diff = {diff_upper:.4f}")
