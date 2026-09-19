# Maba v1.5

Reference PyTorch implementation of the **Maba v1.5** hybrid architecture.

Maba combines linear recurrence (DGDA) with sparse global attention (MABA-SA) in a 3:1 cyclic stack to eliminate quadratic memory growth while preserving strict $O(1)$ per-token generation latency.

---

## Architecture Overview

- **Canonical Configuration**: [config.json](config.json) (101,282,319 parameters).
- **Macro-Topology (20 layers, 3:1 ratio)**:
  - 15 layers: [DGDA (Decoupled Gated Delta Attention)](maba_sparse/layers/dgda.py) linear recurrence.
  - 5 layers: [MABA-SA (Sparse Attention)](maba_sparse/layers/sparse_attention.py) with MLA latent compression ($d_c=128$).
- **Embeddings**: Factorized embeddings (vocab 32,768 -> 128 -> 640) defined in [maba_sparse/model.py](maba_sparse/model.py).
- **Recurrence Engine**: Chunked parallel prefill ($C=16$) with adaptive order-3 Neumann series.
- **Sparse Routing ([DG-Indexer](maba_sparse/layers/indexer.py))**:
  - Anti-dilution hybrid centroid pooling ($0.5 \times \text{mean} + 0.5 \times \text{max}$).
  - Top-32 block selection (2,048 tokens active context) with $\lambda \log(1+\Delta)$ distance penalty.
  - 3-stream superposition: local window (128) + sparse top-k (2,048) + HCA summary (64:1).
- **Positional Encoding**: NoPE (No Positional Embeddings in attention; temporal order maintained through recurrent decay $\alpha_t$).
- **Multi-Backend Acceleration**: [maba_sparse/kernels/](maba_sparse/kernels/) includes fused Triton GPU kernels, CPU OpenMP parallel kernels, and PyTorch reference fallbacks.

---

## Quick Start

### Installation

```bash
git clone https://github.com/ivan-dev35/123123.git
cd 123123/codex/maba-v1.5-exp-architecture
pip install -e .
```

### Autoregressive Generation

```python
import torch
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config

config = get_101m_config()
model = MabaSparseForCausalLM(config).cuda().eval()

input_ids = torch.tensor([[101, 2045, 312]], device="cuda")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=32, temperature=0.7)
print(output)
```

### Distributed Training

```bash
# Multi-GPU training via DistributedDataParallel (DDP)
torchrun --nproc_per_node=2 train.py \
    --model maba_sparse \
    --dataset synthetic \
    --steps 100 \
    --batch_size 4 \
    --fp16
```

---

## Benchmarks & Evaluation

All experimental benchmarks and verification suites are consolidated in [benchmark.py](benchmark.py). Detailed hardware metrics on Tesla T4 GPUs are recorded in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) and [benchmark_results.json](benchmark_results.json).

### Running Benchmark Suites

```bash
# 1. Full end-to-end model & isolated attention scaling (Maba vs Dense Transformer)
python benchmark.py --mode model --contexts 128,256,512,1024,2048,4096

# 2. Strict O(1) decode latency scaling (35-37 ms/tok across 128 to 16,384 tokens)
python benchmark.py --mode decode

# 3. KV-cache footprint comparison (40x reduction vs dense attention)
python benchmark.py --mode memory

# 4. 1,000,000 token single-needle fact extraction test
python benchmark.py --mode needle

# 5. 50 Hard Negatives ('semantic mines') & Multi-Hop reasoning across 640k tokens
python benchmark.py --mode multihop

# 6. Hardware kernel throughput (375k+ tok/s on Tesla T4)
python benchmark.py --mode triton

# 7. Run all suites sequentially
python benchmark.py --mode all
```

### Tesla T4 Hardware Summary

| Context Length | Maba Prefill (ms) | Dense Prefill (ms) | Maba VRAM (MB) | Dense VRAM (MB) | Maba Decode (ms/tok) | Dense Decode (ms/tok) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 49.71 | 16.56 | 920.4 | 863.4 | 36.77 | 15.94 |
| 512 | 134.02 | 38.52 | 1342.6 | 964.2 | 37.02 | 15.14 |
| 2,048 | 1474.72 | 172.97 | 2859.2 | 1350.5 | 37.28 | 17.17 |
| 4,096 | 3230.19 | 427.90 | 2946.7 | 1818.2 | 35.30 | 15.79 |

Key takeaways:
- **Decode Latency**: Invariant at 35–37 ms/token up to 1,000,000 tokens (strict $O(1)$).
- **KV Cache Footprint**: 1.20 GB at 1M tokens (vs 48.8 GB for Dense Transformer — **39.6x savings**).
- **Needle Retrieval**: Single fact extracted at Token #742,189 with Rank #1 out of 15,625 blocks and 100% fine-grained attention focus.

---

## Unit Test Suite

Run the full automated test suite (649 tests):

```bash
pytest -q
```

All 23 test modules in [tests/](tests/) test causal masking, numerical stability, autograd correctness, memory invariance, and multi-backend parity.

---

## License & Attribution

Maba is released under the **MABA Open Architecture License (MOAL-1.0)**.

- **Author**: Andrew Thompson (`AndrewThompson1233`)
- **Commercial & Research Use**: Fully permitted without royalty fees.
- **Attribution**: Any derivative architecture, implementation, checkpoint, or paper must prominently state:
  > `Created based on Maba Architecture by Andrew Thompson`
- **Naming Protection**: The name **Maba** and its foundational mechanisms (**DGDA**, **MABA-SA**, **DG-Indexer**, **HCA**) may not be renamed, rebranded, or claimed under different names when copying or adapting this architecture.

See [LICENSE](LICENSE) for the full license text.
