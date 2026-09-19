<div align="center">
  <img src="assets/logo.svg" width="128" alt="Maba v2 Logo"/>
  <h1>Maba v2 Architecture</h1>
  <p><b>Linear Recurrence &amp; Sparse Attention Hybrid Architecture</b></p>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MOAL--1.0-blue.svg" alt="License"/></a>
    <a href="config.json"><img src="https://img.shields.io/badge/Parameters-101.3M-emerald.svg" alt="Parameters"/></a>
    <a href="BENCHMARK_REPORT.md"><img src="https://img.shields.io/badge/Context-1%2C000%2C000+-cyan.svg" alt="Context"/></a>
    <img src="https://img.shields.io/badge/Decode-O(1)%20Flat-purple.svg" alt="Decode O(1)"/>
    <img src="https://img.shields.io/badge/Tests-649%20Passed-green.svg" alt="Tests"/>
  </p>
</div>

---

## Overview

**Maba v2 Architecture** (`maba-v2-architecture`) is a reference PyTorch implementation of a 3:1 hybrid architecture uniting linear recurrence (**DGDA**) and sparse global attention (**MABA-SA**).

Traditional dense transformers suffer from $O(L^2)$ prefill memory and $O(L)$ linear decode slowdown. Pure linear recurrent models struggle with associative recall across distant context. Maba solves this dilemma by routing 75% of compute through constant-state recurrence and 25% through latent-compressed sparse attention with anti-dilution centroid routing.

- **Strict $O(1)$ Decode Latency**: 35–37 ms/token flat up to 1,000,000 tokens on consumer GPUs.
- **39.6x KV-Cache Compression**: 1.20 GB for 1M tokens in FP16 (vs 48.8 GB for dense attention).
- **1,000,000 Token Fact Extraction**: Single-needle retrieval at Token #742,189 with Rank #1 out of 15,625 blocks and 100% fine-grained attention focus.
- **NoPE Temporal Invariance**: Replaces RoPE with exponential recurrent decay ($\alpha_t$) to prevent frequency phase distortion over long distances.

---

## Frontier Architectural Comparison

<p align="center">
  <img src="assets/architecture_comparison.svg" width="100%" alt="Architectural Comparison: Maba vs Qwen3.8-Flash-Next vs MiniCPM-5 vs Dense Transformer"/>
</p>

| Architecture | Macro-Topology | Attention Paradigm | Decode Complexity | KV Cache @ 131k | KV Cache @ 1M | Max Context |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Maba (Canonical)** | **3:1 Hybrid (15 DGDA : 5 MABA-SA)** | **Latent MLA ($d_c=128$) + 64:1 Centroids** | **O(1) Flat (35 ms)** | **163.6 MB** | **1.20 GB** | **1,000,000+ (NoPE)** |
| **Qwen3.8-Flash-Next** | Hybrid GDN + QSA MoE (6B active) | Micro-Block Sparse Attention | Sublinear $O(\log L)$ | 640.0 MB | 4.80 GB | 262k / 1M (YaRN) |
| **MiniCPM-5** | Dense CausalLM (1B / 2B) | 100% Dense GQA | Linear $O(L)$ | 3.20 GB | 24.50 GB | 131,072 (RoPE) |
| **Dense Transformer** | Standard Transformer | 100% Dense Softmax MHA | Linear $O(L)$ | 6.40 GB | 48.82 GB | 64,000 max (OOM) |

---

## Context Scaling & Memory Footprint

<p align="center">
  <img src="assets/scaling_comparison.svg" width="100%" alt="Scaling Comparison: Latency &amp; Memory vs Context Length"/>
</p>

### KV-Cache Allocation Across Context Lengths (FP16 Megabytes)

| Context Length | Dense MHA (MB) | MiniCPM-5 (MB) | Qwen Flash-Next (MB) | Maba (MB) | Maba Memory Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | 50.00 | 25.00 | 5.00 | **3.60** | **13.9x vs Dense** (6.9x vs MiniCPM-5) |
| **16,384** | 800.00 | 400.00 | 80.00 | **22.50** | **35.6x vs Dense** (17.8x vs MiniCPM-5) |
| **65,536** | 3,200.00 | 1,600.00 | 320.00 | **82.97** | **38.6x vs Dense** (19.3x vs MiniCPM-5) |
| **131,072** | 6,400.00 | 3,200.00 | 640.00 | **163.59** | **39.1x vs Dense** (19.6x vs MiniCPM-5) |
| **262,144** | 12,800.00 | 6,400.00 | 1,280.00 | **324.84** | **39.4x vs Dense** (19.7x vs MiniCPM-5) |
| **1,000,000** | 48,828.12 | 24,414.06 | 4,882.81 | **1,232.58** | **39.6x vs Dense** (19.8x vs MiniCPM-5) |

---

## Architectural Specifications

- **Reference Config**: [config.json](config.json) (101,282,319 parameters).
- **Macro-Stack (20 layers, 3:1 ratio)**:
  - 15 layers: [DGDA (Decoupled Gated Delta Attention)](maba_sparse/layers/dgda.py) linear recurrence.
  - 5 layers: [MABA-SA (Sparse Attention)](maba_sparse/layers/sparse_attention.py) with MLA latent compression ($d_c=128$).
- **Factorized Embeddings**: Vocab 32,768 -> 128 -> 640 in [maba_sparse/model.py](maba_sparse/model.py).
- **Anti-Dilution Routing**: [DG-Indexer](maba_sparse/layers/indexer.py) uses hybrid pooling $\frac{1}{2}(\text{mean} + \text{max})$ with distance decay penalty $\lambda \log(1+\Delta)$ to protect salient single-token facts against background noise.
- **Superposition Attention**: 3 streams dynamically superposed via data-dependent gate logits:
  1. Local Sliding Window (128 tokens + 4 sinks).
  2. Sparse Top-32 Blocks (2,048 gathered tokens).
  3. Hierarchical Context Attention (HCA 64:1 compressed prefix).
- **Multi-Backend Kernels**: [maba_sparse/kernels/](maba_sparse/kernels/) provides fused Triton GPU kernels (264k–375k tok/s on Tesla T4), CPU OpenMP parallel kernels, and PyTorch autograd fallbacks.

---

## Quick Start

### Installation

```bash
git clone https://github.com/AndrewThompson1233/maba-v2-architecture.git
cd maba-v2-architecture
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

### Training

```bash
# Distributed Data Parallel training on multi-GPU
torchrun --nproc_per_node=2 train.py \
    --model maba_sparse \
    --dataset synthetic \
    --steps 100 \
    --batch_size 4 \
    --fp16
```

---

## Benchmark Suite

All benchmark suites are consolidated in [benchmark.py](benchmark.py). Full hardware metrics on Tesla T4 GPUs are recorded in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) and [benchmark_results.json](benchmark_results.json).

```bash
# Frontier architectural comparison vs Qwen3.8-Flash-Next & MiniCPM-5
python benchmark.py --mode arch

# End-to-end model & attention scaling (Maba vs Dense Transformer)
python benchmark.py --mode model --contexts 128,256,512,1024,2048,4096

# Constant O(1) decode latency scaling
python benchmark.py --mode decode

# KV-cache footprint comparison (40x reduction)
python benchmark.py --mode memory

# 1,000,000 token single-needle fact extraction
python benchmark.py --mode needle

# 50 Hard Negatives ('semantic mines') & Multi-Hop reasoning across 640k tokens
python benchmark.py --mode multihop

# Triton hardware kernel throughput
python benchmark.py --mode triton

# Run all suites sequentially
python benchmark.py --mode all
```

---

## Test Suite

Run the full automated unit test suite (649 tests):

```bash
pytest -q
```

All 23 test modules in [tests/](tests/) verify causal masking, autograd graph integrity, memory invariance, numerical stability, and hardware kernel parity.

---

## License & Attribution

Maba v2 Architecture is released under the **MABA Open Architecture License (MOAL-1.0)**.

- **Author**: Andrew Thompson (`AndrewThompson1233`)
- **Commercial & Research Use**: Permitted without royalty fees.
- **Attribution**: Any derivative architecture, implementation, checkpoint, or paper must state:
  > `Created based on Maba v2 Architecture by Andrew Thompson`
- **Anti-Plagiarism Protection**: The name **Maba** / **Maba v2** and its foundational mechanisms (**DGDA**, **MABA-SA**, **DG-Indexer**, **HCA**) may not be renamed, rebranded, or claimed under different names when adapting or copying this architecture.

See [LICENSE](LICENSE) for the full license text.
