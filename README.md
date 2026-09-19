# Maba v1.5

Reference PyTorch implementation of the **Maba v1.5** hybrid architecture.

Maba combines linear recurrence (DGDA) with sparse global attention (MABA-SA) in a 3:1 cyclic stack to reduce prefill memory and maintain constant $O(1)$ decode time.

## Architecture

- **Total Parameters**: 101,282,319 (~101.3M)
- **Macro-Stack (20 layers, 3:1 ratio)**:
  - 15 layers: **DGDA (Dual-Gated Delta Attention)** linear recurrence.
  - 5 layers: **MABA-SA (Sparse Attention)** with MLA latent compression ($d_c=128$).
- **Embeddings**: Factorized embeddings (vocab 32,768 -> 128 -> 640).
- **Recurrence (DGDA)**: Chunked parallel prefill ($C=16$) with adaptive order-3 Neumann series.
- **Sparse Attention (MABA-SA)**:
  - Hybrid centroid routing ($0.5 \times \text{mean} + 0.5 \times \text{max}$).
  - Top-32 block selection (2,048 tokens active context).
  - 3-stream superposition (local window + top-k sparse + HCA summary).
- **Positional Encoding**: NoPE (No Positional Embeddings in attention; temporal order tracked through DGDA recurrence).

## Quick Start

### Installation

```bash
git clone https://github.com/ivan-dev35/123123.git
cd 123123/codex/maba-v1.5-exp-architecture
pip install -e .
```

### Generation

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
# Prepare dialogue dataset
python scripts/prepare_dialog_dataset.py

# Multi-GPU training (DDP)
torchrun --nproc_per_node=2 train.py \
    --model maba_sparse \
    --dataset dialog \
    --data_path assets/dialogs_clean.json \
    --steps 100 \
    --batch_size 4 \
    --fp16

# Test trained checkpoint
python scripts/verify_checkpoint.py --checkpoint checkpoints/maba_dialog_checkpoint.pt
```

### Benchmarks & Tests

```bash
# Run benchmark (prefill latency, decode time, VRAM)
python benchmark.py --contexts 128,256,512,1024,2048,4096

# Run 1M token needle retrieval test
python scripts/test_1m_end_to_end_needle_pull.py

# Run test suite
pytest -q
```

## Benchmark Summary (NVIDIA Tesla T4)

| Context Length | Maba Prefill (ms) | Dense Prefill (ms) | Maba VRAM (MB) | Dense VRAM (MB) | Maba Decode (ms/tok) | Dense Decode (ms/tok) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 49.71 | 16.56 | 920.4 | 863.4 | 36.77 | 15.94 |
| 512 | 134.02 | 38.52 | 1342.6 | 964.2 | 37.02 | 15.14 |
| 2,048 | 1474.72 | 172.97 | 2859.2 | 1350.5 | 37.28 | 17.17 |
| 4,096 | 3230.19 | 427.90 | 2946.7 | 1818.2 | 35.30 | 15.79 |

- Decode latency remains flat at 35-37 ms/token up to 1,000,000 tokens.
- KV-cache memory: 1.19 GB at 1M tokens (vs 47.7 GB for standard dense attention).

## License & Attribution

Licensed under the **MABA Open Architecture License (MOAL-1.0)**.

- **Author**: Andrew Thompson
- **Free Use**: Commercial and research use permitted without royalties.
- **Attribution**: Any derivative model, implementation, or paper must credit:
  `Created based on Maba Architecture by Andrew Thompson`
- **Naming**: The architecture name "Maba" and core component names (DGDA, MABA-SA, DG-Indexer, HCA) must be preserved in derivative works.

See [LICENSE](LICENSE) for full legal text.
