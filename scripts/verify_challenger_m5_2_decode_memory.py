
import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer, ConvState
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config
from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.kernels.dispatcher import is_cuda_sm75_available, get_backend


def run_dgda_layer_adversarial_profiling(device: torch.device, num_decode_steps: int = 150) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"1. ADVERSARIAL PROFILING: DGDALayer.step() across Diverse Histories ({num_decode_steps} steps)")
    print("=" * 80)

    cfg = get_101m_config(n_layers=1)
    layer = DGDALayer(cfg).to(device)
    layer.eval()

    histories = [0, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []

    B, D = 1, cfg.dim
    expected_bytes = B * cfg.n_heads * cfg.d_head * cfg.d_head * 4

    for L in histories:
        torch.cuda.empty_cache()
        gc.collect()

        if L > 0:
            x_hist = torch.randn(B, L, D, device=device)
            with torch.no_grad():
                _, init_state, init_conv = layer(x_hist)
        else:
            init_state, init_conv = None, None

        x_step = torch.randn(B, 1, D, device=device)

        curr_state = init_state.clone() if init_state is not None else None
        curr_conv = ConvState(t.clone() for t in init_conv) if init_conv is not None else None

        with torch.no_grad():
            for _ in range(5):
                _, curr_state, curr_conv = layer.step(x_step, state=curr_state, conv_state=curr_conv)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            mem_start = torch.cuda.memory_allocated(device)
            peak_start = torch.cuda.max_memory_allocated(device)
        else:
            mem_start = 0
            peak_start = 0

        latencies = []
        step_checkpoints = {}

        t_start = time.perf_counter()
        with torch.no_grad():
            for step in range(1, num_decode_steps + 1):
                t0 = time.perf_counter()
                _, curr_state, curr_conv = layer.step(x_step, state=curr_state, conv_state=curr_conv)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                if step in [10, 50, 100, num_decode_steps] and device.type == "cuda":
                    step_checkpoints[step] = torch.cuda.memory_allocated(device)
        t_end = time.perf_counter()

        if device.type == "cuda":
            mem_end = torch.cuda.memory_allocated(device)
            peak_end = torch.cuda.max_memory_allocated(device)
            mem_growth = mem_end - mem_start
            peak_growth = peak_end - peak_start
        else:
            mem_end, peak_end, mem_growth, peak_growth = 0, 0, 0, 0

        state_bytes = curr_state.element_size() * curr_state.nelement()
        state_shape = list(curr_state.shape)
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        row = {
            "history_len": L,
            "decode_steps": num_decode_steps,
            "state_shape": state_shape,
            "state_bytes": state_bytes,
            "expected_state_bytes": expected_bytes,
            "mem_start_bytes": mem_start,
            "mem_end_bytes": mem_end,
            "mem_growth_bytes": mem_growth,
            "peak_vram_mb": peak_end / (1024 * 1024),
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "checkpoints": step_checkpoints,
        }
        results.append(row)

        print(f"History L={L:5d} | State: {state_shape} ({state_bytes} B) | "
              f"Steps: {num_decode_steps} | Mem Growth: {mem_growth:+d} B | "
              f"Peak VRAM: {row['peak_vram_mb']:.2f} MB | Latency: avg={avg_latency:.3f}ms p95={p95_latency:.3f}ms")

    state_sizes = [r["state_bytes"] for r in results]
    mem_growths = [r["mem_growth_bytes"] for r in results]

    assert len(set(state_sizes)) == 1, f"State bytes not constant across histories: {state_sizes}"
    assert state_sizes[0] == expected_bytes, f"State size {state_sizes[0]} != {expected_bytes}"
    assert all(m == 0 for m in mem_growths), f"Memory leak detected in DGDALayer.step: {mem_growths}"

    print(f"\n>> EMPIRICAL CERTIFICATION: DGDALayer.step maintains strict O(1) state ({state_sizes[0]} B)")
    print(f">> EMPIRICAL CERTIFICATION: Zero memory leak (0 bytes net growth) over {num_decode_steps} steps across all histories.")
    return {"results": results, "status": "APPROVED"}


def run_full_model_adversarial_profiling(device: torch.device, num_decode_steps: int = 100) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"2. ADVERSARIAL PROFILING: Full 101M Model Decode Memory Scaling ({num_decode_steps} steps)")
    print("=" * 80)

    cfg = get_101m_config(n_layers=20)
    model = MabaSparseForCausalLM(cfg).to(device)
    model.eval()

    dense_model = DenseTransformerForCausalLM().to(device)
    dense_model.eval()

    histories = [64, 256, 512, 1024, 2048, 4096]
    results = []

    for L in histories:
        torch.cuda.empty_cache()
        gc.collect()

        prompt = torch.randint(1, cfg.vocab_size, (1, L), device=device)
        step_tok = torch.randint(1, cfg.vocab_size, (1, 1), device=device)

        with torch.no_grad():
            out_maba = model(prompt)
            past_maba = out_maba.past_states

        with torch.no_grad():
            out_dense = dense_model(prompt)
            past_dense = out_dense.past_states

        curr_maba = past_maba
        curr_dense = past_dense
        with torch.no_grad():
            for _ in range(3):
                curr_maba = model(step_tok, past_states=curr_maba).past_states
                curr_dense = dense_model(step_tok, past_states=curr_dense).past_states
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            maba_mem_start = torch.cuda.memory_allocated(device)
        else:
            maba_mem_start = 0

        maba_latencies = []
        dgda_bytes_snapshots = []

        with torch.no_grad():
            for step in range(1, num_decode_steps + 1):
                t0 = time.perf_counter()
                out = model(step_tok, past_states=curr_maba)
                curr_maba = out.past_states
                step_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                maba_latencies.append((t1 - t0) * 1000.0)

                if step % 20 == 0 or step == num_decode_steps:
                    dgda_sum = sum(s[0].element_size() * s[0].nelement() for s in curr_maba if s[0] is not None)
                    dgda_bytes_snapshots.append((step, dgda_sum))

        if device.type == "cuda":
            maba_mem_end = torch.cuda.memory_allocated(device)
            maba_mem_growth = maba_mem_end - maba_mem_start
            maba_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        else:
            maba_mem_end, maba_mem_growth, maba_peak_mb = 0, 0, 0

        dense_latencies = []
        with torch.no_grad():
            for _ in range(num_decode_steps):
                t0 = time.perf_counter()
                out_d = dense_model(step_tok, past_states=curr_dense)
                curr_dense = out_d.past_states
                step_tok = torch.argmax(out_d.logits[:, -1, :], dim=-1, keepdim=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                dense_latencies.append((t1 - t0) * 1000.0)

        maba_dgda_bytes = sum(s[0].element_size() * s[0].nelement() for s in curr_maba if s[0] is not None)
        maba_mla_bytes = sum(s[2].element_size() * s[2].nelement() for s in curr_maba if s[2] is not None)
        maba_total_cache = maba_dgda_bytes + maba_mla_bytes

        dense_cache_bytes = 0
        for s in curr_dense:
            if isinstance(s, (tuple, list)):
                for t in s:
                    if isinstance(t, torch.Tensor):
                        dense_cache_bytes += t.element_size() * t.nelement()

        compression = dense_cache_bytes / max(maba_total_cache, 1)

        row = {
            "history_len": L,
            "decode_steps": num_decode_steps,
            "final_seq_len": L + num_decode_steps,
            "maba_dgda_bytes": maba_dgda_bytes,
            "maba_mla_bytes": maba_mla_bytes,
            "maba_total_cache_bytes": maba_total_cache,
            "dense_total_cache_bytes": dense_cache_bytes,
            "cache_compression_ratio": compression,
            "maba_mem_growth_bytes": maba_mem_growth,
            "maba_peak_mb": maba_peak_mb,
            "maba_avg_latency_ms": sum(maba_latencies) / len(maba_latencies),
            "dense_avg_latency_ms": sum(dense_latencies) / len(dense_latencies),
            "dgda_snapshots": dgda_bytes_snapshots,
        }
        results.append(row)

        print(f"History L={L:5d} (+{num_decode_steps:3d} tok) | "
              f"DGDA State: {maba_dgda_bytes/1024:6.1f} KB (CONST) | "
              f"Maba Cache: {maba_total_cache/1024:7.1f} KB | "
              f"Dense KV: {dense_cache_bytes/1024:9.1f} KB | "
              f"Compression: {compression:5.1f}x | "
              f"Maba Latency: {row['maba_avg_latency_ms']:5.2f} ms | "
              f"Dense Latency: {row['dense_avg_latency_ms']:5.2f} ms")

    dgda_bytes_list = [r["maba_dgda_bytes"] for r in results]
    assert len(set(dgda_bytes_list)) == 1, f"DGDA state bytes varied across histories: {dgda_bytes_list}"
    assert dgda_bytes_list[0] == 15 * 163840, f"DGDA state bytes {dgda_bytes_list[0]} != expected {15 * 163840}"

    print(f"\n>> EMPIRICAL CERTIFICATION: Full Model DGDA recurrent state is strictly invariant ({dgda_bytes_list[0]/1024:.1f} KB)")
    print(f">> EMPIRICAL CERTIFICATION: KV Cache Compression reaches {results[-1]['cache_compression_ratio']:.1f}x at L=4096+{num_decode_steps}")
    return {"results": results, "status": "APPROVED"}


def main():
    parser = argparse.ArgumentParser(description="Adversarial Decode Scaling Verification")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dgda_steps", type=int, default=150)
    parser.add_argument("--model_steps", type=int, default=100)
    args = parser.parse_args()

    dev = torch.device(args.device)
    print(f"Starting Adversarial Decode Memory Scaling Profiling on {dev}...")
    if dev.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(dev.index if dev.index is not None else 0)}")
        print(f"Triton Backend Active: {get_backend(dev)}")

    t0 = time.time()
    dgda_res = run_dgda_layer_adversarial_profiling(dev, num_decode_steps=args.dgda_steps)
    model_res = run_full_model_adversarial_profiling(dev, num_decode_steps=args.model_steps)
    total_time = time.time() - t0

    verdict = "APPROVE" if (dgda_res["status"] == "APPROVED" and model_res["status"] == "APPROVED") else "REJECT"

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "verdict": verdict,
        "total_time_seconds": total_time,
        "dgda_layer_profiling": dgda_res,
        "full_model_profiling": model_res,
    }

    out_file = "challenger_m5_2_decode_scaling_results.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"FINAL CHALLENGER VERDICT: {verdict}")
    print(f"Results written to {out_file} (Duration: {total_time:.1f}s)")
    print("=" * 80)

    if verdict != "APPROVE":
        sys.exit(1)


if __name__ == "__main__":
    main()
