
import gc
import math
import time
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config
from maba_sparse.kernels.dispatcher import (
    is_cuda_sm75_available,
)

CUDA_AVAILABLE = torch.cuda.is_available() and is_cuda_sm75_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def count_parameters_by_layer(model: MabaSparseForCausalLM):
    layer_params: Dict[int, List[Tuple[str, nn.Parameter]]] = {i: [] for i in range(len(model.layers))}
    other_params: List[Tuple[str, nn.Parameter]] = []

    for name, p in model.named_parameters():
        if name.startswith("layers."):
            idx = int(name.split(".")[1])
            layer_params[idx].append((name, p))
        else:
            other_params.append((name, p))

    return layer_params, other_params


class TestAutogradCompleteness:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ for GPU kernel testing")
    def test_full_20_layer_autograd_completeness_cuda(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        cfg = get_101m_config(n_layers=20)
        model = MabaSparseForCausalLM(cfg).to("cuda")
        model.train()

        B, L = 2, 64
        x = torch.randint(1, cfg.vocab_size, (B, L), device="cuda")
        targets = torch.roll(x, -1, dims=1)
        targets[:, -1] = 0

        out = model(x, targets=targets)
        loss = out.loss
        assert loss is not None, "Loss must not be None"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        loss.backward()

        total_params = 0
        finite_grad_count = 0
        none_grad_params = []
        nan_or_inf_params = []
        zero_grad_params = []

        layer_params, other_params = count_parameters_by_layer(model)

        for name, p in other_params:
            total_params += 1
            if p.grad is None:
                none_grad_params.append(name)
            elif torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                nan_or_inf_params.append(name)
            elif (p.grad == 0).all():
                zero_grad_params.append(name)
            else:
                finite_grad_count += 1

        for layer_idx in range(20):
            layer = model.layers[layer_idx]
            is_attn = layer.is_attention
            for name, p in layer_params[layer_idx]:
                total_params += 1
                if p.grad is None:
                    none_grad_params.append(name)
                elif torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    nan_or_inf_params.append(name)
                elif (p.grad == 0).all():
                    zero_grad_params.append(name)
                else:
                    finite_grad_count += 1

        print(f"\n--- Autograd Completeness Report (20 Layers, {total_params} Tensors) ---")
        print(f"Finite non-zero gradients : {finite_grad_count}/{total_params} ({finite_grad_count/total_params*100:.1f}%)")
        print(f"None gradients (no graph) : {len(none_grad_params)}")
        print(f"NaN / Inf gradients       : {len(nan_or_inf_params)}")
        print(f"Exact zero gradients      : {len(zero_grad_params)}")

        assert len(nan_or_inf_params) == 0, f"NaN/Inf gradients found in: {nan_or_inf_params}"
        assert len(zero_grad_params) == 0, f"Zero gradients found in: {zero_grad_params}"

        expected_none = [
            f"layers.{i}.mixer.indexer.{proj}.weight"
            for i in [3, 7, 11, 15, 19]
            for proj in ["q_idx_proj", "k_idx_proj"]
        ]
        assert sorted(none_grad_params) == sorted(expected_none), (
            f"Unexpected parameters without gradients: {set(none_grad_params) - set(expected_none)}"
        )

        for layer_idx in range(20):
            if not model.layers[layer_idx].is_attention:
                for name, p in layer_params[layer_idx]:
                    assert p.grad is not None, f"DGDA parameter {name} in layer {layer_idx} has no grad!"
                    assert not torch.isnan(p.grad).any(), f"DGDA parameter {name} has NaN grad!"
                    assert not (p.grad == 0).all(), f"DGDA parameter {name} has zero grad!"

    def test_autograd_completeness_small_cpu_or_cuda(self):
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
        model = MabaSparseForCausalLM(cfg).to(DEVICE)
        model.train()

        B, L = 2, 32
        x = torch.randint(1, 1000, (B, L), device=DEVICE)
        out = model(x, targets=x)
        out.loss.backward()

        for name, p in model.named_parameters():
            if "indexer." in name:
                continue
            assert p.grad is not None, f"Parameter {name} has None grad!"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has NaN/Inf grad!"
            assert not (p.grad == 0).all(), f"Parameter {name} has all-zero grad!"


class TestAutoregressiveDecodeMemoryScaling:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA for GPU VRAM profiling")
    def test_dgda_layer_step_strict_o1_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=1)
        layer = DGDALayer(cfg).to("cuda").eval()

        B, D = 1, cfg.dim
        x = torch.randn(B, 1, D, device="cuda")
        state = None
        conv_state = None

        with torch.no_grad():
            for _ in range(5):
                _, state, conv_state = layer.step(x, state=state, conv_state=conv_state)

            torch.cuda.synchronize()
            mem_start = torch.cuda.memory_allocated()

            step_memories = []
            latencies = []

            for step in range(100):
                t0 = time.perf_counter()
                _, state, conv_state = layer.step(x, state=state, conv_state=conv_state)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000.0)
                step_memories.append(torch.cuda.memory_allocated())

            torch.cuda.synchronize()
            mem_end = torch.cuda.memory_allocated()
            mem_delta = mem_end - mem_start

        print(f"\n--- DGDALayer.step() 100-Step Scaling Profile (Inference) ---")
        print(f"Initial Memory  : {mem_start / 1024:.2f} KB")
        print(f"Final Memory    : {mem_end / 1024:.2f} KB")
        print(f"Net Memory Delta: {mem_delta} bytes")
        print(f"Mean Latency    : {sum(latencies)/len(latencies):.4f} ms/step")
        print(f"Early Latency   : {sum(latencies[:20])/20:.4f} ms/step")
        print(f"Late Latency    : {sum(latencies[-20:])/20:.4f} ms/step")

        assert mem_delta == 0, f"DGDALayer.step() leaked memory: {mem_delta} bytes over 100 steps"

        early_lat = sum(latencies[:20]) / 20
        late_lat = sum(latencies[-20:]) / 20
        ratio = late_lat / (early_lat + 1e-6)
        assert ratio < 1.5, f"Step latency degraded by factor {ratio:.2f} (> 1.5)"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA for GPU VRAM profiling")
    def test_full_model_stateful_sequential_decode_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

        cfg = get_101m_config(n_layers=20)
        model = MabaSparseForCausalLM(cfg).to("cuda").eval()

        B = 1
        prompt = torch.randint(1, cfg.vocab_size, (B, 1), device="cuda")

        with torch.no_grad():
            out = model(prompt)
            past_states = out.past_states

        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        memory_snapshots = []
        latencies = []

        tok = torch.randint(1, cfg.vocab_size, (B, 1), device="cuda")
        with torch.no_grad():
            for t in range(1, 101):
                t0 = time.perf_counter()
                out = model(tok, past_states=past_states)
                past_states = out.past_states
                tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000.0)
                if t % 20 == 0:
                    memory_snapshots.append((t, torch.cuda.memory_allocated()))

        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()

        print(f"\n--- Full 20-Layer Stateful Decode Profile (T=1..100) ---")
        for step_idx, m_val in memory_snapshots:
            print(f"Step {step_idx:3d}: {m_val / (1024*1024):.2f} MB (delta: {(m_val - mem_start)/1024:+.1f} KB)")

        net_growth = mem_end - mem_start
        print(f"Net memory growth over 100 steps: {net_growth / 1024:.1f} KB")
        assert net_growth < 2 * 1024 * 1024, f"Memory grew excessively: {net_growth / (1024*1024):.2f} MB"

        early_lat = sum(latencies[5:25]) / 20
        late_lat = sum(latencies[80:100]) / 20
        ratio = late_lat / (early_lat + 1e-6)
        print(f"Latency: early={early_lat:.2f} ms/step, late={late_lat:.2f} ms/step, ratio={ratio:.2f}")
        assert ratio < 1.6, f"Latency grew by factor {ratio:.2f} (> 1.6)"


class TestAmpFp16GradScalerStability:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ for AMP FP16")
    def test_amp_fp16_forward_backward_multi_step(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)

        cfg = get_101m_config(n_layers=20)
        model = MabaSparseForCausalLM(cfg).to("cuda")
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")

        B, L = 2, 64
        initial_scale = scaler.get_scale()
        step_losses = []

        print(f"\n--- AMP FP16 Training Stress Test (20 Layers, 10 Steps) ---")
        for step in range(1, 11):
            x = torch.randint(1, cfg.vocab_size, (B, L), device="cuda")
            targets = torch.roll(x, -1, dims=1)
            targets[:, -1] = 0

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x, targets=targets)
                loss = out.loss

            assert loss is not None
            loss_val = loss.item()
            assert not math.isnan(loss_val), f"NaN loss at step {step}"
            assert not math.isinf(loss_val), f"Inf loss at step {step}"
            step_losses.append(loss_val)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            has_nan_grad = False
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        has_nan_grad = True
                        break

            assert not has_nan_grad, f"NaN gradient detected in unscaled gradients at step {step}"

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            assert torch.isfinite(grad_norm), f"Grad norm is non-finite at step {step}: {grad_norm}"

            scaler.step(optimizer)
            scaler.update()

            current_scale = scaler.get_scale()
            print(f"Step {step:2d} | Loss: {loss_val:.4f} | Grad Norm: {grad_norm.item():.4f} | Scale: {current_scale}")

        final_scale = scaler.get_scale()
        assert final_scale >= initial_scale * 0.25, (
            f"GradScaler collapsed from {initial_scale} to {final_scale}, indicating persistent NaN gradients!"
        )
        print(f"AMP FP16 test PASSED cleanly: final scale={final_scale}, all 10 steps finite.")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA sm_75+ for AMP FP16")
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
    def test_amp_fp16_across_sequence_lengths(self, seq_len: int):
        cfg = get_101m_config(n_layers=4)
        model = MabaSparseForCausalLM(cfg).to("cuda")
        model.train()

        B = 2
        x = torch.randint(1, cfg.vocab_size, (B, seq_len), device="cuda")
        targets = torch.roll(x, -1, dims=1)
        targets[:, -1] = 0

        scaler = torch.amp.GradScaler("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, targets=targets)
            loss = out.loss

        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite at seq_len={seq_len}: {loss.item()}"

        scaler.scale(loss).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient at seq_len={seq_len} in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient at seq_len={seq_len} in {name}"
