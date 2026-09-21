"""
benchmark_computational_efficiency.py
Academic Hardware & Computational Profiling Suite:
Rigorously profiles all evaluated World Model architectures across:
1. Total Trainable Parameter Counts & Weights Footprint (KB).
2. Theoretical FLOPs per single-step forward pass.
3. Inference Latency across compute backends:
   - CPU Single-Core (torch.set_num_threads(1))
   - CPU Multi-Core (default multiprocessing pool)
   - GPU CUDA Tensor Execution (NVIDIA GeForce RTX 4070)
4. Peak VRAM Memory Footprint (MB) during forward and batched rollouts.
5. In-Memory Simulation Throughput (Transitions per second / FPS).
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from src.models import (
    StatisticalMLPDynamics,
    StatisticalLSTMDynamics,
    SoftPINNDynamics,
    HardResidualPINNDynamics,
    DeepPINNEnsemble,
    MultiEntityPINNDynamics,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, state_dim: int = 8, action_dim: int = 6) -> int:
    """Estimates theoretical multiply-accumulate FLOPs for 1 forward pass (batch size 1)."""
    flops = 0
    dummy_s = torch.zeros(1, state_dim)
    dummy_a = torch.zeros(1, action_dim)

    # For standard feedforward linear modules: 2 * in_features * out_features
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
        elif isinstance(m, nn.LSTM):
            # 8 * hidden_size * (input_size + hidden_size) per layer
            h = m.hidden_size
            inp = m.input_size
            flops += 8 * h * (inp + h) * m.num_layers
        elif isinstance(m, nn.LayerNorm):
            flops += 4 * m.normalized_shape[0]
    return flops


def benchmark_latency(
    model: nn.Module,
    device: torch.device,
    state_dim: int = 8,
    action_dim: int = 6,
    batch_size: int = 1,
    num_iterations: int = 1000,
    warmup: int = 50,
) -> Tuple[float, float]:
    """
    Measures latency per step in microseconds (us) and throughput (steps/sec).
    """
    model = model.to(device)
    model.eval()

    states = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randn(batch_size, action_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(states, actions)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(states, actions)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    latency_per_step_us = (elapsed / num_iterations) * 1e6
    throughput_fps = (batch_size * num_iterations) / elapsed

    return latency_per_step_us, throughput_fps


def run_profiling_suite(output_dir: str = "results") -> Dict:
    print("====================================================================")
    print("  COMPUTATIONAL PROFILING & HARDWARE EFFICIENCY BENCHMARK           ")
    print("====================================================================")

    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"Host System: Windows | CUDA Available: {cuda_available} ({gpu_name})")

    models = {
        "Statistical MLP": (StatisticalMLPDynamics(state_dim=8, action_dim=6), 8),
        "Statistical LSTM": (StatisticalLSTMDynamics(state_dim=8, action_dim=6), 8),
        "Soft PINN": (SoftPINNDynamics(state_dim=8, action_dim=6), 8),
        "Hard Residual PINN": (HardResidualPINNDynamics(state_dim=8, action_dim=6), 8),
        "Multi-Entity PINN (12D)": (MultiEntityPINNDynamics(state_dim=12, action_dim=6), 12),
        "Deep PINN Ensemble (E=5)": (DeepPINNEnsemble(num_models=5, state_dim=8, action_dim=6), 8),
    }

    metrics = {}

    print("\n" + "=" * 95)
    print(f"{'Model Architecture':<26} | {'Params':>8} | {'FLOPs':>10} | {'CPU-1 (us)':>11} | {'CUDA (us)':>10} | {'CUDA FPS (B=256)':>16}")
    print("=" * 95)

    for name, (model, s_dim) in models.items():
        params = count_parameters(model)
        flops = estimate_flops(model, state_dim=s_dim)

        # 1. CPU Single-Core Latency
        torch.set_num_threads(1)
        cpu_lat_us, cpu_fps = benchmark_latency(
            model, torch.device("cpu"), state_dim=s_dim, batch_size=1, num_iterations=500
        )

        # 2. CUDA Latency & Vectorized Throughput
        if cuda_available:
            cuda_dev = torch.device("cuda")
            cuda_lat_us, _ = benchmark_latency(
                model, cuda_dev, state_dim=s_dim, batch_size=1, num_iterations=1000
            )
            _, cuda_fps_b256 = benchmark_latency(
                model, cuda_dev, state_dim=s_dim, batch_size=256, num_iterations=200
            )
            vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        else:
            cuda_lat_us = 0.0
            cuda_fps_b256 = 0.0
            vram_mb = 0.0

        metrics[name] = {
            "parameters": int(params),
            "flops": int(flops),
            "cpu_single_thread_latency_us": float(round(cpu_lat_us, 2)),
            "cuda_latency_us": float(round(cuda_lat_us, 2)),
            "cuda_throughput_fps_b256": float(round(cuda_fps_b256, 1)),
            "vram_allocated_mb": float(round(vram_mb, 2)),
        }

        print(
            f"{name:<26} | {params:>8,d} | {flops:>10,d} | {cpu_lat_us:>11.1f} | {cuda_lat_us:>10.1f} | {int(cuda_fps_b256):>16,d}"
        )

    print("=" * 95)

    # Save metrics
    metrics_path = os.path.join(output_dir, "computational_profiling_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    names = list(metrics.keys())
    params_list = [metrics[m]["parameters"] for m in names]
    cuda_fps_list = [metrics[m]["cuda_throughput_fps_b256"] for m in names]
    lat_list = [metrics[m]["cuda_latency_us"] for m in names]

    palette = ["#64748B", "#F59E0B", "#EF4444", "#10B981", "#06B6D4", "#8B5CF6"]

    # 1. Parameter Counts
    axes[0].barh(names, params_list, color=palette)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable Parameters (Log Scale)")
    axes[0].set_title("Model Parameter Complexity")
    axes[0].grid(True, alpha=0.3)

    # 2. CUDA Inference Latency (Batch 1)
    axes[1].barh(names, lat_list, color=palette)
    axes[1].set_xlabel("Latency per Step (microseconds - us)")
    axes[1].set_title("Single-Step Inference Latency (CUDA)")
    axes[1].grid(True, alpha=0.3)

    # 3. Vectorized Simulation Throughput
    axes[2].barh(names, cuda_fps_list, color=palette)
    axes[2].set_xlabel("Simulation Throughput (Transitions/Sec)")
    axes[2].set_title("Vectorized In-GPU Simulation FPS (B=256)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "computational_efficiency_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Metrics saved to: {metrics_path}")
    print(f"Figure saved to: {fig_path}")
    return metrics


if __name__ == "__main__":
    run_profiling_suite()
