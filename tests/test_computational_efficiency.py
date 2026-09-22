"""
test_computational_efficiency.py
Unit tests for the hardware & computational efficiency profiling suite.
"""

import torch

from src.evaluation.benchmark_computational_efficiency import (
    benchmark_latency,
    count_parameters,
    estimate_flops,
)
from src.models import HardResidualPINNDynamics, StatisticalMLPDynamics


def test_parameter_counter():
    model = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    n_params = count_parameters(model)
    assert isinstance(n_params, int)
    assert n_params > 5000


def test_flops_estimation():
    model = StatisticalMLPDynamics(state_dim=8, action_dim=6)
    flops = estimate_flops(model, state_dim=8, action_dim=6)
    assert isinstance(flops, int)
    assert flops > 10000


def test_latency_benchmark_cpu():
    model = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    lat_us, fps = benchmark_latency(
        model,
        torch.device("cpu"),
        state_dim=8,
        action_dim=6,
        batch_size=4,
        num_iterations=10,
        warmup=2,
    )
    assert lat_us > 0.0
    assert fps > 0.0
