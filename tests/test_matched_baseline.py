"""Parameter-parity baseline: compact MLP (~10k) vs compact Hard PINN (~10k)."""

import torch

from src.models import (
    MATCHED_HIDDEN_DIMS,
    HardResidualPINNDynamics,
    build_param_matched_mlp,
)


def _count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_matched_pair_within_five_percent():
    mlp = build_param_matched_mlp()
    pinn = HardResidualPINNDynamics(hidden_dims=list(MATCHED_HIDDEN_DIMS))
    n_mlp, n_pinn = _count(mlp), _count(pinn)
    assert abs(n_mlp - n_pinn) / n_pinn < 0.05


def test_matched_mlp_forward_and_kinematic_gap():
    mlp = build_param_matched_mlp()
    s = torch.randn(8, 8)
    a = torch.randn(8, 6)
    assert mlp(s, a).shape == (8, 8)
    # The matched MLP has no structural guarantee (unlike the Hard PINN).
    assert not torch.allclose(
        mlp(s, a)[:, 0], s[:, 0] + mlp(s, a)[:, 2] / 16.0, atol=1e-6
    )
