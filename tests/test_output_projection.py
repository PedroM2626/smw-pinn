"""Unit tests for the output projection that implements 10.47's excluded cell."""

import torch
import torch.nn as nn

from src.models import ProjectedDynamics

MAX_VX = 72.0
TERMINAL_VY = 64.0
MIN_VY = -80.0


class _ConstantBase(nn.Module):
    """A state-output network that always proposes the same next state."""

    def __init__(self, next_state: torch.Tensor):
        super().__init__()
        self.next_state = next_state

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.next_state.expand(state.shape[0], -1)


def _state(batch: int = 4) -> torch.Tensor:
    state = torch.zeros(batch, 8)
    state[:, 0] = torch.arange(batch, dtype=torch.float32) * 10.0
    state[:, 1] = 100.0
    state[:, 2] = 5.0
    state[:, 3] = -3.0
    state[:, 4] = 1.0
    return state


def test_projection_caps_an_adversarial_proposal_and_rederives_position() -> None:
    base = _ConstantBase(torch.tensor([[999.0, 999.0, 500.0, -900.0, 1.0, 0.0, 1.0, 0.0]]))
    model = ProjectedDynamics(base=base, state_dim=8)
    state = _state()
    out = model(state, torch.zeros(4, 6))

    assert float(out[:, 2].max()) == MAX_VX
    assert float(out[:, 3].min()) == MIN_VY
    assert torch.allclose(out[:, 0], state[:, 0] + MAX_VX / 16.0)
    assert torch.allclose(out[:, 1], state[:, 1] + MIN_VY / 16.0)


def test_projection_leaves_an_admissible_and_consistent_output_untouched() -> None:
    state = _state(1)
    vx, vy = 30.0, -20.0
    proposal = torch.tensor(
        [[float(state[0, 0]) + vx / 16.0, 100.0 + vy / 16.0, vx, vy, 1.0, 0.0, 0.0, 0.0]]
    )
    model = ProjectedDynamics(base=_ConstantBase(proposal), state_dim=8)
    out = model(state, torch.zeros(1, 6))
    assert torch.allclose(out, proposal, atol=1e-6)


def test_projection_repairs_a_position_the_network_derived_from_a_clipped_velocity() -> None:
    # The network advanced x by 300/16 - a velocity the engine cannot have.
    proposal = torch.tensor([[50.0, 100.0, 300.0, 10.0, 0.0, 0.0, 0.0, 0.0]])
    model = ProjectedDynamics(base=_ConstantBase(proposal), state_dim=8)
    state = _state(1)
    out = model(state, torch.zeros(1, 6))
    assert float(out[0, 2]) == MAX_VX
    assert float(out[0, 0]) == float(state[0, 0]) + MAX_VX / 16.0
    assert float(out[0, 0]) != 50.0, "the position must follow the projected velocity"


def test_contact_channels_pass_through_unprojected() -> None:
    proposal = torch.tensor([[0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 0.0, 1.0]])
    model = ProjectedDynamics(base=_ConstantBase(proposal), state_dim=8)
    out = model(_state(1), torch.zeros(1, 6))
    assert torch.allclose(out[0, 4:], proposal[0, 4:])


def test_projection_reports_its_guarantee_as_structural() -> None:
    model = ProjectedDynamics(base=_ConstantBase(torch.zeros(1, 8)), state_dim=8)
    assert model.bounds_imposed_by_construction is True
