"""
test_online_mbpo.py
Unit tests for RealReplayBuffer and online MBPO components.
"""

import pytest
import numpy as np
from src.training.online_mbpo import RealReplayBuffer


def test_real_replay_buffer_add_and_sample():
    buf = RealReplayBuffer(capacity=100, state_dim=8, action_dim=6)
    assert len(buf) == 0

    s = np.zeros(8, dtype=np.float32)
    a = np.ones(6, dtype=np.float32)
    r = 1.5
    ns = np.ones(8, dtype=np.float32)
    d = False

    for _ in range(50):
        buf.add(s, a, r, ns, d)

    assert len(buf) == 50

    b_s, b_a, b_r, b_ns, b_d = buf.sample(batch_size=16)
    assert b_s.shape == (16, 8)
    assert b_a.shape == (16, 6)
    assert b_r.shape == (16,)
    assert b_ns.shape == (16, 8)
    assert b_d.shape == (16,)

    sampled_states = buf.sample_states(batch_size=16)
    assert sampled_states.shape == (16, 8)
