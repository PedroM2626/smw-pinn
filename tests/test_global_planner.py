"""Tests for hierarchical planning: A*, waypoints, waypoint-MPC (no emulator)."""

import numpy as np
import torch

from src.models import HardResidualPINNDynamics
from src.planning.global_planner import (
    FREE,
    HAZARD,
    SOLID,
    HierarchicalMPCController,
    WaypointObjective,
    astar,
    build_occupancy_grid,
    extract_waypoints,
)


def _open_grid(w=20, h=10):
    return np.zeros((h, w), dtype=np.int64)


def test_astar_open_field_optimal():
    grid = _open_grid()
    path = astar(grid, (0, 0), (6, 0))
    assert path[0] == (0, 0) and path[-1] == (6, 0)
    assert len(path) == 7  # straight line, no detour


def test_astar_routes_around_wall_through_gap():
    grid = _open_grid(w=20, h=10)
    grid[:, 10] = SOLID
    grid[2, 10] = FREE  # single gap off the direct row
    path = astar(grid, (2, 5), (17, 5))
    assert path, "A* must route through the gap"
    assert (10, 2) in path
    # The wall is impermeable except at the gap: no other x=10 cell is used.
    assert [p for p in path if p[0] == 10] == [(10, 2)]


def test_astar_unreachable_and_bad_endpoints():
    grid = _open_grid()
    grid[:, 5] = SOLID  # full wall, no gap
    assert astar(grid, (0, 0), (9, 0)) == []
    assert astar(grid, (5, 0), (9, 0)) == []  # start inside wall
    assert astar(grid, (0, 0), (5, 0)) == []  # goal inside wall
    assert astar(grid, (-1, 0), (9, 0)) == []  # out of bounds


def test_astar_no_corner_cutting():
    grid = _open_grid(w=6, h=6)
    grid[1, 1] = SOLID
    # Direct diagonal (0,0)->(1,1) is blocked; (0,0)->(2,2) must go around.
    path = astar(grid, (0, 0), (2, 2))
    assert path
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        if abs(x1 - x0) == 1 and abs(y1 - y0) == 1:
            assert grid[y0, x1] != SOLID and grid[y1, x0] != SOLID


def test_astar_avoids_hazard_corridor_when_alternative_exists():
    grid = _open_grid(w=12, h=7)
    grid[3, 2:10] = HAZARD  # hazard corridor on the direct row
    cheap = astar(grid, (0, 3), (11, 3), hazard_cost=25.0)
    pricey = astar(grid, (0, 3), (11, 3), hazard_cost=0.0)
    assert cheap and pricey
    cheap_haz = sum(1 for x, y in cheap if grid[y, x] == HAZARD)
    pricey_haz = sum(1 for x, y in pricey if grid[y, x] == HAZARD)
    assert cheap_haz <= pricey_haz


def test_astar_prefers_lower_climb():
    grid = _open_grid(w=10, h=10)
    flat = astar(grid, (0, 8), (9, 8), climb_cost=5.0)
    assert flat
    # A flat route must not contain upward steps when flat cells exist.
    assert all(y1 >= y0 for (_, y0), (_, y1) in zip(flat[:-1], flat[1:]))


def test_build_occupancy_grid_uses_reader():
    grid = build_occupancy_grid(lambda tx, ty: SOLID if ty == 4 else FREE, 6, 6)
    assert grid.shape == (6, 6)
    assert (grid[4] == SOLID).all() and (grid[0] == FREE).all()


def test_extract_waypoints_compresses_and_keeps_goal():
    path = [(x, 5) for x in range(10)] + [(9, 4), (9, 3)]
    pts = extract_waypoints(path, min_gap_px=48.0)
    assert pts[0] == ((0 + 0.5) * 16, (5 + 0.5) * 16)
    assert pts[-1] == ((9 + 0.5) * 16, (3 + 0.5) * 16)
    assert len(pts) < len(path)
    for a, b in zip(pts[:-1], pts[1:]):
        if b != pts[-1]:
            assert abs(b[0] - a[0]) + abs(b[1] - a[1]) >= 1.0
    assert extract_waypoints([]) == []


def test_waypoint_objective_prefers_target():
    obj = WaypointObjective(target_x=100.0, target_y=300.0, weight_target=2.0)
    init = torch.zeros(2, 8)
    good = torch.zeros(2, 5, 8)
    good[:, :, 0] = 100.0
    good[:, :, 1] = 300.0
    bad = torch.zeros(2, 5, 8)
    rewards_good = obj.compute_trajectory_rewards(init, good)
    rewards_bad = obj.compute_trajectory_rewards(init, bad)
    assert (rewards_good > rewards_bad).all()


def test_hierarchical_controller_plans_and_advances():
    device = torch.device("cpu")
    model = HardResidualPINNDynamics(hidden_dims=[16])
    ctrl = HierarchicalMPCController(
        world_model=model,
        device=device,
        horizon=5,
        num_candidates=8,
        cem_iterations=1,
        capture_radius_px=24.0,
    )
    ctrl.set_path([(100.0, 300.0), (200.0, 300.0), (300.0, 300.0)])
    s = np.array([16.0, 300.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action, info = ctrl.plan(s)
    assert action.shape == (6,)
    assert info["num_waypoints"] == 3
    assert info["active_waypoint"] == (100.0, 300.0)
    # Teleport near waypoint 1: index must advance past captured waypoints.
    s2 = s.copy()
    s2[0], s2[1] = 100.0, 300.0
    _, info2 = ctrl.plan(s2)
    assert info2["waypoint_index"] >= 1


def test_hierarchical_controller_empty_path_falls_back():
    device = torch.device("cpu")
    model = HardResidualPINNDynamics(hidden_dims=[16])
    ctrl = HierarchicalMPCController(
        world_model=model, device=device, horizon=5, num_candidates=8, cem_iterations=1
    )
    s = np.array([16.0, 300.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action, info = ctrl.plan(s)
    assert action.shape == (6,)
    assert info["active_waypoint"] is None
