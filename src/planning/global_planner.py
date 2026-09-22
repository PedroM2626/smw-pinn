"""Hierarchical global + local planning (Feature B).

Global layer: A* over the discrete WRAM tile occupancy grid ($7E:C800,
16x16 px tiles, 27 rows) finds a strategic route around vertical obstacles
that trap purely local MPC (limitation §10.30-5). The local layer (Hard PINN
CEM-MPC, 15-frame horizon) tracks the active waypoint.

Division of labor, stated honestly: A* gives *topological* guidance with a
climb penalty approximating jump effort; true jump-arc feasibility is left to
the local MPC, which forward-simulates physics. All helpers are pure
functions of a tile reader, so they unit-test without the emulator.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models import HardResidualPINNDynamics
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.logging import get_logger

logger = get_logger(__name__)

TILE_PX = 16
GRID_HEIGHT_TILES = 27

FREE, SOLID, HAZARD, SLOPE = 0, 1, 2, 3


def classify_tile(tile_id: int, tile_y: int) -> int:
    """Mirrors the WRAM tile classification in `SnesLibretroEmulator`."""
    if tile_id == 0x25:
        return FREE
    if tile_id in (0x80, 0x81, 0x82):
        return SLOPE
    if tile_id in (0x00, 0x3F, 0x73, 0x74, 0x75, 0x76) or (
        tile_id != 0x25 and tile_y >= 22
    ):
        return SOLID
    return SOLID if tile_id != 0x25 else FREE


def build_occupancy_grid(
    tile_reader: Callable[[int, int], int],
    width_tiles: int,
    height_tiles: int = GRID_HEIGHT_TILES,
) -> np.ndarray:
    """Builds [H, W] occupancy grid via `tile_reader(tx, ty) -> class`."""
    grid = np.zeros((height_tiles, width_tiles), dtype=np.int64)
    for ty in range(height_tiles):
        for tx in range(width_tiles):
            grid[ty, tx] = int(tile_reader(tx, ty))
    return grid


def build_global_grid_from_emulator(emu, num_subscreens: int = 32) -> np.ndarray:
    """Reads the full horizontal level buffer ($7E:C800) into an occupancy grid.

    Layout: 16 tile columns per subscreen, 27 rows; row stride 16 within a
    `0x01B0`-byte subscreen block (same formula as `get_local_tilemap_patch`).
    """
    width_tiles = num_subscreens * 16

    def reader(tx: int, ty: int) -> int:
        subscreen = tx // 16
        addr = 0xC800 + subscreen * 0x01B0 + ty * 16 + (tx % 16)
        if addr > 0x1FFFF:
            return FREE
        return classify_tile(emu.read_wram_u8(addr), ty)

    return build_occupancy_grid(reader, width_tiles, GRID_HEIGHT_TILES)


def _traversable(cell: int, hazard_blocked: bool) -> bool:
    if cell == SOLID:
        return False
    if cell == HAZARD and hazard_blocked:
        return False
    return True


def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    hazard_cost: float = 25.0,
    climb_cost: float = 2.0,
    drop_cost: float = 0.2,
    hazard_blocked: bool = False,
) -> List[Tuple[int, int]]:
    """8-connected A* with no corner-cutting and climb-aware costs.

    Moving up (decreasing row) costs `climb_cost` per tile to approximate
    jump effort; drops are cheap; hazard cells add `hazard_cost` (or are
    blocked). Returns tile path [(tx, ty), ...] or [] if unreachable or if
    start/goal are not traversable.
    """
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return []
    if not _traversable(grid[sy, sx], hazard_blocked) or not _traversable(
        grid[gy, gx], hazard_blocked
    ):
        return []

    moves = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]
    counter = itertools.count()
    open_heap = [(0.0, next(counter), sx, sy)]
    g_score = {(sx, sy): 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def heuristic(x: int, y: int) -> float:
        dx, dy = abs(x - gx), abs(y - gy)
        return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

    while open_heap:
        _, _, x, y = heapq.heappop(open_heap)
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path.append((x, y))
            return path[::-1]
        for dx, dy, base in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            cell = int(grid[ny, nx])
            if not _traversable(cell, hazard_blocked):
                continue
            if dx != 0 and dy != 0:  # no corner cutting through solid/hazard
                c1, c2 = int(grid[y, nx]), int(grid[ny, x])
                if not _traversable(c1, hazard_blocked) or not _traversable(c2, hazard_blocked):
                    continue
            step = base + max(0, y - ny) * climb_cost + max(0, ny - y) * drop_cost
            if cell == HAZARD:
                step += hazard_cost
            tentative = g_score[(x, y)] + step
            if tentative < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)] = tentative
                came_from[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (tentative + heuristic(nx, ny), next(counter), nx, ny))
    return []


def extract_waypoints(
    path_tiles: List[Tuple[int, int]], min_gap_px: float = 48.0
) -> List[Tuple[float, float]]:
    """Compresses a tile path to pixel waypoints: direction changes + spacing.

    Tile centers are mapped to pixels ((tx + 0.5) * 16, (ty + 0.5) * 16).
    Collinear runs collapse; consecutive waypoints keep >= min_gap_px unless
    the goal itself is closer (goal is always kept).
    """
    if not path_tiles:
        return []
    pts = [((tx + 0.5) * TILE_PX, (ty + 0.5) * TILE_PX) for tx, ty in path_tiles]
    if len(pts) <= 2:
        return pts
    compressed = [pts[0]]
    prev_dir = None
    for a, b in zip(pts[:-1], pts[1:]):
        direction = (np.sign(b[0] - a[0]), np.sign(b[1] - a[1]))
        if direction != prev_dir and a != compressed[-1]:
            compressed.append(a)
        prev_dir = direction
    compressed.append(pts[-1])
    # Enforce minimum spacing (goal always kept).
    spaced = [compressed[0]]
    for p in compressed[1:-1]:
        q = spaced[-1]
        if math.hypot(p[0] - q[0], p[1] - q[1]) >= min_gap_px:
            spaced.append(p)
    spaced.append(compressed[-1])
    return spaced


class WaypointObjective(TrajectoryObjective):
    """Local MPC objective steered toward the active global waypoint."""

    def __init__(
        self,
        target_x: float = 0.0,
        target_y: float = 0.0,
        weight_target: float = 1.0,
        arrival_bonus: float = 50.0,
        arrival_radius: float = 12.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_x = target_x
        self.target_y = target_y
        self.weight_target = weight_target
        self.arrival_bonus = arrival_bonus
        self.arrival_radius = arrival_radius

    def compute_trajectory_rewards(
        self,
        initial_states: torch.Tensor,
        predicted_trajectories: torch.Tensor,
    ) -> torch.Tensor:
        base = super().compute_trajectory_rewards(initial_states, predicted_trajectories)
        final = predicted_trajectories[:, -1, :2]
        target = torch.tensor(
            [self.target_x, self.target_y],
            dtype=final.dtype,
            device=final.device,
        )
        dist = torch.linalg.norm(final - target, dim=1)
        return base - self.weight_target * dist + self.arrival_bonus * (dist < self.arrival_radius).float()


class HierarchicalMPCController:
    """Global waypoint route + local Hard-PINN CEM-MPC tracker.

    `plan(state8)` selects the first waypoint beyond `capture_radius_px`
    (advancing past reached ones) and runs one local MPC step toward it.
    """

    def __init__(
        self,
        world_model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        horizon: int = 15,
        num_candidates: int = 128,
        cem_iterations: int = 2,
        capture_radius_px: float = 24.0,
        objective: Optional[WaypointObjective] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_model = world_model or HardResidualPINNDynamics()
        self.objective = objective or WaypointObjective()
        self.mpc = ModelPredictiveController(
            world_model=self.world_model,
            device=self.device,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=cem_iterations,
            objective=self.objective,
        )
        self.capture_radius_px = capture_radius_px
        self.path: List[Tuple[float, float]] = []
        self.waypoint_index = 0

    def set_path(self, path_px: List[Tuple[float, float]]) -> None:
        self.path = list(path_px)
        self.waypoint_index = 0

    @property
    def active_waypoint(self) -> Optional[Tuple[float, float]]:
        if not self.path:
            return None
        return self.path[min(self.waypoint_index, len(self.path) - 1)]

    def plan(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """One hierarchical step. Returns (action[6], info dict)."""
        x, y = float(current_state[0]), float(current_state[1])
        while self.waypoint_index < len(self.path):
            wx, wy = self.path[self.waypoint_index]
            if math.hypot(wx - x, wy - y) > self.capture_radius_px:
                break
            self.waypoint_index += 1
        target = self.active_waypoint
        if target is not None:
            self.objective.target_x, self.objective.target_y = target
        action, info = self.mpc.plan(current_state)
        info = dict(info)
        info.update(
            {
                "active_waypoint": target,
                "waypoint_index": min(self.waypoint_index, max(0, len(self.path) - 1)),
                "num_waypoints": len(self.path),
            }
        )
        return action, info
