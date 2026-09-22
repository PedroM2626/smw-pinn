"""
evaluate_hierarchical_mpc.py
Closed-loop hierarchical control on the real console (requires emulator +
ROM; not run in CI): A* global route over the WRAM tile grid, tracked by
local Hard-PINN CEM-MPC.
"""

import os
from typing import Dict

import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models import HardResidualPINNDynamics
from src.planning.global_planner import (
    GRID_HEIGHT_TILES,
    TILE_PX,
    HierarchicalMPCController,
    WaypointObjective,
    astar,
    build_global_grid_from_emulator,
    extract_waypoints,
)
from src.planning.mpc_planner import action_vector_to_joypad
from src.planning.terminal_value import TerminalValueNet, TerminalValueObjective
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    RESULTS_DIR,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    checkpoint_file,
)
from src.utils.provenance import write_metrics
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def run_hierarchical(
    pinn_ckpt: str = checkpoint_file("pinn_hard_best.pt"),
    value_ckpt: str | None = None,
    core_path: str = CORE_PATH,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    max_frames: int = 900,
    seed: int = 42,
    output_dir: str = RESULTS_DIR,
) -> Dict:
    """Set `value_ckpt` to a terminal_value checkpoint for TD-MPC mode."""
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pinn = HardResidualPINNDynamics().to(device)
    pinn.load_state_dict(torch.load(pinn_ckpt, map_location=device, weights_only=True))
    if value_ckpt is not None:
        payload = torch.load(value_ckpt, map_location=device, weights_only=False)
        value_net = TerminalValueNet().to(device)
        value_net.load_state_dict(payload["model"])
        objective = TerminalValueObjective(
            value_net=value_net,
            target_mean=payload["stats"]["target_mean"],
            target_std=payload["stats"]["target_std"],
            gamma=payload.get("gamma", 0.99),
            weight_progress=1.0,
        )
        logger.info(f"TD-MPC mode: terminal value from {value_ckpt}.")
    else:
        objective = WaypointObjective(weight_progress=1.0)
    controller = HierarchicalMPCController(world_model=pinn, device=device, objective=objective)

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()
    start_state = emu.start_episode(initial_savestate)

    grid = build_global_grid_from_emulator(emu)
    s0 = start_state  # post warm-up WRAM, already settled by start_episode
    start = (int(s0["x"]) // TILE_PX, int(s0["y"]) // TILE_PX)
    # Goal: rightmost column, first free cell above ground.
    gx = grid.shape[1] - 1
    gy = next(
        (ty - 1 for ty in range(GRID_HEIGHT_TILES - 1, -1, -1) if grid[ty, gx] == 1),
        GRID_HEIGHT_TILES // 2,
    )
    path = astar(grid, start, (gx, max(0, gy)))
    waypoints = extract_waypoints(path)
    controller.set_path(waypoints)
    logger.info(f"Global route: {len(path)} tiles -> {len(waypoints)} waypoints.")

    traj_x = [s0["x"]]
    survived = 0
    for _ in range(max_frames):
        st = emu.get_smw_state()
        s8 = np.array(
            [
                st["x"],
                st["y"],
                st["vx"],
                st["vy"],
                st["c_ground"],
                st["c_ceiling"],
                st["c_left"],
                st["c_right"],
            ],
            dtype=np.float32,
        )
        action, _ = controller.plan(s8)
        emu.set_input(action_vector_to_joypad(action))
        emu.step_frame()
        st = emu.get_smw_state()
        traj_x.append(st["x"])
        survived += 1
        if st["y"] > 500:
            break
    emu.close()

    metrics = {
        "survived_frames": survived,
        "progress_px": float(traj_x[-1] - traj_x[0]),
        "num_tiles_routed": len(path),
        "num_waypoints": len(waypoints),
        "waypoints_reached": controller.waypoint_index,
        "controller": "A* global + Hard PINN local MPC",
    }
    os.makedirs(output_dir, exist_ok=True)
    write_metrics(
        os.path.join(output_dir, "hierarchical_mpc_metrics.json"),
        metrics,
        seed=seed,
        command="python -m src.evaluation.evaluate_hierarchical_mpc",
    )
    logger.info(f"Hierarchical MPC: {metrics}")
    return metrics


if __name__ == "__main__":
    run_hierarchical()
