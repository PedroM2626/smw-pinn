"""
evaluate_tilemap_mpc.py
Closed-loop Tilemap-PINN MPC on real hardware (requires emulator + ROM):
each frame refreshes the 7x7 WRAM patch, then plans with terrain anticipation.
No contact-flag reflexes are used — geometry comes from the tile buffer.
"""

import json
import os
import time
from typing import Dict

import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models.tilemap_pinn import TilemapPINNDynamics
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.planning.tilemap_mpc import TilemapMPCWrapper
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

BUTTONS = ["B", "Y", "UP", "DOWN", "LEFT", "RIGHT"]


def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {b: bool(vec[i] > 0.5) for i, b in enumerate(BUTTONS)}


def run_tilemap_mpc(
    tilemap_ckpt: str = "results/checkpoints/tilemap_pinn_best.pt",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    rom_path: str = "data/raw/smw_usa.sfc",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    max_frames: int = 400,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tile_model = TilemapPINNDynamics().to(device)
    tile_model.load_state_dict(torch.load(tilemap_ckpt, map_location=device, weights_only=True))
    tile_model.eval()
    world_model = TilemapMPCWrapper(tile_model).to(device)
    controller = ModelPredictiveController(
        world_model=world_model,
        device=device,
        horizon=16,
        num_candidates=256,
        cem_iterations=3,
        objective=TrajectoryObjective(weight_progress=3.0, weight_velocity=0.5),
    )

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        emu.load_state(f.read())
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()
    start_x = emu.get_smw_state()["x"]

    traj, survived, t0 = [], 0, time.time()
    termination = "timeout"
    for _ in range(max_frames):
        st = emu.get_smw_state()
        s8 = np.array(
            [st["x"], st["y"], st["vx"], st["vy"], st["c_ground"],
             st["c_ceiling"], st["c_left"], st["c_right"]], dtype=np.float32,
        )
        world_model.set_patch(emu.get_local_tilemap_patch(st["x"], st["y"], radius=3))
        action, _ = controller.plan(s8)
        emu.set_input(action_vector_to_dict(action))
        emu.step_frame()
        traj.append(st["x"])
        survived += 1
        if st["y"] > 500:
            termination = "pit_fall"
            break
    emu.close()

    metrics = {
        "survived_frames": survived,
        "progress_px": float(traj[-1] - start_x) if traj else 0.0,
        "termination": termination,
        "wall_time_s": time.time() - t0,
        "controller": "Tilemap-PINN MPC (no reflexes)",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tilemap_mpc_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Tilemap-MPC: {metrics}")
    return metrics


if __name__ == "__main__":
    run_tilemap_mpc()
