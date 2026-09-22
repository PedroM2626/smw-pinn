"""
evaluate_pixel_mpc.py
Closed-loop pixel-to-action control on the real SNES console (requires
emulator + ROM + checkpoints; not run in CI):

    pixels -> PixelStateEstimator -> Hard PINN MPC -> joypad

Privileged WRAM truth is read in parallel *only* to measure estimator error
and planning drift — the controller itself never sees it.
"""

import json
import os
from typing import Dict

import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models import HardResidualPINNDynamics
from src.perception.pixel_encoder import PixelStateEstimator, StateNormalizer, preprocess_frame
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

BUTTONS = ["B", "Y", "UP", "DOWN", "LEFT", "RIGHT"]


def action_vector_to_dict(vec: np.ndarray) -> Dict[str, bool]:
    return {b: bool(vec[i] > 0.5) for i, b in enumerate(BUTTONS)}


def run_pixel_mpc(
    estimator_ckpt: str = "results/checkpoints/pixel_estimator_best.pt",
    pinn_ckpt: str = "results/checkpoints/pinn_hard_best.pt",
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    rom_path: str = "data/raw/smw_usa.sfc",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    max_frames: int = 400,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    est_payload = torch.load(estimator_ckpt, map_location=device, weights_only=False)
    estimator = PixelStateEstimator().to(device)
    estimator.load_state_dict(est_payload["model"])
    estimator.eval()
    normalizer = StateNormalizer.from_dict(est_payload["normalizer"])

    pinn = HardResidualPINNDynamics().to(device)
    pinn.load_state_dict(torch.load(pinn_ckpt, map_location=device, weights_only=True))
    mpc = ModelPredictiveController(
        world_model=pinn, device=device, horizon=15, num_candidates=256,
        cem_iterations=3, objective=TrajectoryObjective(),
    )

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        emu.load_state(f.read())
    emu.enable_frame_capture(True)

    traj_x, est_errors = [], []
    x0 = emu.get_smw_state()["x"]
    survived = 0
    for frame in range(max_frames):
        pixels = emu.get_frame()
        truth = emu.get_smw_state()
        if pixels is None:
            emu.set_input({})
            emu.step_frame()
            continue
        with torch.no_grad():
            s_hat = normalizer.denormalize(
                estimator(preprocess_frame(pixels).to(device)).cpu()
            )[0].numpy().astype(np.float32)
        truth_vec = np.array(
            [truth["x"], truth["y"], truth["vx"], truth["vy"],
             truth["c_ground"], truth["c_ceiling"], truth["c_left"], truth["c_right"]],
            dtype=np.float32,
        )
        est_errors.append(float(np.mean(np.abs(s_hat[:4] - truth_vec[:4]))))
        action, _ = mpc.plan(s_hat)
        emu.set_input(action_vector_to_dict(action))
        emu.step_frame()
        traj_x.append(truth["x"])
        survived += 1
        if truth["y"] > 500:
            break
    emu.close()

    metrics = {
        "survived_frames": survived,
        "progress_px": float(traj_x[-1] - x0) if traj_x else 0.0,
        "mean_estimator_mae_xyv": float(np.mean(est_errors)) if est_errors else float("nan"),
        "controller": "pixel-estimator + Hard PINN MPC (blind to WRAM)",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pixel_mpc_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Pixel-MPC: {metrics}")
    return metrics


if __name__ == "__main__":
    run_pixel_mpc()
