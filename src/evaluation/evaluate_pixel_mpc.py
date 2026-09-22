"""
evaluate_pixel_mpc.py
Closed-loop pixel-to-action control on the real SNES console (requires
emulator + ROM + checkpoints; not run in CI):

    pixels -> PixelStateEstimator -> Hard PINN MPC -> joypad

Privileged WRAM truth is read in parallel *only* to measure estimator error
and planning drift — the controller itself never sees it.
"""

import os
from typing import Dict

import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.models import HardResidualPINNDynamics
from src.perception.pixel_encoder import PixelStateEstimator, StateNormalizer, preprocess_frame
from src.planning.mpc_planner import (
    ModelPredictiveController,
    TrajectoryObjective,
    action_vector_to_joypad,
)
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


def run_pixel_mpc(
    estimator_ckpt: str = checkpoint_file("pixel_estimator_best.pt"),
    pinn_ckpt: str = checkpoint_file("pinn_hard_best.pt"),
    core_path: str = CORE_PATH,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    max_frames: int = 400,
    seed: int = 42,
    output_dir: str = RESULTS_DIR,
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
        world_model=pinn,
        device=device,
        horizon=15,
        num_candidates=256,
        cem_iterations=3,
        objective=TrajectoryObjective(),
    )

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        initial_savestate = f.read()
    emu.enable_frame_capture(True)
    start = emu.start_episode(initial_savestate)

    traj_x, est_errors = [], []
    x0 = start["x"]
    survived = 0
    for frame in range(max_frames):
        pixels = emu.get_frame()
        truth = emu.get_smw_state()
        if pixels is None:
            emu.set_input({})
            emu.step_frame()
            continue
        with torch.no_grad():
            s_hat = (
                normalizer.denormalize(estimator(preprocess_frame(pixels).to(device)).cpu())[0]
                .numpy()
                .astype(np.float32)
            )
        truth_vec = np.array(
            [
                truth["x"],
                truth["y"],
                truth["vx"],
                truth["vy"],
                truth["c_ground"],
                truth["c_ceiling"],
                truth["c_left"],
                truth["c_right"],
            ],
            dtype=np.float32,
        )
        est_errors.append(float(np.mean(np.abs(s_hat[:4] - truth_vec[:4]))))
        action, _ = mpc.plan(s_hat)
        emu.set_input(action_vector_to_joypad(action))
        emu.step_frame()
        traj_x.append(truth["x"])
        survived += 1
        if truth["y"] > 500:
            break
    emu.close()

    metrics = {
        "survived_frames": survived,
        "progress_px": float(traj_x[-1] - x0) if traj_x else 0.0,
        "mean_estimator_mae_xyv": float(np.mean(est_errors)) if est_errors else None,
        "controller": "pixel-estimator + Hard PINN MPC (blind to WRAM)",
    }
    path = os.path.join(output_dir, "pixel_mpc_metrics.json")
    write_metrics(
        path,
        metrics,
        seed=seed,
        command="python -m src.evaluation.evaluate_pixel_mpc",
    )
    logger.info(f"Pixel-MPC: {metrics}")
    return metrics


if __name__ == "__main__":
    run_pixel_mpc()
