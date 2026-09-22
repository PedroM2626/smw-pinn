"""
evaluate_distilled_policy_snes.py
Evaluates the amortized distilled MPC policy directly on the authentic SNES emulator.
Measures survival, total progress in pixels, Rex evasion success, and inference throughput (FPS).
"""

import json
import os
import time
from typing import Dict

import numpy as np
import torch

from src.environment.snes_emulator import SnesLibretroEmulator
from src.training.distill_mpc_policy import DistilledActorPolicy, extract_12d_vector
from src.utils.logging import get_logger
from src.utils.paths import (
    CORE_PATH,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1,
    checkpoint_file,
    results_file,
)

logger = get_logger(__name__)


def evaluate_distilled_policy(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_path: str = STATE_YOSHI_ISLAND_1,
    policy_checkpoint: str = checkpoint_file("distilled_mpc_policy.pt"),
    output_metrics: str = results_file("distilled_policy_metrics.json"),
    max_frames: int = 400,
) -> Dict:
    logger.info("====================================================================")
    logger.info("  EVALUATING DISTILLED MPC POLICY ON AUTHENTIC SNES HARDWARE        ")
    logger.info("====================================================================")

    # Load policy onto CPU to profile pure lightweight inference
    policy = DistilledActorPolicy(state_dim=12, action_dim=6)
    policy.load_state_dict(torch.load(policy_checkpoint, map_location="cpu", weights_only=True))
    policy.eval()

    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)

    with open(state_path, "rb") as f:
        initial_savestate = f.read()

    emu.load_state(initial_savestate)
    emu.enable_gameplay_mode()
    for _ in range(5):
        emu.step_frame()

    init_state = emu.get_smw_extended_state()
    start_x = init_state["x"]

    survived_frames = 0
    rex_evaded = False
    prev_b = False
    policy_infer_times = []

    t_start = time.time()

    for frame in range(max_frames):
        s_dict = emu.get_smw_extended_state()
        s_vec = extract_12d_vector(s_dict)

        # Profile pure neural network inference latency
        t_infer_0 = time.perf_counter()
        action_dict, _ = policy.predict_action(s_vec, threshold=0.5)
        t_infer_1 = time.perf_counter()
        policy_infer_times.append(t_infer_1 - t_infer_0)

        # Re-jump edge trigger handling
        if action_dict["B"] and prev_b and frame % 14 == 0:
            action_dict["B"] = False
        prev_b = action_dict["B"]

        # Track Rex evasion: Mario passes enemy with safe clearance
        dx_enemy = s_dict["delta_x_enemy"]
        is_active = s_dict["hazard_active"] > 0.5
        if is_active and dx_enemy < -10.0 and s_dict["x"] > 200.0:
            rex_evaded = True

        emu.set_input(action_dict)
        emu.step_frame()
        survived_frames += 1

        if s_dict["y"] > 450.0:
            logger.info(f"Mario fell into pit at frame {frame}")
            break

    total_time = time.time() - t_start
    final_state = emu.get_smw_extended_state()
    total_progress = float(final_state["x"] - start_x)

    emu.close()

    mean_infer_us = float(np.mean(policy_infer_times) * 1e6)
    infer_fps = float(1.0 / np.mean(policy_infer_times))

    metrics = {
        "survived_frames": survived_frames,
        "total_progress_pixels": total_progress,
        "rex_evaded": rex_evaded,
        "mean_inference_latency_us": mean_infer_us,
        "inference_throughput_fps": infer_fps,
        "total_evaluation_time_seconds": total_time,
        "hardware_framerate_fps": float(survived_frames / max(1e-5, total_time)),
    }

    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
    with open(output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("\n--- DISTILLED POLICY EVALUATION RESULTS ---")
    logger.info(f"Survived Frames:            {survived_frames} / {max_frames}")
    logger.info(f"Total Progress:             {total_progress:.2f} pixels")
    logger.info(f"Rex Evaded:                 {rex_evaded}")
    logger.info(f"Inference Latency:          {mean_infer_us:.2f} us / step")
    logger.info(f"Policy Inference Throughput: {infer_fps:.1f} FPS (vs 23.7 FPS for CEM MPC)")
    logger.info(f"Metrics saved to: {output_metrics}")

    return metrics


if __name__ == "__main__":
    evaluate_distilled_policy()
