"""
train_terminal_value.py
Fits the TD-MPC terminal value V([x, y, vx, vy]) by Monte-Carlo regression on
genuine full-level hardware returns (no new environment interaction needed).
"""

import argparse
import json
import os

import numpy as np
import torch

from src.planning.terminal_value import compute_mc_returns, fit_terminal_value
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import (
    RESULTS_DIR,
    results_file,
)

logger = get_logger(__name__)


def run_training(
    trajectory_log: str = results_file("full_level_trajectory_log.json"),
    gamma: float = 0.99,
    hidden_dim: int = 64,
    epochs: int = 200,
    output_dir: str = RESULTS_DIR,
):
    with open(trajectory_log, encoding="utf-8") as f:
        log = json.load(f)
    states = np.stack([log["x"], log["y"], log["vx"], log["vy"]], axis=1).astype(np.float32)
    returns = compute_mc_returns(np.asarray(log["x"], dtype=np.float64), gamma=gamma)
    net, stats = fit_terminal_value(states, returns, hidden_dim=hidden_dim, epochs=epochs)

    with torch.no_grad():
        pred = net(torch.tensor(states)).numpy() * stats["target_std"] + stats["target_mean"]
    ss_res = float(np.sum((returns - pred) ** 2))
    ss_tot = float(np.sum((returns - returns.mean()) ** 2))
    r2 = 1.0 - ss_res / max(1e-12, ss_tot)

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(output_dir, "checkpoints", "terminal_value_best.pt")
    torch.save({"model": net.state_dict(), "stats": stats, "gamma": gamma}, ckpt)
    metrics = {"r2_on_training_log": r2, "mean_return": float(returns.mean()), **stats}
    with open(os.path.join(output_dir, "terminal_value_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Terminal value R2={r2:.4f} -> {ckpt}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit TD-MPC terminal value on hardware log.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--trajectory-log", default=results_file("full_level_trajectory_log.json"))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parse_args_with_config(parser)
    run_training(
        trajectory_log=args.trajectory_log,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )
