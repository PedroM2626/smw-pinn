"""
train_pixel_estimator.py
Supervised training of the CNN pixel-to-state estimator on paired
(frame, WRAM state) data. Reports per-variable error on denormalized states.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.per_variable_metrics import compute_per_variable_metrics
from src.perception.pixel_encoder import PixelStateEstimator
from src.perception.vision_dataset import create_vision_loaders
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def run_training(
    dataset_path: str = "data/raw/smw_pixel_dataset.npz",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    output_dir: str = "results",
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data = np.load(dataset_path)
    train_loader, val_loader, normalizer = create_vision_loaders(
        data["frames"], data["states"], batch_size=batch_size, seed=seed
    )
    logger.info(f"Frames: {data['frames'].shape} | train batches: {len(train_loader)}")

    model = PixelStateEstimator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss()

    best_val, patience, bad = float("inf"), 6, 0
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(output_dir, "checkpoints", "pixel_estimator_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for frames, states in train_loader:
            frames, states = frames.to(device), states.to(device)
            opt.zero_grad()
            loss = loss_fn(model(frames), states)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss, preds, targets = 0.0, [], []
        with torch.no_grad():
            for frames, states in val_loader:
                frames, states = frames.to(device), states.to(device)
                pred = model(frames)
                val_loss += loss_fn(pred, states).item()
                preds.append(normalizer.denormalize(pred.cpu()))
                targets.append(normalizer.denormalize(states.cpu()))
        val_loss /= max(1, len(val_loader))
        sched.step(val_loss)
        logger.info(f"Epoch {epoch}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, bad = val_loss, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "normalizer": normalizer.to_dict(),
                    "config": {"dataset_path": dataset_path, "seed": seed},
                },
                ckpt,
            )
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    per_var = compute_per_variable_metrics(torch.cat(preds), torch.cat(targets))
    metrics = {"best_val_loss": best_val, "per_variable": per_var}
    with open(os.path.join(output_dir, "pixel_estimator_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {ckpt} + pixel_estimator_metrics.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN pixel-to-state estimator.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-path", default="data/raw/smw_pixel_dataset.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parse_args_with_config(parser)
    run_training(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
