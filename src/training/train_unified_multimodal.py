"""
train_unified_multimodal.py
Joint end-to-end training of the Unified Multimodal PINN (orphan connection).

Alternates batches from the two genuine datasets each optimizer step:
  - tilemap batches: kinematics + 7x7 patch -> next_mario + contact logits
    (BCE against WRAM contact flags);
  - multi-entity batches: Mario 8D + hazard 4D (patch=None) -> next_mario +
    next_hazard (MSE masked to active hazards; inactive rows carry sentinels).

No synthetic data: every sample is genuine WRAM telemetry.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.pinn_unified_multimodal import UnifiedMultimodalPINNDynamics
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_MULTI_ENTITY,
    DATASET_TILEMAP,
    RESULTS_DIR,
)
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def _tile_loaders(path: str, batch_size: int, seed: int):
    data = np.load(path)
    n = len(data["states"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    cut = max(1, int(0.85 * n))
    tr, va = perm[:cut], perm[cut:] or perm[-1:]

    def mk_tile(idx):
        return TensorDataset(
            torch.tensor(data["states"][idx], dtype=torch.float32),
            torch.tensor(data["actions"][idx], dtype=torch.float32),
            torch.tensor(data["tile_patches"][idx], dtype=torch.long),
            torch.tensor(data["next_states"][idx], dtype=torch.float32),
        )

    g = torch.Generator()
    g.manual_seed(seed)
    train = DataLoader(mk_tile(tr), batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(mk_tile(va), batch_size=batch_size, shuffle=False)
    return train, val


def _multi_loaders(path: str, batch_size: int, seed: int):
    data = np.load(path)
    n = len(data["states"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    cut = max(1, int(0.85 * n))
    tr, va = perm[:cut], perm[cut:] or perm[-1:]

    def mk_multi(idx):
        return TensorDataset(
            torch.tensor(data["states"][idx], dtype=torch.float32),
            torch.tensor(data["actions"][idx], dtype=torch.float32),
            torch.tensor(data["next_states"][idx], dtype=torch.float32),
        )

    g = torch.Generator()
    g.manual_seed(seed + 1)
    train = DataLoader(mk_multi(tr), batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(mk_multi(va), batch_size=batch_size, shuffle=False)
    return train, val


def run_training(
    tilemap_path: str = DATASET_TILEMAP,
    multi_path: str = DATASET_MULTI_ENTITY,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    output_dir: str = RESULTS_DIR,
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_train, tile_val = _tile_loaders(tilemap_path, batch_size, seed)
    multi_train, multi_val = _multi_loaders(multi_path, batch_size, seed)

    model = UnifiedMultimodalPINNDynamics().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    reg_loss, bce_loss = nn.SmoothL1Loss(), nn.BCEWithLogitsLoss()

    def tile_step(batch, train: bool):
        s, a, p, t = [b.to(device) for b in batch]
        out = model(s, a, tilemap_patch=p)
        loss = reg_loss(out["next_mario"], t) + 0.5 * bce_loss(out["contact_logits"], t[:, 4:8])
        return loss, out

    def multi_step(batch, train: bool):
        s12, a, t12 = [b.to(device) for b in batch]
        out = model(s12[:, :8], a, hazard_4d=s12[:, 8:12])
        active = (s12[:, 11:12] > 0.5).float()
        loss = reg_loss(out["next_mario"], t12[:, :8]) + reg_loss(
            out["next_hazard"] * active, t12[:, 8:12] * active
        )
        return loss, out

    best, bad = float("inf"), 0
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(output_dir, "checkpoints", "unified_joint_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for b_tile, b_multi in zip(tile_train, multi_train):
            for step_fn, batch in ((tile_step, b_tile), (multi_step, b_multi)):
                opt.zero_grad()
                loss, _ = step_fn(batch, True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
                steps += 1
        model.eval()
        val_total, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in tile_val:
                loss, _ = tile_step(batch, False)
                val_total += loss.item()
                val_steps += 1
            for batch in multi_val:
                loss, _ = multi_step(batch, False)
                val_total += loss.item()
                val_steps += 1
        val = val_total / max(1, val_steps)
        logger.info(f"Epoch {epoch}/{epochs} | train {total / max(1, steps):.4f} | val {val:.4f}")
        if val < best:
            best, bad = val, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= 4:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    metrics = {"best_val_loss": best, "epochs_run": epoch}
    with open(os.path.join(output_dir, "unified_joint_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {ckpt} (val {best:.4f})")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint training of Unified Multimodal PINN.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tilemap-path", default=DATASET_TILEMAP)
    parser.add_argument("--multi-path", default=DATASET_MULTI_ENTITY)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parse_args_with_config(parser)
    run_training(
        tilemap_path=args.tilemap_path,
        multi_path=args.multi_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
