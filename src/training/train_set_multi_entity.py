"""
train_set_multi_entity.py
Supervised training of SetMultiEntityPINNDynamics on full 12-slot sprite
sets (orphan connection): predicts next Mario 8D + next entity rows for all
K slots. Entity loss is masked to slots active in the *target* frame, since
inactive rows are zero-padding, not physics.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.pinn_set_multi_entity import SetMultiEntityPINNDynamics
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def run_training(
    dataset_path: str = "data/raw/smw_set_multi_entity_dataset.npz",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    output_dir: str = "results",
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(dataset_path)
    n = len(data["mario"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    cut = max(1, int(0.85 * n))
    tr, va = perm[:cut], perm[cut:] or perm[-1:]

    def mk(idx):
        return TensorDataset(
            torch.tensor(data["mario"][idx], dtype=torch.float32),
            torch.tensor(data["entities"][idx], dtype=torch.float32),
            torch.tensor(data["actions"][idx], dtype=torch.float32),
            torch.tensor(data["next_mario"][idx], dtype=torch.float32),
            torch.tensor(data["next_entities"][idx], dtype=torch.float32),
        )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(mk(tr), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(mk(va), batch_size=batch_size, shuffle=False)

    k = int(data["entities"].shape[1])
    model = SetMultiEntityPINNDynamics(max_entities=k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best, bad = float("inf"), 0
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(output_dir, "checkpoints", "set_multi_entity_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, steps = 0.0, 0
        for m_s, ent, act, m_t, ent_t in train_loader:
            m_s, ent, act, m_t, ent_t = (b.to(device) for b in (m_s, ent, act, m_t, ent_t))
            opt.zero_grad()
            pred_m, pred_e, _ = model(m_s, ent, act)
            mask = (ent_t[..., 4:5] > 0.5).float()
            loss = loss_fn(pred_m, m_t) + loss_fn(pred_e * mask, ent_t * mask)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            steps += 1
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for m_s, ent, act, m_t, ent_t in val_loader:
                m_s, ent, act, m_t, ent_t = (b.to(device) for b in (m_s, ent, act, m_t, ent_t))
                pred_m, pred_e, _ = model(m_s, ent, act)
                mask = (ent_t[..., 4:5] > 0.5).float()
                val_loss += (loss_fn(pred_m, m_t) + loss_fn(pred_e * mask, ent_t * mask)).item()
                val_steps += 1
        val_loss /= max(1, val_steps)
        logger.info(f"Epoch {epoch}/{epochs} | train {train_loss / max(1, steps):.4f} | val {val_loss:.4f}")
        if val_loss < best:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= 4:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    metrics = {"best_val_loss": best, "max_entities": k}
    with open(os.path.join(output_dir, "set_multi_entity_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {ckpt} (val {best:.4f})")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SetMultiEntityPINN on 12-slot sets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-path", default="data/raw/smw_set_multi_entity_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
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
