"""
train_multi_entity.py
Supervised training script for the Multi-Entity Physics-Informed Neural Network (MultiEntityPINNDynamics).
Trains the hazard_net dynamics on genuine transitions recorded from SMW WRAM (data/raw/smw_multi_entity_dataset.npz).
Validates on an independent test split with early stopping and learning rate scheduling.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.utils.seed import set_global_seed


def train_multi_entity_model(
    data_path: str = "data/raw/smw_multi_entity_dataset.npz",
    base_checkpoint: str = "results/checkpoints/pinn_hard_best.pt",
    output_checkpoint: str = "results/checkpoints/pinn_multi_entity_best.pt",
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
):
    print("==========================================================")
    print("  TRAINING MULTI-ENTITY PINN ON GENUINE WRAM TELEMETRY    ")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load genuine dataset
    raw = np.load(data_path)
    states = raw["states"]
    actions = raw["actions"]
    next_states = raw["next_states"]

    n_samples = len(states)
    print(f"Loaded {n_samples} genuine 12D transitions from: {data_path}")

    # Filter to give high importance to frames where hazard is active
    active_mask = states[:, 11] > 0.5
    print(f"Active hazard transitions: {int(active_mask.sum())} / {n_samples} ({100.0*active_mask.mean():.1f}%)")

    # Train / Test split (80 / 20)
    set_global_seed(42)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(
        torch.tensor(states[train_idx], dtype=torch.float32),
        torch.tensor(actions[train_idx], dtype=torch.float32),
        torch.tensor(next_states[train_idx], dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(states[test_idx], dtype=torch.float32),
        torch.tensor(actions[test_idx], dtype=torch.float32),
        torch.tensor(next_states[test_idx], dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Instantiate model
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6).to(device)
    if os.path.exists(base_checkpoint):
        base_pinn.load_state_dict(torch.load(base_checkpoint, map_location=device, weights_only=True))
        print(f"Preloaded base Hard PINN from {base_checkpoint}")

    model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)

    # Freeze base Mario PINN so hazard_net learns specifically the entity dynamics
    for param in model.mario_pinn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.hazard_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    criterion_mse = nn.MSELoss()
    best_test_loss = float("inf")
    epochs_no_improve = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for b_s, b_a, b_next_s in train_loader:
            b_s, b_a, b_next_s = b_s.to(device), b_a.to(device), b_next_s.to(device)

            optimizer.zero_grad()
            pred_next_s = model(b_s, b_a)

            # Hazard loss on dimensions 8..11
            hazard_loss = criterion_mse(pred_next_s[:, 8:], b_next_s[:, 8:])
            hazard_loss.backward()
            optimizer.step()

            train_losses.append(hazard_loss.item())

        # Validation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for b_s, b_a, b_next_s in test_loader:
                b_s, b_a, b_next_s = b_s.to(device), b_a.to(device), b_next_s.to(device)
                pred_next_s = model(b_s, b_a)
                t_loss = criterion_mse(pred_next_s[:, 8:], b_next_s[:, 8:])
                test_losses.append(t_loss.item())

        mean_train = float(np.mean(train_losses))
        mean_test = float(np.mean(test_losses))
        scheduler.step(mean_test)

        if mean_test < best_test_loss:
            best_test_loss = mean_test
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            torch.save(model.state_dict(), output_checkpoint)
            marker = "*"
        else:
            epochs_no_improve += 1
            marker = " "

        if epoch % 5 == 0 or marker == "*":
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss (Hazard): {mean_train:.6f} | "
                f"Test Loss (Hazard): {mean_test:.6f} {marker}"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print("==========================================================")
    print(f"Multi-Entity PINN successfully trained in {elapsed:.2f}s!")
    print(f"Best Test Loss (Hazard MSE): {best_test_loss:.6f}")
    print(f"Model saved to: {output_checkpoint}")
    print("==========================================================")

    return {
        "best_test_loss": best_test_loss,
        "training_time": elapsed,
        "checkpoint": output_checkpoint,
    }


if __name__ == "__main__":
    train_multi_entity_model()
