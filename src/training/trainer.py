"""
trainer.py
Unified training and optimization engine for all benchmark models:
Statistical MLP, Temporal LSTM, Soft PINN, and Hard Residual PINN.
"""

import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.physics_losses import CompositePINNLoss
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DynamicsTrainer:
    """
    Trainer supporting pure supervised data losses as well as physics-informed (PINN) losses,
    early stopping, learning rate scheduling, and fine-grained metric tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,  # 'mlp', 'lstm', 'pinn_soft', 'pinn_hard'
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_fn: Optional[nn.Module] = None,
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.model_type = model_type.lower()
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Assign default loss if not explicitly provided
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif "pinn_soft" in self.model_type:
            self.loss_fn = CompositePINNLoss(
                lambda_kin=1.0,
                lambda_bound=0.5,
                lambda_contact=0.5,
            )
        else:
            self.loss_fn = nn.SmoothL1Loss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        data_loss_sum = 0.0
        kin_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            if "lstm" in self.model_type:
                state_seq, action_seq, target_next = [b.to(self.device) for b in batch]
                pred_seq, _ = self.model(state_seq, action_seq)
                pred_next = pred_seq[:, -1, :]
                curr_state = state_seq[:, -1, :]
                curr_action = action_seq[:, -1, :]
            else:
                curr_state, curr_action, target_next = [b.to(self.device) for b in batch]
                pred_next = self.model(curr_state, curr_action)

            self.optimizer.zero_grad()

            if isinstance(self.loss_fn, CompositePINNLoss):
                loss, metrics = self.loss_fn(curr_state, curr_action, pred_next, target_next)
                data_loss_sum += metrics.get("loss_data", 0.0)
                kin_loss_sum += metrics.get("loss_kinematics", 0.0)
            else:
                loss = self.loss_fn(pred_next, target_next)
                data_loss_sum += loss.item()
                # Compute kinematic metric for unbiased comparative monitoring
                with torch.no_grad():
                    dx_pred = pred_next[:, 0] - curr_state[:, 0]
                    dx_exp = curr_state[:, 2] / 16.0
                    kin_res = torch.mean((dx_pred - dx_exp) ** 2).item()
                    kin_loss_sum += kin_res

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            "loss_total": total_loss / max(1, num_batches),
            "loss_data": data_loss_sum / max(1, num_batches),
            "loss_kinematics": kin_loss_sum / max(1, num_batches),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        data_loss_sum = 0.0
        kin_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            if "lstm" in self.model_type:
                state_seq, action_seq, target_next = [b.to(self.device) for b in batch]
                pred_seq, _ = self.model(state_seq, action_seq)
                pred_next = pred_seq[:, -1, :]
                curr_state = state_seq[:, -1, :]
                curr_action = action_seq[:, -1, :]
            else:
                curr_state, curr_action, target_next = [b.to(self.device) for b in batch]
                pred_next = self.model(curr_state, curr_action)

            if isinstance(self.loss_fn, CompositePINNLoss):
                loss, metrics = self.loss_fn(curr_state, curr_action, pred_next, target_next)
                data_loss_sum += metrics.get("loss_data", 0.0)
                kin_loss_sum += metrics.get("loss_kinematics", 0.0)
            else:
                loss = self.loss_fn(pred_next, target_next)
                data_loss_sum += loss.item()
                dx_pred = pred_next[:, 0] - curr_state[:, 0]
                dx_exp = curr_state[:, 2] / 16.0
                kin_res = torch.mean((dx_pred - dx_exp) ** 2).item()
                kin_loss_sum += kin_res

            total_loss += loss.item()
            num_batches += 1

        return {
            "val_loss_total": total_loss / max(1, num_batches),
            "val_loss_data": data_loss_sum / max(1, num_batches),
            "val_loss_kinematics": kin_loss_sum / max(1, num_batches),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
        experiment: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_kin": [],
            "val_loss": [],
            "val_kin": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, f"{self.model_type}_best.pt")

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_total = val_metrics["val_loss_data"]
            self.scheduler.step(val_total)

            history["train_loss"].append(train_metrics["loss_total"])
            history["train_kin"].append(train_metrics["loss_kinematics"])
            history["val_loss"].append(val_metrics["val_loss_data"])
            history["val_kin"].append(val_metrics["val_loss_kinematics"])

            if experiment is not None:
                try:
                    experiment.log_metrics(
                        {
                            f"{self.model_type}/train_loss": train_metrics["loss_total"],
                            f"{self.model_type}/train_kin": train_metrics["loss_kinematics"],
                            f"{self.model_type}/val_loss": val_metrics["val_loss_data"],
                            f"{self.model_type}/val_kin": val_metrics["val_loss_kinematics"],
                            f"{self.model_type}/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=epoch,
                    )
                except Exception:
                    pass

            if val_total < best_val_loss:
                best_val_loss = val_total
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                logger.info(
                    f"Epoch {epoch:3d}/{epochs:3d} | "
                    f"Train Loss: {train_metrics['loss_total']:.4f} (Kin: {train_metrics['loss_kinematics']:.4f}) | "
                    f"Val Loss: {val_metrics['val_loss_data']:.4f} (Kin: {val_metrics['val_loss_kinematics']:.4f}) | "
                    f"Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Restore best model checkpoint
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))

        elapsed = time.time() - t0
        if verbose:
            logger.info(f"Training completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.4f}")

        return history
