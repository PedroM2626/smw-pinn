"""
pinn_ensemble.py
Deep Ensemble of Physics-Informed Neural Networks (PIML / MBRL).

Implements:
1. Deep Ensemble of E=5 Hard Residual PINNs with distinct seed initializations and bootstrap data splits.
2. Epistemic uncertainty quantification via predictive model variance:
       mu(s_{t+1}) = 1/E * sum_e f_{theta_e}(s_t, a_t)
       sigma^2(s_{t+1}) = 1/E * sum_e ||f_{theta_e}(s_t, a_t) - mu(s_{t+1})||^2
3. Pessimistic reward penalization for safe trajectory planning:
       r_safe(s, a) = r(s, a) - beta * sigma(s, a)
4. Out-of-Distribution (OOD) dynamics detection to eliminate model exploitation.
"""

from typing import Dict, List, Optional, Tuple
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.environment.dataset_loader import load_and_preprocess_data, create_dataloaders
from src.utils.seed import set_global_seed


class DeepPINNEnsemble(nn.Module):
    """
    Ensemble of E Hard Residual PINN dynamics models for epistemic uncertainty estimation.
    """

    def __init__(
        self,
        num_models: int = 5,
        state_dim: int = 8,
        action_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 128],
    ):
        super().__init__()
        self.num_models = num_models
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.members = nn.ModuleList(
            [
                HardResidualPINNDynamics(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                )
                for _ in range(num_models)
            ]
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes ensemble forward pass, returning:
        - mean_pred: [B, state_dim]
        - variance: [B, state_dim]
        - uncertainty_norm: [B] (L2 norm of epistemic standard deviation)
        """
        # Collect predictions from all E members: [E, B, state_dim]
        all_preds = torch.stack([member(state, action) for member in self.members], dim=0)

        # Predictive mean across ensemble
        mean_pred = torch.mean(all_preds, dim=0)  # [B, state_dim]

        # Epistemic variance across ensemble (disagreement)
        variance = torch.var(all_preds, dim=0, unbiased=True)  # [B, state_dim]

        # Scalar epistemic uncertainty magnitude per sample
        uncertainty_norm = torch.sqrt(torch.sum(variance, dim=-1) + 1e-8)  # [B]

        return mean_pred, variance, uncertainty_norm

    def predict_with_pessimism(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        raw_reward: torch.Tensor,
        beta: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies pessimistic MBRL penalization:
            r_safe = r_raw - beta * uncertainty
        """
        mean_pred, _, uncertainty = self.forward(state, action)
        safe_reward = raw_reward - beta * uncertainty
        return mean_pred, safe_reward


def train_pinn_ensemble(
    num_models: int = 5,
    num_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    output_dir: str = "results",
) -> DeepPINNEnsemble:
    """
    Trains all E ensemble members using bootstrap partitions and distinct random seeds.
    """
    print("====================================================================")
    print(f"  TRAINING DEEP PINN ENSEMBLE (E = {num_models} MEMBERS)              ")
    print("====================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ensemble Compute Device: {device}")

    ensemble = DeepPINNEnsemble(num_models=num_models).to(device)
    ckpt_dir = os.path.join(output_dir, "checkpoints_ensemble")
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.MSELoss()
    t0 = time.time()

    for m_idx, member in enumerate(ensemble.members):
        seed = 42 + m_idx * 17
        set_global_seed(seed)

        data = load_and_preprocess_data(seed=seed)
        train_loader, val_loader, _ = create_dataloaders(data, batch_size=batch_size)

        optimizer = torch.optim.AdamW(member.parameters(), lr=lr, weight_decay=1e-4)

        member.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            n_batches = 0
            for s, a, ns in train_loader:
                s, a, ns = s.to(device), a.to(device), ns.to(device)
                optimizer.zero_grad()
                pred = member(s, a)
                loss = criterion(pred, ns)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        # Validation
        member.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for s, a, ns in val_loader:
                s, a, ns = s.to(device), a.to(device), ns.to(device)
                val_loss += criterion(member(s, a), ns).item()
                n_val += 1
        mean_val = val_loss / n_val if n_val else 0.0

        member_path = os.path.join(ckpt_dir, f"pinn_member_{m_idx}.pt")
        torch.save(member.state_dict(), member_path)
        print(f"  Member {m_idx + 1}/{num_models} trained (Seed {seed}) | Val MSE: {mean_val:.4f} | Saved: {member_path}")

    elapsed = time.time() - t0
    print(f"\nAll {num_models} ensemble members trained successfully in {elapsed:.1f}s!")

    # Epistemic Uncertainty & OOD Detection Benchmark
    print("\nEvaluating Epistemic Uncertainty Quantification (In-Distribution vs. OOD)...")
    test_data = load_and_preprocess_data(seed=42)
    test_s = torch.tensor(test_data["test_states"][:500], dtype=torch.float32, device=device)
    test_a = torch.tensor(test_data["test_actions"][:500], dtype=torch.float32, device=device)

    # In-distribution evaluation
    with torch.no_grad():
        _, var_id, unc_id = ensemble(test_s, test_a)

    # Out-of-distribution perturbation (extreme non-physical velocities and perturbed inputs)
    ood_s = test_s.clone()
    ood_s[:, 2] += torch.randn_like(ood_s[:, 2]) * 80.0  # extreme velocity jitter
    ood_s[:, 3] += torch.randn_like(ood_s[:, 3]) * 100.0
    with torch.no_grad():
        _, var_ood, unc_ood = ensemble(ood_s, test_a)

    mean_unc_id = float(unc_id.mean().item())
    mean_unc_ood = float(unc_ood.mean().item())
    ood_ratio = mean_unc_ood / (mean_unc_id + 1e-8)

    print(f"  In-Distribution Mean Epistemic Uncertainty (sigma):  {mean_unc_id:.4f}")
    print(f"  Out-of-Distribution Mean Epistemic Uncertainty (sigma): {mean_unc_ood:.4f} ({ood_ratio:.1f}x higher)")

    metrics = {
        "num_models": num_models,
        "training_time_seconds": elapsed,
        "in_distribution_uncertainty": mean_unc_id,
        "out_of_distribution_uncertainty": mean_unc_ood,
        "ood_uncertainty_ratio": ood_ratio,
    }

    import json
    metrics_path = os.path.join(output_dir, "pinn_ensemble_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Ensemble metrics saved to: {metrics_path}")

    # Plot Epistemic Uncertainty Distribution
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(unc_id.cpu().numpy(), bins=30, alpha=0.7, color="#2ecc71", label=f"In-Distribution (μ={mean_unc_id:.2f})")
    ax.hist(unc_ood.cpu().numpy(), bins=30, alpha=0.7, color="#e74c3c", label=f"Out-of-Distribution (μ={mean_unc_ood:.2f})")
    ax.set_title("Deep PINN Ensemble: Epistemic Uncertainty Disagreement (In-Dist vs. OOD)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epistemic Uncertainty Magnitude σ(s, a)")
    ax.set_ylabel("Transition Count")
    ax.legend(loc="upper right")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "figures", "pinn_ensemble_uncertainty.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Uncertainty plot saved to: {fig_path}")

    return ensemble


if __name__ == "__main__":
    train_pinn_ensemble(num_models=5, num_epochs=10)

