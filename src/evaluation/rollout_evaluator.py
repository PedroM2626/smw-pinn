"""
rollout_evaluator.py
Avaliador rigoroso de desvio de trajetória em horizonte longo (Multi-step Autoregressive Rollout).
Mede o acúmulo de erro (drift) em relação à trajetória real do jogo ao longo de H frames.
"""

from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn


class RolloutEvaluator:
    """
    Avalia a estabilidade autorregressiva de modelos de dinâmica:
        hat_s_{t+1} = f(hat_s_t, a_t)
    e compara diretamente com a trajetória do chão de verdade (Ground Truth).
    """

    def __init__(self, device: torch.device):
        self.device = device

    @torch.no_grad()
    def evaluate_rollout(
        self,
        model: nn.Module,
        model_type: str,
        initial_state: np.ndarray,  # [state_dim]
        action_sequence: np.ndarray,  # [H, action_dim]
        ground_truth_states: np.ndarray,  # [H, state_dim]
    ) -> Dict[str, np.ndarray]:
        """
        Executa o rollout autorregressivo completo de comprimento H.

        Returns:
            dict contendo:
                predicted_trajectory: [H, state_dim]
                ground_truth_trajectory: [H, state_dim]
                euclidean_drift: [H] (distância euclidiana em pixels a cada passo)
                kinematic_violations: int
                velocity_violations: int
        """
        model.eval()
        H = len(action_sequence)
        state_dim = len(initial_state)

        pred_traj = np.zeros((H, state_dim), dtype=np.float32)
        curr_state = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Buffer para LSTM se aplicável
        if model_type.lower() == "lstm":
            hidden = None

        kin_violations = 0
        vel_violations = 0

        for t in range(H):
            curr_action = torch.tensor(
                action_sequence[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if model_type.lower() == "lstm":
                curr_input_state = curr_state.unsqueeze(1)  # [1, 1, D]
                curr_input_action = curr_action.unsqueeze(1)  # [1, 1, A]
                pred_out, hidden = model(curr_input_state, curr_input_action, hidden)
                next_state_pred = pred_out.squeeze(1)
            else:
                next_state_pred = model(curr_state, curr_action)

            # Verificar violações físicas no passo predito
            hat_vx = next_state_pred[0, 2].item()
            hat_vy = next_state_pred[0, 3].item()
            hat_x = next_state_pred[0, 0].item()
            prev_x = curr_state[0, 0].item()

            # Violação cinemática (> 0.2 pixels de desvio de dx = vx/16)
            if abs((hat_x - prev_x) - (curr_state[0, 2].item() / 16.0)) > 0.2:
                kin_violations += 1

            # Violação de limites de velocidade
            if abs(hat_vx) > 72.0 or hat_vy > 64.0:
                vel_violations += 1

            pred_traj[t] = next_state_pred[0].cpu().numpy()
            curr_state = next_state_pred  # Realimentação autorregressiva pura

        # Cálculo do desvio euclidiano no espaço 2D (X, Y) em pixels
        gt_xy = ground_truth_states[:, :2]
        pred_xy = pred_traj[:, :2]
        euclidean_drift = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))

        return {
            "predicted_trajectory": pred_traj,
            "ground_truth_trajectory": ground_truth_states,
            "euclidean_drift": euclidean_drift,
            "mean_drift": float(np.mean(euclidean_drift)),
            "final_drift": float(euclidean_drift[-1]),
            "kinematic_violations": kin_violations,
            "velocity_violations": vel_violations,
        }

    def benchmark_all_models(
        self,
        models: Dict[str, nn.Module],
        initial_state: np.ndarray,
        action_sequence: np.ndarray,
        ground_truth_states: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, model in models.items():
            model_type = "lstm" if "lstm" in name.lower() else "mlp"
            rollout_res = self.evaluate_rollout(
                model=model,
                model_type=model_type,
                initial_state=initial_state,
                action_sequence=action_sequence,
                ground_truth_states=ground_truth_states,
            )
            results[name] = {
                "mean_drift_pixels": rollout_res["mean_drift"],
                "final_drift_pixels": rollout_res["final_drift"],
                "kin_violations": rollout_res["kinematic_violations"],
                "vel_violations": rollout_res["velocity_violations"],
            }
        return results
