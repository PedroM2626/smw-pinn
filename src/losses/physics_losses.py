"""
physics_losses.py
Módulo contendo as funções de perda informadas pela física (PINN)
para sistemas dinâmicos discretos inspirados no Super Mario World.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteKinematicsLoss(nn.Module):
    """
    Penaliza violações da conservação cinemática euleriana de tempo discreto:
        R_x = (X_{t+1} - X_t) - (v_{x,t} / 16.0)
        R_y = (Y_{t+1} - Y_t) - (v_{y,t} / 16.0)
    Em Super Mario World, 16 subpixels equivalem a 1 pixel de deslocamento por frame.
    """

    def __init__(self, subpixels_per_pixel: float = 16.0):
        super().__init__()
        self.subpixels_per_pixel = subpixels_per_pixel

    def forward(
        self,
        current_state: torch.Tensor,
        predicted_next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_state: [B, D] contendo [X_t, Y_t, vx_t, vy_t, ...]
            predicted_next_state: [B, D] contendo [hat_X_{t+1}, hat_Y_{t+1}, hat_vx_{t+1}, hat_vy_{t+1}, ...]

        Returns:
            loss_kinematics: escalar (MSE do resíduo cinemático)
        """
        x_t = current_state[:, 0]
        y_t = current_state[:, 1]
        vx_t = current_state[:, 2]
        vy_t = current_state[:, 3]

        hat_x_next = predicted_next_state[:, 0]
        hat_y_next = predicted_next_state[:, 1]

        expected_dx = vx_t / self.subpixels_per_pixel
        expected_dy = vy_t / self.subpixels_per_pixel

        actual_dx = hat_x_next - x_t
        actual_dy = hat_y_next - y_t

        res_x = actual_dx - expected_dx
        res_y = actual_dy - expected_dy

        return torch.mean(res_x**2 + res_y**2)


class VelocityBoundsLoss(nn.Module):
    """
    Penaliza velocidades que excedem os limites físicos estruturais do jogo:
        |v_x| <= max_vx (72 subpixels/frame = 4.5 pixels/frame)
        v_y <= terminal_vy (64 subpixels/frame = 4.0 pixels/frame queda máxima)
        v_y >= min_vy (-80 subpixels/frame = impulso de salto máximo)
    """

    def __init__(
        self,
        max_vx: float = 72.0,
        terminal_vy: float = 64.0,
        min_vy: float = -80.0,
    ):
        super().__init__()
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.min_vy = min_vy

    def forward(self, predicted_next_state: torch.Tensor) -> torch.Tensor:
        hat_vx = predicted_next_state[:, 2]
        hat_vy = predicted_next_state[:, 3]

        excess_vx = F.relu(torch.abs(hat_vx) - self.max_vx)
        excess_vy_fall = F.relu(hat_vy - self.terminal_vy)
        excess_vy_rise = F.relu(self.min_vy - hat_vy)

        loss_bounds = torch.mean(excess_vx**2 + excess_vy_fall**2 + excess_vy_rise**2)
        return loss_bounds


class GroundContactConsistencyLoss(nn.Module):
    """
    Penaliza velocidade vertical descendente quando o Mario está em solo sólido
    sem intenção de salto.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        predicted_next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        current_state[:, 4]: ground_contact_flag (1 se no chão, 0 se no ar)
        action[:, 0]: jump_button_flag (1 se botão de salto ativo)
        """
        ground_contact = current_state[:, 4]
        jump_action = action[:, 0]
        hat_vy_next = predicted_next_state[:, 3]

        # Condição de repouso no chão: grounded == 1 e sem salto
        stationary_ground_mask = (ground_contact > 0.5) & (jump_action < 0.5)

        if stationary_ground_mask.any():
            ground_vy = hat_vy_next[stationary_ground_mask]
            # No solo, vy deve ser exatamente 0 (não deve afundar nem flutuar)
            return torch.mean(ground_vy**2)
        return torch.tensor(0.0, device=current_state.device)


class CompositePINNLoss(nn.Module):
    """
    Função de perda total para PINN discreta:
        L_total = L_data + lambda_kin * L_kin + lambda_bound * L_bound + lambda_contact * L_contact
    """

    def __init__(
        self,
        lambda_kin: float = 1.0,
        lambda_bound: float = 0.5,
        lambda_contact: float = 0.5,
        use_huber: bool = True,
    ):
        super().__init__()
        self.lambda_kin = lambda_kin
        self.lambda_bound = lambda_bound
        self.lambda_contact = lambda_contact

        self.kin_loss_fn = DiscreteKinematicsLoss()
        self.bound_loss_fn = VelocityBoundsLoss()
        self.contact_loss_fn = GroundContactConsistencyLoss()

        self.data_loss_fn = nn.SmoothL1Loss() if use_huber else nn.MSELoss()

    def forward(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        predicted_next_state: torch.Tensor,
        target_next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Perda de dados empíricos supervisionados
        loss_data = self.data_loss_fn(predicted_next_state, target_next_state)

        # 2. Perda cinemática euleriana
        loss_kin = self.kin_loss_fn(current_state, predicted_next_state)

        # 3. Limites de velocidade físicos
        loss_bound = self.bound_loss_fn(predicted_next_state)

        # 4. Consistência de contato
        loss_contact = self.contact_loss_fn(current_state, action, predicted_next_state)

        total_loss = (
            loss_data
            + self.lambda_kin * loss_kin
            + self.lambda_bound * loss_bound
            + self.lambda_contact * loss_contact
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_kinematics": loss_kin.item(),
            "loss_bounds": loss_bound.item(),
            "loss_contact": loss_contact.item(),
        }

        return total_loss, metrics
