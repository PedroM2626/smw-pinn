"""
pinn_set_multi_entity.py
Permutation-Invariant Set-Based Multi-Entity Physics-Informed Neural Network (Set-Multi-Entity PINN).
Models variable number of dynamic stage entities/sprites (K in [0, K_max])
interacting with Mario using Cross-Attention and Deep Sets with exact kinematic consistency:

    hat_X_{mario, t+1} = X_{mario, t} + hat_vx_{mario, t+1} / 16.0
    hat_Delta_X_{i, t+1} = Delta_X_{i, t} + (hat_vx_{i, t+1} - hat_vx_{mario, t+1}) / 16.0   (for all i in 1..K)

Guarantees 0.0% kinematic violation across all dynamic entities simultaneously,
independent of entity slot ordering or number of active sprites.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.pinn_hard_residual import HardResidualPINNDynamics


class SetMultiEntityPINNDynamics(nn.Module):
    """
    Permutation-Invariant Multi-Entity World Model for Super Mario World:
    - Mario State (8D): [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
    - Dynamic Sprites ([B, K_max, 5]): [delta_x_i, delta_y_i, vx_i, vy_i, active_i]
    - Action (6D): [B, Y, UP, DOWN, LEFT, RIGHT]
    """

    def __init__(
        self,
        base_pinn: Optional[HardResidualPINNDynamics] = None,
        max_entities: int = 6,
        entity_feature_dim: int = 5,
        hidden_dim: int = 64,
        action_dim: int = 6,
        num_heads: int = 2,
        subpixels_per_pixel: float = 16.0,
    ):
        super().__init__()
        self.max_entities = max_entities
        self.entity_feature_dim = entity_feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.subpixels_per_pixel = subpixels_per_pixel

        # Base Mario physics model
        if base_pinn is not None:
            self.mario_pinn = base_pinn
        else:
            self.mario_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=action_dim)

        # Entity encoder: projects 5D sprite telemetry to hidden_dim
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-Attention: Mario Query attends to active Sprite Keys/Values
        self.mario_query_proj = nn.Linear(8, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Entity dynamic update head (parameterizes sprite acceleration)
        # Input: [entity_emb (hidden_dim) + mario_vx + mario_vy] -> [delta_vx_i, delta_vy_i]
        self.hazard_dynamics = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        mario_state: torch.Tensor,
        entities: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward simulate Mario and all K dynamic entities simultaneously.

        Args:
            mario_state: [B, 8] Mario kinematics
            entities: [B, K, 5] tensor of [delta_x, delta_y, vx, vy, active]
            action: [B, action_dim] Joypad inputs

        Returns:
            next_mario_state: [B, 8]
            next_entities: [B, K, 5]
            hazard_context: [B, hidden_dim] pooled hazard influence on Mario
        """
        B, K, _ = entities.shape

        # 1. Forward simulate Mario kinematics via Hard Residual PINN
        next_mario_state = self.mario_pinn(mario_state, action)
        mario_vx_next = next_mario_state[:, 2]
        mario_vy_next = next_mario_state[:, 3]

        # 2. Extract entity features and active mask
        # active flag is at index 4
        active_mask = entities[..., 4] > 0.5  # [B, K] boolean
        # Key padding mask for MultiheadAttention: True means IGNORE
        key_padding_mask = ~active_mask  # [B, K]

        # In case a batch item has 0 active entities, avoid NaN in attention by setting at least slot 0
        all_inactive = key_padding_mask.all(dim=-1, keepdim=True)  # [B, 1]
        safe_padding_mask = key_padding_mask.clone()
        safe_padding_mask[:, 0] = safe_padding_mask[:, 0] & (~all_inactive.squeeze(-1))

        # 3. Project entities to embedding space
        entity_emb = self.entity_encoder(entities)  # [B, K, hidden_dim]

        # 4. Multihead Cross-Attention: Mario queries active entities
        mario_q = self.mario_query_proj(mario_state).unsqueeze(1)  # [B, 1, hidden_dim]
        attn_out, _ = self.multihead_attn(
            query=mario_q,
            key=entity_emb,
            value=entity_emb,
            key_padding_mask=safe_padding_mask,
        )  # [B, 1, hidden_dim]

        # Mask out context for batch items where all entities were inactive
        hazard_context = attn_out.squeeze(1) * (~all_inactive).float()  # [B, hidden_dim]

        # 5. Predict accelerations for each entity
        # Condition on next Mario velocity
        mario_v_expanded = (
            torch.stack([mario_vx_next, mario_vy_next], dim=-1).unsqueeze(1).expand(-1, K, -1)
        )  # [B, K, 2]
        dyn_in = torch.cat([entity_emb, mario_v_expanded], dim=-1)  # [B, K, hidden_dim + 2]
        accel_residuals = self.hazard_dynamics(dyn_in)  # [B, K, 2]

        delta_vx = accel_residuals[..., 0]  # [B, K]
        delta_vy = accel_residuals[..., 1]  # [B, K]

        # 6. Entity velocity integration
        curr_vx = entities[..., 2]
        curr_vy = entities[..., 3]
        next_vx = torch.clamp(curr_vx + delta_vx, -32.0, 32.0)
        next_vy = torch.clamp(curr_vy + delta_vy, -64.0, 64.0)

        # 7. Exact Analytical Relative Kinematics Integration:
        # Delta X_{i, t+1} = Delta X_{i, t} + (vx_{i, t+1} - vx_{mario, t+1}) / 16.0
        # Delta Y_{i, t+1} = Delta Y_{i, t} + (vy_{i, t+1} - vy_{mario, t+1}) / 16.0
        curr_dx = entities[..., 0]
        curr_dy = entities[..., 1]

        rel_vx = next_vx - mario_vx_next.unsqueeze(1)
        rel_vy = next_vy - mario_vy_next.unsqueeze(1)

        next_dx = curr_dx + (rel_vx / self.subpixels_per_pixel)
        next_dy = curr_dy + (rel_vy / self.subpixels_per_pixel)

        # Zero out inactive entities
        active_f = active_mask.float()
        next_dx = torch.where(active_mask, next_dx, torch.full_like(next_dx, 999.0))
        next_dy = torch.where(active_mask, next_dy, torch.zeros_like(next_dy))
        next_vx = torch.where(active_mask, next_vx, torch.zeros_like(next_vx))
        next_vy = torch.where(active_mask, next_vy, torch.zeros_like(next_vy))

        next_entities = torch.stack([next_dx, next_dy, next_vx, next_vy, active_f], dim=-1)

        return next_mario_state, next_entities, hazard_context
