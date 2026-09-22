"""
pinn_unified_multimodal.py
Unified Multimodal Physics-Informed Neural Network (Unified Multimodal PINN).
Unifies:
1. Mario 8D continuous kinematics [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
2. Local 7x7 discrete spatial tilemap patch from WRAM ($7E:C800)
3. Dynamic hazard relative states (4D: [delta_x, delta_y, vx_hazard, active])
4. Action vector [B, Y, UP, DOWN, LEFT, RIGHT]

Maintains exact analytical 0.0% kinematic consistency for all interacting entities:
    hat_X_{m, t+1} = X_{m, t} + hat_vx_{m, t+1} / 16.0
    hat_Y_{m, t+1} = Y_{m, t} + hat_vy_{m, t+1} / 16.0
    hat_Delta_X_{h, t+1} = Delta_X_{h, t} + (hat_vx_{h, t+1} - hat_vx_{m, t+1}) / 16.0
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.tilemap_pinn import TilemapEncoder
from src.models.pinn_hard_residual import HardResidualPINNDynamics


class UnifiedMultimodalPINNDynamics(nn.Module):
    """
    Unified Multimodal PINN combining local terrain geometry,
    player physics, and dynamic hazards.
    """

    def __init__(
        self,
        base_pinn: Optional[HardResidualPINNDynamics] = None,
        tilemap_encoder: Optional[TilemapEncoder] = None,
        subpixels_per_pixel: float = 16.0,
        max_vx: float = 37.875,
        max_vy: float = 64.0,
    ):
        super().__init__()
        self.subpixels_per_pixel = subpixels_per_pixel
        self.max_vx = max_vx
        self.max_vy = max_vy

        # 1. Base Mario Physics
        if base_pinn is not None:
            self.mario_pinn = base_pinn
        else:
            self.mario_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)

        # 2. Spatial Terrain Encoder
        if tilemap_encoder is not None:
            self.tilemap_encoder = tilemap_encoder
        else:
            self.tilemap_encoder = TilemapEncoder(num_tile_classes=4, embed_dim=8, out_features=32)

        # 3. Hazard Dynamics Head
        self.hazard_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2),  # [delta_vx_hazard, delta_vy_hazard]
        )

        # 4. Multimodal Fusion Head (Terrain + Mario + Action -> Terrain Grounding)
        # Input: 8D Mario + 32D Tilemap + 6D Action = 46D
        self.terrain_grounding_net = nn.Sequential(
            nn.Linear(46, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 4),  # [c_ground, c_ceiling, c_left, c_right] logits
        )

    def forward(
        self,
        mario_8d: torch.Tensor,
        action: torch.Tensor,
        hazard_4d: Optional[torch.Tensor] = None,
        tilemap_patch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward step of the unified multimodal physics model.

        Args:
            mario_8d: [B, 8] Mario state [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
            action: [B, 6] Controls [B, Y, UP, DOWN, LEFT, RIGHT]
            hazard_4d: Optional [B, 4] Hazard state [delta_x, delta_y, vx_hazard, active]
            tilemap_patch: Optional [B, 7, 7] Local block IDs (0..3)

        Returns:
            Dictionary containing:
            - next_mario: [B, 8]
            - next_hazard: [B, 4] (if hazard_4d provided)
            - contact_logits: [B, 4]
        """
        # Base Mario update with analytical kinematics
        next_mario_base = self.mario_pinn(mario_8d, action)  # [B, 8]

        # Terrain feature fusion if tilemap patch is supplied
        if tilemap_patch is not None:
            terrain_feat = self.tilemap_encoder(tilemap_patch)  # [B, 32]
            fusion_in = torch.cat([mario_8d, terrain_feat, action], dim=-1)  # [B, 46]
            contact_logits = self.terrain_grounding_net(fusion_in)  # [B, 4]
            contact_probs = torch.sigmoid(contact_logits)

            # Construct refined Mario state with terrain-grounded contacts
            # Retain exact analytical kinematic positions [X, Y] and physical velocities [vx, vy]
            next_mario = torch.cat([
                next_mario_base[:, 0:4],  # X, Y, vx, vy
                contact_probs,            # c_ground, c_ceiling, c_left, c_right
            ], dim=-1)
        else:
            contact_logits = next_mario_base[:, 4:8]
            next_mario = next_mario_base

        output = {
            "next_mario": next_mario,
            "contact_logits": contact_logits,
        }

        # Dynamic Hazard update if present
        if hazard_4d is not None:
            # delta_x, delta_y, vx_h, active
            delta_x = hazard_4d[:, 0:1]
            delta_y = hazard_4d[:, 1:2]
            vx_h = hazard_4d[:, 2:3]
            active = hazard_4d[:, 3:4]

            mario_vx_next = next_mario[:, 2:3]
            mario_vy_next = next_mario[:, 3:4]

            h_in = torch.cat([delta_x, delta_y, vx_h, active, mario_vx_next, mario_vy_next], dim=-1)
            h_acc = self.hazard_net(h_in)  # [B, 2]

            # In SMW, active Rex walks at constant -16 subpx/frame unless compressed
            next_vx_h = torch.where(active > 0.5, vx_h + h_acc[:, 0:1], torch.zeros_like(vx_h))
            next_vy_h = torch.where(active > 0.5, h_acc[:, 1:2], torch.zeros_like(vx_h))

            # EXACT ANALYTICAL RELATIVE KINEMATICS:
            next_delta_x = torch.where(
                active > 0.5,
                delta_x + (next_vx_h - mario_vx_next) / self.subpixels_per_pixel,
                torch.full_like(delta_x, 999.0),
            )
            next_delta_y = torch.where(
                active > 0.5,
                delta_y + (next_vy_h - mario_vy_next) / self.subpixels_per_pixel,
                torch.zeros_like(delta_y),
            )

            next_hazard = torch.cat([next_delta_x, next_delta_y, next_vx_h, active], dim=-1)
            output["next_hazard"] = next_hazard

        return output
