"""
tilemap_pinn.py
Tilemap-Conditioned Physics-Informed Neural Network (Tilemap-PINN).
Integrates local discrete WRAM level block geometry ($7E:C800) with Hard Kinematic Constraints.

Combines:
1. Mario 8D kinematic state vector [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
2. Local 7x7 spatial tilemap patch centered at Mario
3. 6D Joypad action vector [B, Y, UP, DOWN, LEFT, RIGHT]

Maintains exact 0.0% analytical kinematic conservation:
    hat_X_{t+1} = X_t + hat_vx_{t+1} / 16.0
    hat_Y_{t+1} = Y_t + hat_vy_{t+1} / 16.0
while grounding terrain contact flags (c_ground, c_left, etc.) in authentic level geometry.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class TilemapEncoder(nn.Module):
    """
    Lightweight 2D Convolutional Encoder for 7x7 local tile patches.
    Maps discrete block IDs to spatial terrain feature embeddings.
    """

    def __init__(self, num_tile_classes: int = 4, embed_dim: int = 8, out_features: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_tile_classes, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # [B, 24, 4, 4]
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # [B, 24, 2, 2]
            nn.Flatten(),                 # [B, 96]
            nn.Linear(96, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, tile_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tile_patch: [B, H=7, W=7] integer tensor with tile class IDs (0..3)
        Returns:
            features: [B, out_features]
        """
        emb = self.embedding(tile_patch)  # [B, 7, 7, embed_dim]
        emb = emb.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, 7, 7]
        return self.conv(emb)


class TilemapPINNDynamics(nn.Module):
    """
    Structural PINN conditioned on both continuous kinematics and discrete local terrain.
    """

    def __init__(
        self,
        kinematic_dim: int = 8,
        action_dim: int = 6,
        tile_features_dim: int = 32,
        hidden_dim: int = 64,
        subpixels_per_pixel: float = 16.0,
        max_vx: float = 48.0,
        max_vy: float = 64.0,
    ):
        super().__init__()
        self.subpixels_per_pixel = subpixels_per_pixel
        self.max_vx = max_vx
        self.max_vy = max_vy

        # 1. Terrain spatial encoder
        self.tile_encoder = TilemapEncoder(num_tile_classes=4, embed_dim=8, out_features=tile_features_dim)

        # 2. Residual dynamics network
        # Input: kinematic_dim (8) + tile_features_dim (32) + action_dim (6) = 46
        total_in = kinematic_dim + tile_features_dim + action_dim
        self.backbone = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Output heads
        self.head_accel = nn.Linear(hidden_dim, 2)     # [delta_vx, delta_vy]
        self.head_contact = nn.Linear(hidden_dim, 4)   # [c_ground, c_ceiling, c_left, c_right]

    def forward(
        self,
        kinematics_8d: torch.Tensor,
        tile_patch: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            kinematics_8d: [B, 8] [X, Y, vx, vy, c_ground, c_ceiling, c_left, c_right]
            tile_patch: [B, 7, 7] integer tensor of discrete tile classes
            action: [B, 6] action primitives [B, Y, UP, DOWN, LEFT, RIGHT]

        Returns:
            next_kinematics_8d: [B, 8]
            terrain_features: [B, 32]
        """
        b_x = kinematics_8d[:, 0]
        b_y = kinematics_8d[:, 1]
        b_vx = kinematics_8d[:, 2]
        b_vy = kinematics_8d[:, 3]

        # 1. Extract terrain geometry embedding
        terrain_emb = self.tile_encoder(tile_patch)

        # 2. Predict force and contact residuals
        joint_input = torch.cat([kinematics_8d, terrain_emb, action], dim=-1)
        feat = self.backbone(joint_input)

        accel = self.head_accel(feat)
        contact_logits = self.head_contact(feat)
        contact_probs = torch.sigmoid(contact_logits)

        # 3. Velocity update with physical saturation
        next_vx = torch.clamp(b_vx + accel[:, 0], -self.max_vx, self.max_vx)
        next_vy = torch.clamp(b_vy + accel[:, 1], -self.max_vy, self.max_vy)

        # 4. Exact Discrete Kinematic Conservation Layer:
        # hat_X_{t+1} = X_t + hat_vx_{t+1} / 16.0
        # hat_Y_{t+1} = Y_t + hat_vy_{t+1} / 16.0
        next_x = b_x + (next_vx / self.subpixels_per_pixel)
        next_y = b_y + (next_vy / self.subpixels_per_pixel)

        # 5. Assemble next 8D kinematic state
        next_kinematics = torch.stack(
            [
                next_x,
                next_y,
                next_vx,
                next_vy,
                contact_probs[:, 0],
                contact_probs[:, 1],
                contact_probs[:, 2],
                contact_probs[:, 3],
            ],
            dim=-1,
        )

        return next_kinematics, terrain_emb
