"""Variable-size sprite-set helpers (12 WRAM slots -> padded entity rows).

Centralizes the conversion from raw sprite dicts (as returned by
`SnesLibretroEmulator.get_active_sprites()`) to the [K, 5] entity rows
`[delta_x, delta_y, vx, vy, active]` consumed by `SetMultiEntityPINNDynamics`,
so recorders, trainers and tests share one implementation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

MAX_SLOTS = 12
ENTITY_DIM = 5


def sprites_to_entity_rows(
    sprites: List[Dict],
    mario_x: float,
    mario_y: float,
    max_entities: int = MAX_SLOTS,
) -> np.ndarray:
    """Maps active sprites to [max_entities, 5] rows (zero-padded).

    Rows hold egocentric telemetry [dx, dy, vx, vy, 1.0]; inactive slots are
    all zeros. Sprites are ordered by slot id for determinism.
    """
    rows = np.zeros((max_entities, ENTITY_DIM), dtype=np.float32)
    ordered = sorted(sprites, key=lambda s: int(s.get("slot", 0)))
    for i, s in enumerate(ordered[:max_entities]):
        rows[i] = np.array(
            [
                float(s["x"]) - mario_x,
                float(s["y"]) - mario_y,
                float(s["vx"]),
                float(s.get("vy", 0.0)),
                1.0,
            ],
            dtype=np.float32,
        )
    return rows
