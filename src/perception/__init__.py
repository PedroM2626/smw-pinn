"""Pixel-to-state perception front-end (Feature A).

Supervised visual state estimation: RGB SNES frames -> 8D WRAM state vector.
This closes the "privileged telemetry" limitation (§10.30-2) by showing the
kinematic prior can operate on estimated states when WRAM is unavailable.
"""

from src.perception.pixel_encoder import PixelStateEstimator, StateNormalizer
from src.perception.vision_dataset import FrameStateDataset, create_vision_loaders

__all__ = [
    "PixelStateEstimator",
    "StateNormalizer",
    "FrameStateDataset",
    "create_vision_loaders",
]
