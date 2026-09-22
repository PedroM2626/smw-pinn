"""Dynamics architectures: statistical baselines and physics-informed world models."""

from src.models.statistical_mlp import StatisticalMLPDynamics
from src.models.statistical_lstm import StatisticalLSTMDynamics
from src.models.pinn_soft import SoftPINNDynamics
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.models.pinn_invariant import TranslationInvariantPINNDynamics
from src.models.pinn_ensemble import DeepPINNEnsemble
from src.models.pinn_set_multi_entity import SetMultiEntityPINNDynamics
from src.models.tilemap_pinn import TilemapEncoder, TilemapPINNDynamics
from src.models.pinn_unified_multimodal import UnifiedMultimodalPINNDynamics

PINNInvariantDynamics = TranslationInvariantPINNDynamics
PINNEnsembleDynamics = DeepPINNEnsemble

__all__ = [
    "StatisticalMLPDynamics",
    "StatisticalLSTMDynamics",
    "SoftPINNDynamics",
    "HardResidualPINNDynamics",
    "MultiEntityPINNDynamics",
    "TranslationInvariantPINNDynamics",
    "DeepPINNEnsemble",
    "SetMultiEntityPINNDynamics",
    "TilemapEncoder",
    "TilemapPINNDynamics",
    "UnifiedMultimodalPINNDynamics",
    "PINNInvariantDynamics",
    "PINNEnsembleDynamics",
]
