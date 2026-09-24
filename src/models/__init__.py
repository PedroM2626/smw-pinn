"""Dynamics architectures: statistical baselines and physics-informed world models."""

from src.models.analytical_kinematics import (
    AnalyticalKinematicsDynamics,
    EngineRuleParameters,
)
from src.models.cbf_projection import (
    CBFQPLayer,
    DiscreteCBFCategoricalFilter,
    smw_barrier_affine,
)
from src.models.deeponet import DeepONetDynamics, PhysicsConstrainedDeepONetDynamics
from src.models.fno import FNODynamics
from src.models.pinn_ensemble import DeepPINNEnsemble
from src.models.pinn_gravity import GravityIdentifiedPINNDynamics
from src.models.pinn_hard_residual import HardResidualPINNDynamics
from src.models.pinn_invariant import TranslationInvariantPINNDynamics
from src.models.pinn_multi_entity import MultiEntityPINNDynamics
from src.models.pinn_set_multi_entity import SetMultiEntityPINNDynamics
from src.models.pinn_soft import SoftPINNDynamics
from src.models.pinn_unified_multimodal import UnifiedMultimodalPINNDynamics
from src.models.statistical_lstm import StatisticalLSTMDynamics
from src.models.statistical_mlp import (
    MATCHED_HIDDEN_DIMS,
    StatisticalMLPDynamics,
    build_param_matched_mlp,
)
from src.models.tilemap_pinn import TilemapEncoder, TilemapPINNDynamics

PINNInvariantDynamics = TranslationInvariantPINNDynamics
PINNEnsembleDynamics = DeepPINNEnsemble

__all__ = [
    "AnalyticalKinematicsDynamics",
    "EngineRuleParameters",
    "CBFQPLayer",
    "DiscreteCBFCategoricalFilter",
    "smw_barrier_affine",
    "StatisticalMLPDynamics",
    "build_param_matched_mlp",
    "MATCHED_HIDDEN_DIMS",
    "StatisticalLSTMDynamics",
    "DeepONetDynamics",
    "PhysicsConstrainedDeepONetDynamics",
    "FNODynamics",
    "SoftPINNDynamics",
    "HardResidualPINNDynamics",
    "GravityIdentifiedPINNDynamics",
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
