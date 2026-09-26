"""
src.inverse
Inverse-problem methods for the SMW physics world model.

Where the rest of the repository solves the *forward* problem (predict ``s_{t+1}`` and plan
given known physics), this package solves the *inverse* problem: recover the engine's
physical constants from observed trajectories (system identification) and show that the
recovered constants let a model-based controller transfer to dynamics with an unknown
discretization scheme. See ``parameter_identification.py`` and README Section 10.40.

``symbolic_regression.py`` removes the assumption that the law is already known: genetic
programming discovers the one-step update laws from transitions, and a probe stage reads the
physical constants back out of whatever expression was discovered (README Section 10.43).
"""

from src.inverse.parameter_identification import (
    PARAM_NAMES,
    EngineParams,
    bootstrap_ci,
    generate_synthetic_windows,
    identify_params,
    log_posterior,
    make_windows,
    mcmc_random_walk,
    mpc_random_shooting,
    per_variable_mse,
    posterior_laplace,
    rollout_mse,
    sample_posterior,
    simulate_rollout,
    simulate_step,
    theta_tensor,
)
from src.inverse.symbolic_regression import (
    GP_FUNCTION_SET,
    LAW_FEATURES,
    LAW_NAMES,
    AnalyticLaw,
    BaggedLaw,
    LawBank,
    SymbolicLaw,
    TransitionBank,
    bank_from_transitions,
    bank_from_windows,
    evaluate_law,
    excitation_weights,
    fit_law,
    fit_law_bank,
    law_matrices,
    model_rollout,
    parametric_laws,
    probe_constants,
    relative_error,
    structural_tags,
    summarize_constants,
    symbolic_step,
    truth_vector,
)

__all__ = [
    "AnalyticLaw",
    "BaggedLaw",
    "EngineParams",
    "GP_FUNCTION_SET",
    "LAW_FEATURES",
    "LAW_NAMES",
    "LawBank",
    "PARAM_NAMES",
    "SymbolicLaw",
    "TransitionBank",
    "bank_from_transitions",
    "bank_from_windows",
    "bootstrap_ci",
    "evaluate_law",
    "excitation_weights",
    "fit_law",
    "fit_law_bank",
    "generate_synthetic_windows",
    "identify_params",
    "law_matrices",
    "log_posterior",
    "make_windows",
    "mcmc_random_walk",
    "model_rollout",
    "mpc_random_shooting",
    "parametric_laws",
    "per_variable_mse",
    "posterior_laplace",
    "probe_constants",
    "relative_error",
    "rollout_mse",
    "sample_posterior",
    "simulate_rollout",
    "simulate_step",
    "structural_tags",
    "summarize_constants",
    "symbolic_step",
    "theta_tensor",
    "truth_vector",
]
