"""
src.inverse
Inverse-problem methods for the SMW physics world model.

Where the rest of the repository solves the *forward* problem (predict ``s_{t+1}`` and plan
given known physics), this package solves the *inverse* problem: recover the engine's
physical constants from observed trajectories (system identification) and show that the
recovered constants let a model-based controller transfer to dynamics with an unknown
discretization scheme. See ``parameter_identification.py`` and README Section 10.40.
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

__all__ = [
    "EngineParams",
    "PARAM_NAMES",
    "bootstrap_ci",
    "generate_synthetic_windows",
    "identify_params",
    "log_posterior",
    "make_windows",
    "mpc_random_shooting",
    "per_variable_mse",
    "posterior_laplace",
    "mcmc_random_walk",
    "rollout_mse",
    "sample_posterior",
    "simulate_rollout",
    "simulate_step",
    "theta_tensor",
]
