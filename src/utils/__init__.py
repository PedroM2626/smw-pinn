"""Reproducibility utilities: global seeding and experiment tracking.

Submodules are imported lazily on purpose: `src.utils.seed` requires
torch, while `src.utils.experiment` works with the standard library
alone (TensorBoard/wandb are optional runtime mirrors).

    from src.utils.seed import set_global_seed
    from src.utils.experiment import ExperimentLogger
"""
