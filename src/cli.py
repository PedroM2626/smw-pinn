"""
cli.py
Cross-platform task runner: the ``smw-pinn`` console script.

The Makefile is the canonical entry point on Linux/macOS, but ``make`` is not
present on a default Windows install (and this project's reference hardware is
Windows, because that is where the Libretro core loads). Every workflow is
therefore also reachable through this module, which simply forwards to
``python -m <module>`` in a *subprocess* so behaviour is byte-identical to the
documented commands (no in-process ``__main__`` surprises for spawn-based
DataLoader workers).

Usage::

    smw-pinn list
    smw-pinn install-info
    smw-pinn lint | format | format-check | typecheck | test | test-cov
    smw-pinn reproduce | benchmark | sample-efficiency | multiseed
    smw-pinn run src.evaluation.spatial_holdout_benchmark --extra-flag
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

# Workflow name -> module executed with `python -m`.
MODULE_COMMANDS: dict[str, str] = {
    "reproduce": "src.training.benchmark_experiment",
    "benchmark": "src.training.benchmark_experiment",
    "sample-efficiency": "src.evaluation.sample_efficiency_benchmark",
    "multiseed": "src.evaluation.multiseed_benchmark",
    "ablation": "src.evaluation.ablation_benchmark",
    "profiling": "src.evaluation.benchmark_computational_efficiency",
    "mpc": "src.evaluation.mbrl_mpc_benchmark",
    "dyna-ppo": "src.training.dyna_ppo",
    "policy-eval": "src.evaluation.evaluate_policy_snes",
    "sprites": "src.evaluation.evaluate_sprites_snes",
    "model-free-ppo": "src.training.model_free_ppo",
    "ensemble": "src.models.pinn_ensemble",
    "mbpo": "src.training.online_mbpo",
    "multi-entity-train": "src.training.train_multi_entity",
    "multi-entity-mpc": "src.evaluation.evaluate_multi_entity_mpc",
    "multi-entity-hardware": "src.evaluation.evaluate_multi_entity_snes",
    "tilemap-train": "src.training.train_tilemap",
    "tilemap-mpc": "src.evaluation.evaluate_tilemap_mpc",
    "distill": "src.training.distill_mpc_policy",
    "distill-eval": "src.evaluation.evaluate_distilled_policy_snes",
    "extended-navigation": "src.evaluation.evaluate_extended_navigation",
    "cross-level": "src.evaluation.cross_level_benchmark",
    "cross-level-control": "src.evaluation.evaluate_cross_level_control",
    "unified-ppo": "src.training.train_unified_ppo",
    "unified-joint": "src.training.train_unified_multimodal",
    "full-clearance": "src.evaluation.evaluate_full_level_clearance",
    "dagger": "src.training.train_dagger",
    "dagger-eval": "src.evaluation.evaluate_dagger_snes",
    "video": "src.evaluation.render_level_clearance_video",
    "animation": "src.evaluation.render_comparison_animation",
    "diagnose-obstacle": "src.evaluation.diagnose_obstacle_1000",
    "reflex-ablation": "src.evaluation.mpc_reflex_ablation",
    "set-multi-entity-train": "src.training.train_set_multi_entity",
    "pixel-train": "src.training.train_pixel_estimator",
    "pixel-mpc": "src.evaluation.evaluate_pixel_mpc",
    "hierarchical-mpc": "src.evaluation.evaluate_hierarchical_mpc",
    "terminal-value-train": "src.training.train_terminal_value",
    "spatial-holdout": "src.evaluation.spatial_holdout_benchmark",
    "learning-curves": "src.evaluation.plot_learning_curves",
    "baselines": "src.evaluation.analytical_baselines",
    "piml-mfrl": "src.training.piml_mfrl",
}

# Config file injected for the workflows that accept --config.
CONFIG_FLAGS: dict[str, str] = {
    "reproduce": "configs/reproduce.yaml",
    "benchmark": "configs/benchmark.yaml",
    "sample-efficiency": "configs/sample_efficiency.yaml",
    "multiseed": "configs/multiseed.yaml",
    "piml-mfrl": "configs/piml_mfrl.yaml",
}

# Seconds-scale counterparts of the studies that need a GPU and minutes. They run
# the same code paths but write into `results_smoke/`, so a smoke run can never
# overwrite a published artifact (`make smoke-all` runs exactly this list).
SMOKE_OUTPUT_DIR = "results_smoke"
SMOKE_RUNS: tuple[tuple[str, str], ...] = (
    ("src.models.pinn_ensemble", "configs/smoke_pinn_ensemble.yaml"),
    ("src.training.train_unified_ppo", "configs/smoke_unified_ppo.yaml"),
    ("src.training.train_set_multi_entity", "configs/smoke_set_multi_entity.yaml"),
    ("src.evaluation.multiseed_benchmark", "configs/smoke_multiseed.yaml"),
    ("src.evaluation.sample_efficiency_benchmark", "configs/smoke_sample_efficiency.yaml"),
)

# Quality gates: the module plus the arguments this project standardizes on.
CHECK_COMMANDS: dict[str, list[str]] = {
    "lint": ["ruff", "check", "src", "tests", "scripts"],
    "format": ["ruff", "format", "src", "tests", "scripts"],
    "format-check": ["ruff", "format", "--check", "src", "tests", "scripts"],
    "typecheck": [
        "mypy",
        "src/utils/config.py",
        "src/utils/logging.py",
        "src/utils/paths.py",
        "src/utils/seed.py",
        "src/environment/wram.py",
        "src/evaluation/per_variable_metrics.py",
        "src/evaluation/rollout_evaluator.py",
        "src/environment/dataset_loader.py",
        "src/training/trainer.py",
        "src/models/statistical_mlp.py",
        "src/planning/terminal_value.py",
        "src/planning/global_planner.py",
        "src/planning/tilemap_mpc.py",
        "src/perception/pixel_encoder.py",
        "src/perception/vision_dataset.py",
        "src/environment/sprite_sets.py",
        "src/models/pinn_gravity.py",
        "src/losses/physics_rl_losses.py",
        "src/models/cbf_projection.py",
    ],
    "test": ["pytest", "tests/", "-q", "-p", "no:cacheprovider"],
    "test-cov": [
        "pytest",
        "tests/",
        "-q",
        "-p",
        "no:cacheprovider",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-fail-under=30",
    ],
}


def _run(argv: Sequence[str]) -> int:
    """Run `python -m argv[0] argv[1:]` and propagate its exit code."""
    print(f"$ {sys.executable} -m {' '.join(argv)}", flush=True)
    return subprocess.run([sys.executable, "-m", *argv], check=False).returncode


def _install_info() -> int:
    """Report whether the repository's assets and tooling are reachable."""
    from src.utils import paths

    print(f"repository root  : {paths.REPO_ROOT}")
    for label, value in (
        ("libretro core", paths.CORE_PATH),
        ("rom", paths.ROM_PATH),
        ("gameplay dataset", paths.DATASET_GAMEPLAY),
        ("yoshi's island 1 state", paths.STATE_YOSHI_ISLAND_1),
    ):
        print(f"{label:<17}: {value} [{'found' if _is_file(value) else 'MISSING'}]")
    if _is_file(paths.ROM_PATH):
        digest = paths.verify_rom_sha1(paths.ROM_PATH)
        status = "matches published SHA-1" if digest == paths.ROM_SHA1_USA else "SHA-1 MISMATCH"
        print(f"rom integrity    : {digest} ({status})")
    for tool in ("torch", "numpy", "matplotlib"):
        try:
            mod = __import__(tool)
            print(f"{tool:<17}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{tool:<17}: NOT installed (run `pip install -e .[dev]`)")
    return 0


def _is_file(value: str) -> bool:
    return bool(value) and os.path.isfile(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smw-pinn",
        description="Canonical workflows for the SMW PINN benchmark (no `make` required).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="workflow name (see `smw-pinn list`), or 'run' to invoke any module",
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER, help="arguments forwarded to the workflow"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command is None or command in ("-h", "--help", "help"):
        parser.print_help()
        return 0
    if command in ("list", "--list"):
        _print_catalog()
        return 0
    if command == "install-info":
        return _install_info()

    if command == "run":
        rest = [a for a in args.rest if a != "--"]
        if not rest:
            print("usage: smw-pinn run <module.dotted.path> [args...]", file=sys.stderr)
            return 2
        return _run(rest)

    if command == "check-all":
        for step in ("lint", "format-check", "typecheck", "test-cov"):
            code = main([step])
            if code:
                print(f"`{step}` failed", file=sys.stderr)
                return code
        return 0

    if command in ("smoke", "smoke-all"):
        for module, config in SMOKE_RUNS:
            code = _run([module, "--config", config, "--output-dir", SMOKE_OUTPUT_DIR])
            if code:
                print(f"smoke run of {module} failed", file=sys.stderr)
                return code
        print(f"smoke artifacts written under {SMOKE_OUTPUT_DIR}/")
        return 0

    if command in CHECK_COMMANDS:
        return _run(CHECK_COMMANDS[command])

    if command in MODULE_COMMANDS:
        module = MODULE_COMMANDS[command]
        rest = list(args.rest)
        config = CONFIG_FLAGS.get(command)
        if config and not any(a.startswith("--config") for a in rest):
            rest = ["--config", config, *rest]
        return _run([module, *rest])

    print(f"unknown command {command!r}\n", file=sys.stderr)
    _print_catalog()
    return 2


def _print_catalog() -> None:
    print("quality gates  :", ", ".join(sorted(CHECK_COMMANDS)))
    print("benchmarks     :", ", ".join(sorted(CONFIG_FLAGS)))
    print("entry points   :", ", ".join(sorted(MODULE_COMMANDS)))
    print("other          : run <module>, install-info, check-all, smoke-all")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
