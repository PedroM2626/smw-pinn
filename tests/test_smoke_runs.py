"""Smoke counterpart of the studies that otherwise need minutes and a GPU.

The full §8.3/§8.4/§10.10/§10.22/§10.29 studies are expensive, so CI only ever
*reads* their committed artifacts - which means the code that produces them can rot
without any test noticing. These tests exercise that code end to end at a budget
that costs seconds.

Two isolation rules keep a smoke run from damaging the published record:
`make smoke-all` / `smw-pinn smoke-all` write into `results_smoke/` (git-ignored),
and the executable tests below write into a pytest temporary directory.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.cli import SMOKE_OUTPUT_DIR, SMOKE_RUNS
from src.utils.provenance import read_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]

# Artifacts each smoke run must produce, used by the executable smoke test. Only the
# two cheapest pipelines are executed here; the rest are covered by the contract
# tests below and by `make smoke-all` on developer machines.
SMOKE_ARTIFACTS = {
    "src.models.pinn_ensemble": "pinn_ensemble_metrics.json",
    "src.training.train_unified_ppo": "unified_ppo_metrics.json",
}


def _run_python(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _cli_flags(module: str) -> set[str]:
    """Every `--flag` the entry point really accepts, read from its own --help."""
    proc = _run_python("-m", module, "--help")
    assert proc.returncode == 0, (
        f"`python -m {module} --help` exited {proc.returncode}. Entry points documented "
        f"in README 11.5 must expose a CLI that accepts --config.\n"
        f"stderr: {proc.stderr[-500:]}"
    )
    return {f.lstrip("-").replace("-", "_") for f in re.findall(r"--[a-z][a-z0-9-]*", proc.stdout)}


@pytest.mark.parametrize("module,config", SMOKE_RUNS)
def test_smoke_config_only_sets_supported_flags(module: str, config: str) -> None:
    """A config key that the parser does not know is silently ignored.

    This is the exact defect class that made `--no-per-variable-metrics` and
    `--non-deterministic` documented-but-dead: the run looked configured and was
    not. Check the documented smoke budgets against the real CLI surface.
    """
    path = REPO_ROOT / config
    assert path.is_file(), f"{config} is listed in src/cli.py SMOKE_RUNS but does not exist"
    keys = set(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    accepted = _cli_flags(module)
    unsupported = sorted(keys - accepted)
    assert not unsupported, (
        f"{config} sets {unsupported}, which `python -m {module}` does not accept"
    )


@pytest.mark.parametrize("module,config", SMOKE_RUNS)
def test_smoke_config_sets_a_real_parameter(module: str, config: str) -> None:
    """Guard against a budget file that only ever sets --output-dir."""
    keys = set(yaml.safe_load((REPO_ROOT / config).read_text(encoding="utf-8")) or {})
    assert keys - {"output_dir", "dataset_path"}, f"{config} overrides nothing but paths"


# This file costs about half a minute in total (each smoke run is a subprocess with
# a torch import), which is the price of covering seven pipelines that no other test
# reaches. Keep the budgets inside configs/smoke_*.yaml, not here.
@pytest.mark.parametrize("module,config", SMOKE_RUNS)
def test_smoke_runs_execute_and_record_provenance(module: str, config: str, tmp_path) -> None:
    """Execute the smoke pipelines that are fast enough to run on every commit."""
    artifact = SMOKE_ARTIFACTS.get(module)
    if artifact is None:
        pytest.skip(f"{module} is exercised by `make smoke-all`, not per-commit")
    proc = _run_python("-m", module, "--config", config, "--output-dir", str(tmp_path))
    assert proc.returncode == 0, (
        f"`python -m {module}` failed:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
    )
    produced = tmp_path / artifact
    assert produced.is_file(), f"{module} did not write {artifact}"
    payload = read_metrics(produced)
    assert "_meta" in payload, (
        f"{artifact} lacks provenance: write it with src.utils.provenance.write_metrics"
    )
    assert payload["_meta"]["command"], f"{artifact} records no regenerating command"


def test_smoke_output_directory_is_not_committed() -> None:
    """`results/` is the published record; smoke output must never be committable."""
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert SMOKE_OUTPUT_DIR in ignore, (
        f"{SMOKE_OUTPUT_DIR}/ is produced by `make smoke-all` but not git-ignored"
    )
    # If the directory exists at all, git must consider every file in it ignorable.
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", f"{SMOKE_OUTPUT_DIR}/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "", (
        f"smoke output is not ignored by git, so a `make smoke-all` run could get "
        f"committed over published numbers:\n{proc.stdout}"
    )
