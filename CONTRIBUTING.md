# Contributing to smw-pinn

Short rules for keeping this repo reproducible. English or Portuguese, both fine.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback: ... --extra-index-url https://download.pytorch.org/whl/cpu
git lfs install && git lfs pull   # datasets, checkpoints, videos
```

You also need your own dump of the commercial ROM (SHA-1 in README);
never commit ROMs, savestates with personal data, or API keys.

## Dependency envelope

`torch>=2.5.1,<2.7`, `torchvision>=0.20.1,<0.22`, `numpy>=1.24.0,<2.1` are
validated on RTX 4070 + CUDA 12.1. Dependabot is configured
(`.github/dependabot.yml`) to stay inside these caps — widen them only with a
full hardware re-validation (benchmarks + `make test-cov` on CUDA). CI uploads
`pytest.log` as an artifact on failure; check it before re-running blindly.

## Canonical commands (use these, not ad-hoc scripts)

```bash
make lint        # ruff check src tests scripts — must pass
make typecheck   # mypy on typed core modules — must pass
make test        # fast unit tests
make test-cov    # tests + coverage gate (baseline 30%)
make reproduce   # 2-epoch CPU smoke benchmark (configs/reproduce.yaml)
make benchmark sample-efficiency multiseed
```

Run entry points as modules (`python -m src.training.benchmark_experiment`),
never `cd src/` + relative imports. `src/` must not contain `sys.path` hacks;
only `scripts/` bootstraps the repo root.

## Code conventions

- **Logging, not `print()`:** `from src.utils.logging import get_logger; log = get_logger(__name__)`.
  `print()` is allowed only in `scripts/` (CLI tools).
- **Determinism:** seed via `src.utils.seed.set_global_seed`; DataLoaders take
  an explicit `seed=` (seeded `torch.Generator` + `seed_worker`). Never use
  unseeded `shuffle=True` or bare `np.random` in training/eval code.
- **Configs:** shared knobs live in `configs/*.yaml`; entry points accept
  `--config` with CLI-overrides-file semantics (`src/utils/config.py`).
  Warn (don't silently ignore) on unknown keys.
- **Metrics:** aggregate MSE hides scale imbalance — report per-variable
  metrics (`src/evaluation/per_variable_metrics.py`) alongside it, and
  multi-start rollouts instead of a single trajectory.
- **Tests:** every new module gets a test file. Emulator-dependent tests must
  use `@requires_emulator` from `tests/conftest.py` (Windows DLL doesn't load
  on Linux CI — tests must skip, never error).
- **Types:** new/edited code in the typed core (`src/utils/`, dataset loader,
  trainer, rollout evaluator, per-variable metrics) must pass
  `make typecheck`. Pre-commit (`pre-commit install`) runs ruff on every commit.
- **Regression gate:** if you change training/eval code, check
  `tests/test_metrics_regression.py` still passes; regenerate `results/*.json`
  on GPU and update the README tables that cite them (§8.1–§8.4).
- **Pixel datasets are huge:** `scripts/record_pixel_gameplay.py` defaults to
  frame stride + downscale for a reason (full-res frames ≈ GBs). Keep frame
  `.npz` files in Git-LFS and never commit ROMs or smoke-test dirs.

## Pull requests

- `make lint` + `make typecheck` + `make test-cov` green (CI runs all three on ubuntu-latest).
- Small, focused PRs with a description of what changed and which
  `results/*.json` / README tables were affected, if any.
- Don't commit: `.venv/`, `runs/`, `*.mp4/*.gif` outside `results/figures/`,
  ROMs, `.coverage`, or smoke-test dirs (`results_smoke*/`).
