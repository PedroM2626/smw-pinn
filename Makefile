# smw-pinn - canonical workflows (Linux/macOS).
# All python entry points assume `pip install -e ".[dev]"` (imports `from src...`)
# and run as modules: `python -m src.training.benchmark_experiment`.
# On Windows, where `make` is usually absent, use the equivalent console script:
# `smw-pinn test-cov`, `smw-pinn reproduce`, `smw-pinn run <module>` (see src/cli.py).
PY := python
CONFIG_DIR := configs
SMOKE_DIR := results_smoke

.PHONY: install install-cuda test test-cov lint format format-check typecheck check-all \
       reproduce benchmark sample-efficiency multiseed piml-mfrl piml-mfrl-study inverse-transfer deeponet operators symbolic-inverse symbolic-tilemap symbolic-engines inverse-mpc smoke smoke-all run install-info help

help:
	@echo "install            pip install -e .[dev] (CPU torch)"
	@echo "install-cuda       pip install -e .[dev] (CUDA 12.1 torch)"
	@echo "install-info       report resolved ROM/core/dataset paths and torch version"
	@echo "test               pytest tests/ -q"
	@echo "test-cov           pytest with coverage (fail under 30%)"
	@echo "lint               ruff check src tests scripts"
	@echo "format             ruff format (rewrites files)"
	@echo "format-check       ruff format --check (what CI and pre-commit enforce)"
	@echo "typecheck          mypy on the typed core modules"
	@echo "check-all          lint + format-check + typecheck + test-cov"
	@echo "reproduce          fast CPU smoke benchmark (configs/reproduce.yaml)"
	@echo "smoke / smoke-all  seconds-scale runs of the slow studies -> $(SMOKE_DIR)/"
	@echo "benchmark          full 4-model benchmark (configs/benchmark.yaml)"
	@echo "sample-efficiency  Pareto study (configs/sample_efficiency.yaml)"
	@echo "multiseed          K-seed significance study (configs/multiseed.yaml)"
	@echo "piml-mfrl          Physics-Informed Model-Free RL on real SNES (configs/piml_mfrl.yaml)"
	@echo "piml-mfrl-study    PIML-MFRL per-mechanism ablation (baseline/A/B/C/full, real SNES)"
	@echo "inverse-transfer   Physics parameter identification + zero-shot transfer (inverse problem, emulator-free)"
	@echo "deeponet           DeepONet neural-operator baseline (README 10.41, emulator-free)"
	@echo "operators          PC-DeepONet + FNO operator family study (README 10.42, emulator-free)"
	@echo "symbolic-inverse   Symbolic-regression inverse study: GP law discovery + probes (README 10.43, emulator-free)"
	@echo "symbolic-tilemap   Tilemap-conditioned residual discovery, the 10.43.5 counterfactual (README 10.43.8)"
	@echo "symbolic-engines   Three-engine discovery ablation: gplearn vs PySR vs template/BIC (README 10.43.9)"
	@echo "inverse-mpc        Closed-loop MPC with the inverse-problem world models (README 10.44, needs core + ROM)"
	@echo "run MOD=... ARGS=...  any module, e.g. make run MOD=src.evaluation.spatial_holdout_benchmark"
	@echo "clean              remove caches (portable: runs on Windows + Unix)"

install:
	pip install --upgrade pip
	pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu

install-cuda:
	pip install --upgrade pip
	pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121

test:
	$(PY) -m pytest tests/ -q -p no:cacheprovider

test-cov:
	$(PY) -m pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=30

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

format-check:
	$(PY) -m ruff format --check src tests scripts

# The typed-core module list lives in src/cli.py so `make`, CI and `smw-pinn`
# can never drift apart.
typecheck:
	$(PY) -m src.cli typecheck

check-all: lint format-check typecheck test-cov

install-info:
	$(PY) -m src.cli install-info

reproduce:
	$(PY) -m src.training.benchmark_experiment --config $(CONFIG_DIR)/reproduce.yaml

benchmark:
	$(PY) -m src.training.benchmark_experiment --config $(CONFIG_DIR)/benchmark.yaml

sample-efficiency:
	$(PY) -m src.evaluation.sample_efficiency_benchmark --config $(CONFIG_DIR)/sample_efficiency.yaml

multiseed:
	$(PY) -m src.evaluation.multiseed_benchmark --config $(CONFIG_DIR)/multiseed.yaml

# PIML-MFRL (README 10.39): model-free PPO on the real console with the physics-informed
# critic (A), discrete CBF filter (B) and action-violation penalty (C) enabled.
piml-mfrl:
	$(PY) -m src.training.piml_mfrl --config $(CONFIG_DIR)/piml_mfrl.yaml

# Per-mechanism ablation behind README 10.39.1 (baseline + A / B / C / A+B+C, 3 seeds).
piml-mfrl-study:
	$(PY) -m src.evaluation.piml_mfrl_study --seeds 42,43,44 --total-timesteps 10000

# Physics parameter identification (inverse problem) + zero-shot control transfer (README 10.40).
# Emulator-free: runs on CPU from the recorded dataset, so it is a CI-safe study.
inverse-transfer:
	$(PY) -m src.evaluation.inverse_transfer_benchmark

# DeepONet neural-operator baseline under the unified protocol (README 10.41).
# Emulator-free: published comparison rows are read from benchmark_metrics.json.
deeponet:
	$(PY) -m src.evaluation.deeponet_benchmark

# Neural-operator family study: DeepONet + Physics-Constrained DeepONet + FNO
# (README 10.42). Emulator-free; each row re-seeds independently.
operators:
	$(PY) -m src.evaluation.operator_benchmark

# Symbolic regression on the inverse problem: discover the update laws with genetic
# programming, probe the constants back out, compare against 10.40 (README 10.43).
# Emulator-free CPU study; writes results/symbolic_inverse_metrics.json.
symbolic-inverse:
	$(PY) -m src.evaluation.symbolic_inverse_benchmark

# Tilemap-conditioned residual discovery: the 10.43.5 grey-box control re-run with the
# recorded 7x7 terrain patch available, against a shuffled-geometry placebo (README 10.43.8).
symbolic-tilemap:
	$(PY) -m src.evaluation.symbolic_tilemap_residual_benchmark

# Three-engine discovery ablation: bounded-terminal gplearn, PySR with numerically
# optimised constants, and nested-template BIC selection (README 10.43.9).
symbolic-engines:
	$(PY) -m src.evaluation.symbolic_engine_ablation_benchmark

# Closed-loop control on the real console with the world models recovered by the inverse
# problem: established WRAM rules vs 10.40 identified constants vs 10.43 discovered laws
# vs the published Hard PINN (README 10.44). Needs the Libretro core and a ROM dump.
inverse-mpc:
	$(PY) -m src.evaluation.inverse_model_mpc_benchmark

# Seconds-scale counterparts of the studies that otherwise need a GPU and minutes
# (configs/smoke_*.yaml). They write into $(SMOKE_DIR)/ so no published artifact in
# results/ can ever be overwritten by a smoke run; tests/test_smoke_runs.py asserts
# on the same code paths.
smoke-all:
	$(PY) -m src.models.pinn_ensemble --config $(CONFIG_DIR)/smoke_pinn_ensemble.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.training.train_unified_ppo --config $(CONFIG_DIR)/smoke_unified_ppo.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.training.train_set_multi_entity --config $(CONFIG_DIR)/smoke_set_multi_entity.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.multiseed_benchmark --config $(CONFIG_DIR)/smoke_multiseed.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.sample_efficiency_benchmark --config $(CONFIG_DIR)/smoke_sample_efficiency.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.deeponet_benchmark --config $(CONFIG_DIR)/smoke_deeponet.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.operator_benchmark --config $(CONFIG_DIR)/smoke_operators.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.symbolic_inverse_benchmark --config $(CONFIG_DIR)/smoke_symbolic_inverse.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.symbolic_tilemap_residual_benchmark --config $(CONFIG_DIR)/smoke_symbolic_tilemap.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.symbolic_engine_ablation_benchmark --config $(CONFIG_DIR)/smoke_symbolic_engines.yaml --output-dir $(SMOKE_DIR)

smoke: smoke-all

# Generic passthrough for the ~40 documented entry points:
#   make run MOD=src.evaluation.spatial_holdout_benchmark
run:
	$(PY) -m $(MOD) $(ARGS)

# Portable cleanup (no rm -rf / find: works in PowerShell, cmd and sh).
clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['runs', 'results_smoke', '.pytest_cache']]; [pathlib.Path(p).unlink(missing_ok=True) for p in ['.coverage', 'coverage.xml']]; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('src').rglob('__pycache__')) + list(pathlib.Path('tests').rglob('__pycache__')) + list(pathlib.Path('scripts').rglob('__pycache__'))]"
