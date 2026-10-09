# smw-pinn - canonical workflows (Linux/macOS).
# All python entry points assume `pip install -e ".[dev]"` (imports `from src...`)
# and run as modules: `python -m src.training.benchmark_experiment`.
# On Windows, where `make` is usually absent, use the equivalent console script:
# `smw-pinn test-cov`, `smw-pinn reproduce`, `smw-pinn run <module>` (see src/cli.py).
PY := python
CONFIG_DIR := configs
SMOKE_DIR := results_smoke

.PHONY: install install-cuda test test-cov lint format format-check typecheck check-all \
       reproduce benchmark sample-efficiency multiseed piml-mfrl piml-mfrl-study inverse-transfer deeponet operators symbolic-inverse symbolic-tilemap symbolic-engines inverse-mpc sprint-excitation learned-probes physics-injection plateau-provenance projection-cell gate-excitation effective-velocity physics-claims neural-ode sindy corrected-physics record-jump record-sprint metric-decomposition velocity-classes smoke smoke-all run install-info help

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
	@echo "record-sprint      Excitation-targeted WRAM recording that saturates the speed bound (README 10.45, needs core + ROM)"
	@echo "sprint-excitation  The velocity ceiling measured on both recordings (README 10.45)"
	@echo "learned-probes     Fixed-point and gravity-gate probes on the learned models (README 10.46)"
	@echo "physics-injection  Target x mechanism x family grid of world models (README 10.47)"
	@echo "metric-decomposition  Split the rollout violation figure into consistency/smoothness (10.48)"
	@echo "velocity-classes  Which documented speed constant the telemetry supports (README 10.49)"
	@echo "plateau-provenance  Does a learned plateau follow the bound or the data support? (10.50)"
	@echo "projection-cell  The state x hard output-projection cell, fitted and flown (10.51)"
	@echo "gate-excitation  The held-jump gate measured on three recordings (README 10.52)"
	@echo "effective-velocity  Predict the velocity the engine integrates with (10.53)"
	@echo "physics-claims  Audit README section 4 against the code and the telemetry (10.54)"
	@echo "neural-ode  Learn a continuous field, choose the integrator (10.55)"
	@echo "sindy  Sparse identification of the engine's law on every recording (10.56)"
	@echo "corrected-physics  What the three section-4 corrections cost, open loop and on the console (10.57, 10.57.1)"
	@echo "record-jump        Record jump-excited WRAM telemetry (needs the console)"
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

# The recording step is separate from the analysis because it is the only part that
# needs the console: the study then compares the two recordings.
record-sprint:
	$(PY) scripts/record_sprint_gameplay.py

sprint-excitation:
	$(PY) -m src.evaluation.sprint_excitation_benchmark

# The 10.43 structural probes applied to the committed learned checkpoints, plus the
# identification-through-a-surrogate control (README 10.46). Emulator-free.
learned-probes:
	$(PY) -m src.evaluation.learned_structure_probe_benchmark

physics-injection:
	$(PY) -m src.evaluation.operator_physics_injection_benchmark

metric-decomposition:
	$(PY) -m src.evaluation.kinematic_metric_decomposition_benchmark

velocity-classes:
	$(PY) -m src.evaluation.velocity_class_benchmark

plateau-provenance:
	$(PY) -m src.evaluation.plateau_provenance_benchmark

projection-cell:
	$(PY) -m src.evaluation.projection_cell_benchmark
	$(PY) -m src.evaluation.physics_injection_mpc_benchmark --study projection

gate-excitation:
	$(PY) -m src.evaluation.gate_excitation_benchmark

effective-velocity:
	$(PY) -m src.evaluation.effective_velocity_benchmark
	$(PY) -m src.evaluation.physics_injection_mpc_benchmark --study effective

physics-claims:
	$(PY) -m src.evaluation.physics_claim_audit

neural-ode:
	$(PY) -m src.evaluation.neural_ode_integrator_benchmark
	$(PY) -m src.evaluation.physics_injection_mpc_benchmark --study ode

sindy:
	$(PY) -m src.evaluation.sindy_identification_benchmark

# What the corrected forms of section 4 cost or buy, per family, paired over seeds
# (README 10.57), then the same arms flown by the published planner on the real console
# (10.57.1, needs core + ROM). The forward leg publishes the eight flown arms' weights under
# its own corrphys_ prefix; run the module directly to train without writing any.
corrected-physics:
	$(PY) -m src.evaluation.corrected_physics_ablation --save-checkpoints
	$(PY) -m src.evaluation.physics_injection_mpc_benchmark --study corrected

record-jump:
	$(PY) scripts/record_jump_gameplay.py

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
	$(PY) -m src.evaluation.operator_physics_injection_benchmark --config $(CONFIG_DIR)/smoke_physics_injection.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.kinematic_metric_decomposition_benchmark --config $(CONFIG_DIR)/smoke_metric_decomposition.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.velocity_class_benchmark --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.plateau_provenance_benchmark --config $(CONFIG_DIR)/smoke_plateau_provenance.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.projection_cell_benchmark --config $(CONFIG_DIR)/smoke_projection_cell.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.gate_excitation_benchmark --config $(CONFIG_DIR)/smoke_gate_excitation.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.effective_velocity_benchmark --config $(CONFIG_DIR)/smoke_effective_velocity.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.physics_claim_audit --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.neural_ode_integrator_benchmark --config $(CONFIG_DIR)/smoke_neural_ode.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.sindy_identification_benchmark --config $(CONFIG_DIR)/smoke_sindy.yaml --output-dir $(SMOKE_DIR)
	$(PY) -m src.evaluation.corrected_physics_ablation --config $(CONFIG_DIR)/smoke_corrected_physics.yaml --output-dir $(SMOKE_DIR)

smoke: smoke-all

# Generic passthrough for the ~40 documented entry points:
#   make run MOD=src.evaluation.spatial_holdout_benchmark
run:
	$(PY) -m $(MOD) $(ARGS)

# Portable cleanup (no rm -rf / find: works in PowerShell, cmd and sh).
clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['runs', 'results_smoke', '.pytest_cache']]; [pathlib.Path(p).unlink(missing_ok=True) for p in ['.coverage', 'coverage.xml']]; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('src').rglob('__pycache__')) + list(pathlib.Path('tests').rglob('__pycache__')) + list(pathlib.Path('scripts').rglob('__pycache__'))]"
