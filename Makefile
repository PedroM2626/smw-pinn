# smw-pinn — canonical workflows.
# All python entry points assume `pip install -e ".[dev]"` (imports `from src...`)
# and run as modules: `python -m src.training.benchmark_experiment`.
PY := python
CONFIG_DIR := configs

.PHONY: install install-cuda test test-cov lint typecheck reproduce benchmark sample-efficiency multiseed clean help

help:
	@echo "install            pip install -e .[dev] (CPU torch)"
	@echo "install-cuda       pip install -e .[dev] (CUDA 12.1 torch)"
	@echo "test               pytest tests/ -q"
	@echo "test-cov           pytest with coverage (fail under 30%)"
	@echo "lint               ruff check src tests scripts"
	@echo "typecheck          mypy on typed core modules"
	@echo "reproduce          fast CPU smoke benchmark (configs/reproduce.yaml)"
	@echo "benchmark          full 4-model benchmark (configs/benchmark.yaml)"
	@echo "sample-efficiency  Pareto study (configs/sample_efficiency.yaml)"
	@echo "multiseed          K-seed significance study (configs/multiseed.yaml)"
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

typecheck:
	$(PY) -m mypy src/utils/config.py src/utils/logging.py src/utils/seed.py src/evaluation/per_variable_metrics.py src/evaluation/rollout_evaluator.py src/environment/dataset_loader.py src/training/trainer.py src/models/statistical_mlp.py src/planning/terminal_value.py src/planning/global_planner.py src/planning/tilemap_mpc.py src/perception/pixel_encoder.py src/perception/vision_dataset.py src/environment/sprite_sets.py src/models/pinn_gravity.py

reproduce:
	$(PY) -m src.training.benchmark_experiment --config $(CONFIG_DIR)/reproduce.yaml

benchmark:
	$(PY) -m src.training.benchmark_experiment --config $(CONFIG_DIR)/benchmark.yaml

sample-efficiency:
	$(PY) -m src.evaluation.sample_efficiency_benchmark --config $(CONFIG_DIR)/sample_efficiency.yaml

multiseed:
	$(PY) -m src.evaluation.multiseed_benchmark --config $(CONFIG_DIR)/multiseed.yaml

# Portable cleanup (no rm -rf / find: works in PowerShell, cmd and sh).
clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['runs', '.pytest_cache']]; [pathlib.Path(p).unlink(missing_ok=True) for p in ['.coverage', 'coverage.xml']]; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('src').rglob('__pycache__')) + list(pathlib.Path('tests').rglob('__pycache__')) + list(pathlib.Path('scripts').rglob('__pycache__'))]"
