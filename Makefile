# smw-pinn — canonical workflows.
# All python entry points assume `pip install -e ".[dev]"` (imports `from src...`)
# and run as modules: `python -m src.training.benchmark_experiment`.
PY := python
CONFIG_DIR := configs

.PHONY: install test test-cov lint reproduce benchmark sample-efficiency multiseed rollout clean help

help:
	@echo "install            pip install -e .[dev] (CPU torch)"
	@echo "install-cuda       pip install -e .[dev] (CUDA 12.1 torch)"
	@echo "test               pytest tests/ -q"
	@echo "test-cov           pytest with coverage (fail under 70%)"
	@echo "lint               ruff check src tests scripts"
	@echo "reproduce          fast CPU smoke benchmark (configs/reproduce.yaml)"
	@echo "benchmark          full 4-model benchmark (configs/benchmark.yaml)"
	@echo "sample-efficiency  Pareto study (configs/sample_efficiency.yaml)"
	@echo "multiseed          K-seed significance study (configs/multiseed.yaml)"

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

reproduce:
	$(PY) -m src.training.benchmark_experiment --config $(CONFIG_DIR)/reproduce.yaml

benchmark:
	$(PY) -m src.training.benchmark_experiment --config $(CONFIG_DIR)/benchmark.yaml

sample-efficiency:
	$(PY) -m src.evaluation.sample_efficiency_benchmark --config $(CONFIG_DIR)/sample_efficiency.yaml

multiseed:
	$(PY) -m src.evaluation.multiseed_benchmark --config $(CONFIG_DIR)/multiseed.yaml

rollout:
	$(PY) -m src.evaluation.multiseed_benchmark --config $(CONFIG_DIR)/multiseed.yaml

clean:
	rm -rf runs/ .pytest_cache/ .coverage coverage.xml
	find src tests scripts -name "__pycache__" -type d -prune -exec rm -rf {} +
