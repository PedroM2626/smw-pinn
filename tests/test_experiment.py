"""
test_experiment.py
Unit tests for the lightweight experiment tracker (src/utils/experiment.py).
Runs without torch or TensorBoard: file-based logging must always work.
"""

import json
import os

from src.utils.experiment import ExperimentLogger


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_logger_creates_run_dir_and_hparams(tmp_path):
    logger = ExperimentLogger(
        experiment_name="unit_test",
        log_dir=str(tmp_path),
        hparams={"lr": 1e-3, "seed": 42},
        use_tensorboard=False,
    )
    try:
        assert os.path.isdir(logger.dir)
        with open(os.path.join(logger.dir, "hparams.json"), encoding="utf-8") as f:
            stored = json.load(f)
        assert stored["lr"] == 1e-3
        assert stored["seed"] == 42
        assert os.path.exists(os.path.join(logger.dir, "metrics.jsonl"))
    finally:
        logger.close()


def test_log_metrics_appends_jsonl_rows(tmp_path):
    logger = ExperimentLogger(
        experiment_name="unit_test",
        log_dir=str(tmp_path),
        use_tensorboard=False,
    )
    try:
        logger.log_metrics({"mlp/train_loss": 0.5, "mlp/val_loss": 0.4}, step=1)
        logger.log_metrics({"mlp/train_loss": 0.3, "mlp/val_loss": 0.25}, step=2)
        rows = _read_jsonl(os.path.join(logger.dir, "metrics.jsonl"))
        assert len(rows) == 2
        assert rows[0]["step"] == 1
        assert rows[1]["mlp/train_loss"] == 0.3
    finally:
        logger.close()


def test_log_hparams_merges(tmp_path):
    logger = ExperimentLogger(
        experiment_name="unit_test",
        log_dir=str(tmp_path),
        hparams={"seed": 1},
        use_tensorboard=False,
    )
    try:
        logger.log_hparams({"epochs": 10})
        with open(os.path.join(logger.dir, "hparams.json"), encoding="utf-8") as f:
            stored = json.load(f)
        assert stored["seed"] == 1
        assert stored["epochs"] == 10
    finally:
        logger.close()


def test_context_manager_closes(tmp_path):
    with ExperimentLogger(
        experiment_name="unit_test",
        log_dir=str(tmp_path),
        use_tensorboard=False,
    ) as logger:
        logger.log_metrics({"loss": 1.0}, step=0)
    rows = _read_jsonl(os.path.join(logger.dir, "metrics.jsonl"))
    assert rows[0]["loss"] == 1.0


def test_tensorboard_missing_falls_back_to_jsonl(tmp_path):
    # tensorboard is optional: requesting it without the package installed
    # must degrade gracefully to file-only logging (or use it if present).
    logger = ExperimentLogger(
        experiment_name="unit_test",
        log_dir=str(tmp_path),
        use_tensorboard=True,
    )
    try:
        logger.log_metrics({"loss": 2.0}, step=0)
        rows = _read_jsonl(os.path.join(logger.dir, "metrics.jsonl"))
        assert rows[0]["loss"] == 2.0
    finally:
        logger.close()


def test_wandb_missing_disables_gracefully(tmp_path):
    logger = ExperimentLogger(
        experiment_name="unit_test",
        log_dir=str(tmp_path),
        use_tensorboard=False,
        use_wandb=True,
        wandb_project="smw-pinn-unit-test",
    )
    try:
        logger.log_metrics({"loss": 3.0}, step=0)
        rows = _read_jsonl(os.path.join(logger.dir, "metrics.jsonl"))
        assert rows[0]["loss"] == 3.0
    finally:
        logger.close()
