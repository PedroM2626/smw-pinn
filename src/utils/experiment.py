"""
experiment.py
Lightweight experiment tracking: TensorBoard + JSONL metrics + optional wandb.

Design goals:
- Zero hard dependency beyond `tensorboard` (already in requirements.txt).
- Graceful degradation: if TensorBoard/wandb are missing or disabled,
  metrics still land in `metrics.jsonl` + `hparams.json` on disk.
- Backward compatible: every existing training script keeps working
  unchanged; passing an `ExperimentLogger` only adds logging.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentLogger:
    """Unified logger for dynamics training and MBRL benchmarks.

    Directory layout per experiment::

        runs/<experiment_name>_<timestamp>/
            hparams.json
            metrics.jsonl
            tensorboard/          # TensorBoard event files (if available)

    Example:
        logger = ExperimentLogger("benchmark_mlp_vs_pinn", hparams={...})
        logger.log_metrics({"train/loss": 0.5}, step=1)
        logger.close()
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        log_dir: str = "runs",
        hparams: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.run_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.hparams = dict(hparams or {})
        with open(os.path.join(self.run_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(self.hparams, f, indent=2, default=str)

        self._metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        # Ensure the file exists so tail/watchers work from step 0.
        open(self._metrics_path, "a", encoding="utf-8").close()

        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = os.path.join(self.run_dir, "tensorboard")
                self._writer = SummaryWriter(log_dir=tb_dir)
            except Exception as exc:  # tensorboard missing/broken -> file logging only
                logger.info(f"[ExperimentLogger] TensorBoard disabled ({exc}); using JSONL only.")

        self._wandb_run = None
        if use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project or "smw-pinn",
                    name=f"{experiment_name}_{timestamp}",
                    config=self.hparams,
                )
            except Exception as exc:
                logger.info(f"[ExperimentLogger] wandb disabled ({exc}); using local logs only.")
                self._wandb_run = None

        logger.info(f"[ExperimentLogger] Logging to {self.run_dir}")

    @property
    def dir(self) -> str:
        return self.run_dir

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Append one row of scalar metrics (JSONL) + TensorBoard/wandb mirrors."""
        row = {"step": int(step)}
        for key, value in metrics.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

        if self._writer is not None:
            for key, value in row.items():
                if key == "step":
                    continue
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, float(value), int(step))

        if self._wandb_run is not None:
            try:
                self._wandb_run.log(row)
            except Exception:
                pass

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        """Merge extra hyperparameters into the stored hparams.json."""
        self.hparams.update(hparams)
        with open(os.path.join(self.run_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(self.hparams, f, indent=2, default=str)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
