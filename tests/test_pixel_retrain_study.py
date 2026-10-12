r"""Gates for README 10.65: the 2x2 retraining, the parity cell, and the pairing helper.

The section corrects a published conclusion, so the gates are arranged to catch the ways that could
be wrong for the wrong reason. The parity cell must equal 10.31's own artifact exactly - if the
protocol had drifted, the comparison would be between two different experiments. The published
checkpoint and metrics file must be untouched on disk. And `stack_consecutive_frames` is checked
against the one bug that would quietly invalidate every two-frame number: pairing a frame with an
unrelated one after the split has already shuffled the order.

CPU-only, small models; the study artifact itself is skipped when the retraining has not been run.

Run:  pytest tests/test_pixel_retrain_study.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.pixel_retrain_study import render_section
from src.perception.pixel_encoder import PixelStateEstimator
from src.perception.vision_dataset import stack_consecutive_frames
from src.utils.paths import REPO_ROOT, RESULTS_DIR

ARTIFACT = REPO_ROOT / "results" / "pixel_retrain_metrics.json"
PUBLISHED_METRICS = REPO_ROOT / "results" / "pixel_estimator_metrics.json"
PUBLISHED_CKPT = REPO_ROOT / "results" / "checkpoints" / "pixel_estimator_best.pt"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the pixel retraining study has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_10_65_is_the_generated_block() -> None:
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.65 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.65 "), "the generator must start at the section heading"


def test_the_parity_cell_is_the_published_run_and_nothing_else() -> None:
    """If this drifts, the 2x2 is comparing two protocols and the section is void."""
    published = json.loads(PUBLISHED_METRICS.read_text(encoding="utf-8"))
    cell = _payload()["cells"]["published_single"]
    assert cell["best_val_loss"] == published["best_val_loss"]
    for channel in ("x", "y", "vx", "vy"):
        assert cell["per_variable"][channel] == published["per_variable"][channel]
    assert cell["samples"] == published["_meta"]["samples"]


def test_the_published_pixel_artifacts_were_not_rewritten() -> None:
    import subprocess

    for path in (PUBLISHED_METRICS, PUBLISHED_CKPT):
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.stdout.strip() == "", f"{path.name} is modified: {proc.stdout}"


def test_the_new_cells_are_new_files() -> None:
    payload = _payload()
    for name, cell in payload["cells"].items():
        if name == "published_single":
            continue
        assert cell["checkpoint"].startswith("pixel_estimator_retrain_")
        assert (Path(RESULTS_DIR) / "checkpoints" / cell["checkpoint"]).is_file(), (
            f"{cell['checkpoint']} is named by the artifact and not on disk"
        )
        assert (Path(RESULTS_DIR) / cell["metrics_file"]).is_file()


def test_the_direction_claims_are_re_derived_from_the_cells() -> None:
    payload = _payload()
    verdict = payload["verdict"]
    cells = payload["cells"]

    def best_velocity(cell: str) -> float:
        per = cells[cell]["per_variable"]
        return max(float(per[c]["r2"]) for c in ("vx", "vy"))

    assert verdict["strong_form_survives"] == (best_velocity("level_single") < 0.5)
    assert verdict["two_frames_beat_one"] == (
        best_velocity("level_pair") > best_velocity("level_single") + 0.1
    )
    assert verdict["two_frame_gain_vx"] == pytest.approx(
        cells["level_pair"]["per_variable"]["vx"]["r2"]
        - cells["level_single"]["per_variable"]["vx"]["r2"],
        abs=1e-9,
    )
    assert verdict["two_frame_gain_vy"] > 0.0
    assert verdict["one_frame_gain_over_menu_vx"] > 0.0, (
        "the section says a scrolling frame carries more speed information than a menu frame"
    )


def test_stacking_pairs_each_frame_with_its_predecessor_in_the_original_order() -> None:
    frames = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
    stacked = stack_consecutive_frames(frames)
    assert stacked.shape == (6, 2, 2, 6)
    assert np.array_equal(stacked[3][..., :3], frames[2]), "the earlier frame must come first"
    assert np.array_equal(stacked[3][..., 3:], frames[3])
    assert np.array_equal(stacked[0][..., :3], frames[0]), "frame 0 is paired with itself"


def test_stacking_rejects_an_already_stacked_batch() -> None:
    with pytest.raises(ValueError):
        stack_consecutive_frames(np.zeros((4, 2, 2, 6), dtype=np.uint8))


def test_the_widened_first_convolution_is_the_only_change() -> None:
    """The published architecture must still load its own checkpoint, so nothing else may differ."""
    single = PixelStateEstimator(state_dim=8, base_channels=4)
    pair = PixelStateEstimator(state_dim=8, base_channels=4, in_channels=6)
    assert single.features[0].in_channels == 3
    assert pair.features[0].in_channels == 6
    assert single.features[0].kernel_size == pair.features[0].kernel_size
    shapes_single = {n: tuple(p.shape) for n, p in single.named_parameters()}
    shapes_pair = {n: tuple(p.shape) for n, p in pair.named_parameters()}
    assert set(shapes_single) == set(shapes_pair)
    differing = {n for n in shapes_single if shapes_single[n] != shapes_pair[n]}
    assert differing == {"features.0.weight"}, (
        f"widening the input also changed {sorted(differing)}"
    )
    assert single(torch.zeros(1, 3, 16, 16)).shape == (1, 8)
    assert pair(torch.zeros(1, 6, 16, 16)).shape == (1, 8)
