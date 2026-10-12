r"""Gates for README 10.66: the multi-stage dashboard, the map survey, and the clear definition.

Three claims here are the kind that survive a careless test, so each one is re-derived from the
artifact rather than read from the prose:

* a *clear* means the engine put the cursor on a node other than the one it entered. The first
  version of the campaign called any return to the map a clear and reported four of them; the
  sequence it was actually watching (`0x0B` -> `0x0C` -> `0x0D` -> `0x0E`, same node) is a game over.
* a *stage* is what the survey's own gate says it is: interactive mode, plausible coordinates, and a
  player that advances under a held RIGHT. Not the translevel byte, which a blank level also has.
* the numbers on the picture are the numbers in the recording, per stage, including the line in the
  bottom-right corner.

CPU-only, numpy and json only; skipped when the study has not been run.

Run:  pytest tests/test_level_dashboard_study.py -q
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SURVEY = REPO / "results" / "world_map_survey.json"
PROGRESSION = REPO / "results" / "world_map_progression.json"
STUDY = REPO / "results" / "level_dashboards.json"
STUDY_ARTIFACT = REPO / "results" / "level_dashboard_study_metrics.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"{path.name} has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_readme_10_66_is_the_generated_block() -> None:
    from src.evaluation.level_dashboard_study import render_section

    payload = _load(STUDY_ARTIFACT)
    lines = render_section(payload)
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.66 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.66 "), "the generator must start at the section heading"


def test_a_clear_needs_the_cursor_to_have_moved() -> None:
    """The definition this campaign got wrong once, enforced on its own artifact."""
    progression = _load(PROGRESSION)
    for round_record in progression["rounds"]:
        cleared = round_record.get("cleared")
        if cleared is None:
            continue
        assert cleared["cursor_after"] != round_record["node"], (
            "a round that came back to the node it left is a game over, not a clear"
        )
    assert progression["cleared_rounds"] == sum(
        1 for r in progression["rounds"] if r.get("cleared") is not None
    )


def test_the_walkable_flag_is_the_gate_it_claims() -> None:
    survey = _load(SURVEY)
    for node in survey["nodes"]:
        expected = bool(
            node.get("interactive")
            and node.get("plausible")
            and float(node.get("right_response_px") or 0.0) > 6.0
        )
        assert bool(node.get("walkable")) is expected, f"node {node['node']} is misclassified"
        assert bool(node.get("level")) is bool(expected and node["timer_hundreds"] > 0)
        assert node["cursor_reached"], "a probe that did not land where it meant to proves nothing"


def test_the_survey_walked_the_cursor_rather_than_writing_it() -> None:
    """10.36.3's `warp_on_map` writes the cursor bytes; that is not a teleport, and this is the proof."""
    survey = _load(SURVEY)
    assert survey["protocol"]["cursor_write_is_a_teleport"] is False
    positions = survey["discovery"]["positions"]
    assert len(positions) >= 2, "the cursor walk found nothing to walk"
    assert all(
        len(path) == 0 or tuple(path) and all(d in ("UP", "DOWN", "LEFT", "RIGHT") for d in path)
        for path in survey["discovery"]["paths"].values()
    ), "a position must be reached by a direction sequence, not by a byte write"


def test_every_stage_has_an_episode_and_the_media_that_shows_it() -> None:
    study = _load(STUDY)
    assert study["stages"], "the study recorded no stage"
    for stage in study["stages"]:
        dataset = REPO / stage["episode"]["dataset"]
        assert dataset.is_file(), f"{dataset} is quoted by the README and not committed"
        assert float(stage["walk_right_frontier"]["frontier_px"]) > 0.0
        assert stage["mpc"]["best_progress_px"] is not None


def test_the_kept_episode_is_the_attempt_it_replaces() -> None:
    """The recorder keeps frames only for the selected run, and re-runs it. Both must agree."""
    study = _load(STUDY)
    for stage in study["stages"]:
        assert stage["episode"]["repeats_the_measured_attempt_px"] < 1e-4, (
            "the actor and the console were supposed to be deterministic; "
            f"{stage['identity']['translevel']} disagrees by "
            f"{stage['episode']['repeats_the_measured_attempt_px']} px"
        )


def test_the_picture_line_quotes_the_recording() -> None:
    from src.evaluation.level_dashboard_study import bottom_line, furthest_measured_px

    study = _load(STUDY)
    for stage in study["stages"]:
        line = bottom_line(stage)
        assert f"{stage['episode']['progress_px']:.0f} px" in line
        assert f"{furthest_measured_px(stage):.0f} px" in line


def test_the_rendered_stage_media_exists_and_is_a_readme_can_carry() -> None:
    payload = _load(STUDY_ARTIFACT)
    rendered = payload.get("rendered") or []
    assert rendered, "the study was gathered without rendering"
    for record in rendered:
        for path in (record["video"]["mp4"], record["gif"]["gif"], record["still"]["png"]):
            file = Path(path) if Path(path).is_absolute() else REPO / path
            assert file.is_file(), f"{file} is referenced and missing"
            assert file.stat().st_size < 10_000_000, (
                f"{file} is over GitHub's per-file comfort limit"
            )
