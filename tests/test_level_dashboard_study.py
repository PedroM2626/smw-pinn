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
CLEARANCE = REPO / "results" / "world_map_clearance.json"
STUDY = REPO / "results" / "level_dashboards_cleared.json"
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


def test_a_clear_never_passed_through_the_death_mode() -> None:
    """The discriminator the clearance section rests on, enforced on its own artifact.

    A game over also ends on the map, which is the definition this campaign used for exactly one
    run of itself before the byte was read properly. A cleared round is one whose mode went from
    `0x14` to the exit sequence without the death animation in between, and whose cursor moved.
    """
    clearance = _load(CLEARANCE)
    for round_record in clearance["rounds"]:
        clear = round_record.get("cleared")
        if clear is None:
            continue
        attempt = next(a for a in round_record["attempts"] if a["attempt"] == clear["attempt"])
        assert not attempt["died_on_the_way"], "a run that died is not a stage that finished"
        assert 0x0B not in [int(m, 16) for m in clear["exit_modes"]]
        assert clear["cursor_after"] != round_record["node"], (
            "back at the node it left is a game over, not a clear"
        )
    assert clearance["cleared_rounds"] == sum(
        1 for r in clearance["rounds"] if r.get("cleared") is not None
    )


def test_a_seeded_clear_reproduces_itself() -> None:
    """The headline of the section is one number; it has to survive being flown twice."""
    clearance = _load(CLEARANCE)
    assert clearance["protocol"]["seed"] is not None, "an unseeded clear is an anecdote"
    for round_record in clearance["rounds"]:
        replay = (round_record.get("cleared") or {}).get("replay")
        if replay is None:
            continue
        assert replay["progress_difference_px"] < 1e-6, (
            f"the same seed from the same state differs by {replay['progress_difference_px']} px"
        )
        assert replay["reached_the_map"], "the replay did not finish the stage"


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


def test_distinct_stages_have_distinct_buffer_profiles() -> None:
    """Two stages whose terrain profile is identical are one stage reported twice.

    10.64 settled a buffer question by comparing whole columns rather than summary counts, and the
    same trap is live here: the survey keys stages by the translevel byte, and the byte can differ
    while the buffer does not. The per-subscreen non-air profile is cheap to compare and is exactly
    the thing the progress bar's denominator is taken from, so if two stages share it the section
    must not present them as two levels.
    """
    study = _load(STUDY)
    profiles = {
        tuple(stage["terrain"]["non_air_cells_by_subscreen"]): stage["identity"]["translevel"]
        for stage in study["stages"]
    }
    assert len(profiles) == len(study["stages"]), "stages share a terrain profile: " + ", ".join(
        f"{t} -> {len(p)} subscreens" for p, t in profiles.items()
    )


def test_every_stage_has_an_episode_and_the_media_that_shows_it() -> None:
    study = _load(STUDY)
    assert study["stages"], "the study recorded no stage"
    for stage in study["stages"]:
        dataset = REPO / stage["episode"]["dataset"]
        assert dataset.is_file(), f"{dataset} is quoted by the README and not committed"
        assert float(stage["walk_right_frontier"]["frontier_px"]) > 0.0
        # a stage the campaign flew has its best attempt quoted; one no flight ever finished says
        # so in `source` instead of inventing a number
        if int(stage["mpc"]["tries"]) > 0:
            assert stage["mpc"]["best_progress_px"] is not None


def test_the_kept_episode_is_the_attempt_it_replaces() -> None:
    """The 0x2A episode is the 10.66 clearance draw itself, re-flown with its frames kept.

    The old recorder flew each episode twice (cheap, then kept) and compared; the 10.68 recorder
    flies the seeded draw once with everything kept, so the control that replaces the comparison is
    the seed: the draw that cleared 0x2A in the campaign must conclude the recorded episode too.
    The frame to compare is the engine's exit, not the episode's length - the episode keeps
    recording past the verdict now (the outro), and only the gameplay up to the exit was the
    flight.
    """
    study = _load(STUDY)
    clearance = _load(CLEARANCE)
    cleared_round = next(r for r in clearance["rounds"] if r.get("cleared"))
    attempt = next(
        a for a in cleared_round["attempts"] if a["attempt"] == cleared_round["cleared"]["attempt"]
    )
    recorded = next(
        s for s in study["stages"] if str(s["identity"]["translevel"]).lower() == "0x2a"
    )
    assert recorded["episode"]["controller"] == "mpc", (
        "the cleared stage is flown by its clearance draw"
    )
    assert recorded["episode"]["planner_seed"] == clearance["protocol"]["seed"]
    assert recorded["clear_evidence"]["concluded"], (
        "the campaign cleared this stage and the recorded episode did not - the draw is not the draw"
    )
    assert abs(recorded["episode"]["engine_exit_frame"] - attempt["frames"]) <= 1, (
        f"{recorded['episode']['engine_exit_frame']} frames to the verdict against "
        f"{attempt['frames']} flown: the same seed from the same state is a different run"
    )
    assert abs(recorded["episode"]["progress_px"] - attempt["max_progress_px"]) < 1.0, (
        "the recorded episode and the published attempt disagree about how far the draw went"
    )


def test_the_picture_line_quotes_the_recording() -> None:
    """The corner the mock puts a clearance rate in quotes the engine's verdict and the recording."""
    from src.evaluation.level_dashboard_study import bottom_line, stage_width_px

    study = _load(STUDY)
    for stage in study["stages"]:
        line = bottom_line(stage)
        assert f"{stage['episode']['progress_px']:.0f} / {stage_width_px(stage):.0f} px" in line
        evidence = stage["clear_evidence"]
        if evidence["concluded"] and str(evidence["exit_mode"]) == "0x0B":
            assert "concluded: out the door and onto the map, no life lost" in line
        elif evidence["concluded"]:
            assert "cleared: the engine ended the stage" in line
        elif int(stage["mpc"]["tries"]) > 0:
            assert f"{stage['mpc']['clears']} of {stage['mpc']['tries']} attempts" in line
        else:
            assert "no run has ever finished this session" in line
        assert line == bottom_line(stage, stage["episode"]["progress_px"])


def test_the_progress_bar_denominator_is_the_stages_own_terrain() -> None:
    """A bar scaled to a goal nobody has seen the engine confirm is a claim about the level.

    Each render records the denominator it drew, so this compares the picture against the buffer read
    rather than against a constant someone chose.
    """
    from src.evaluation.level_dashboard_study import stage_width_px

    payload = _load(STUDY_ARTIFACT)
    study = _load(STUDY)
    by_dataset = {s["episode"]["dataset"]: s for s in study["stages"]}
    assert payload["rendered"], "the study was gathered without rendering"
    for record in payload["rendered"]:
        stage = by_dataset.get(record["dataset"])
        assert stage is not None, f"{record['dataset']} is drawn but not measured"
        assert float(record["goal_px"]) == stage_width_px(stage), (
            f"{record['label']} draws a bar of {record['goal_px']} px against "
            f"{stage_width_px(stage):.0f} px of terrain"
        )


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
