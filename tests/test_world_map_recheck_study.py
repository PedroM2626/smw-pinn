r"""Gates for README 10.67: the world-map survey re-run two ways.

Three claims here are the kind a prose paragraph can make and an artifact can deny, so each is
re-derived from the survey files rather than read from the section:

* the survey *repeats*. The replay of the published route must return the same probe records and, now
  that every capture is hashed, the same console bytes as the savestates 10.66 published. A hash that
  moved would mean the section's "reproduced exactly" is about a run nobody can get again.
* the map a clear opens has no new stage. The opened route's level numbers must be the stage list the
  recorder actually built its episodes from, and the position the published route cannot reach must be
  the same session as a position it can.
* what does change with the route is the bytes. The two routes may not be quoted as having captured
  one state at a node, and the distances that differ are the ones the section prints.

CPU-only, json, hashlib and pathlib. Skipped when the study has not been run.

Run:  pytest tests/test_world_map_recheck_study.py -q
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PUBLISHED = RESULTS / "world_map_survey.json"
REPLAY = RESULTS / "world_map_survey_replay.json"
OPENED = RESULTS / "world_map_survey_opened.json"
STAGES = RESULTS / "level_dashboards_cleared.json"
METRICS = RESULTS / "world_map_recheck_metrics.json"

#: the fields the section says a second route must reproduce; `path` is a property of the map each
#: route starts from, so it is compared only between runs of the same route.
FIELDS = (
    "cursor_before_A",
    "cursor_reached",
    "mode_sequence",
    "entry_frame",
    "translevel",
    "timer_hundreds",
    "interactive",
    "x_after_settle",
    "y_after_settle",
    "right_response_px",
    "plausible",
    "walkable",
    "level",
)


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        pytest.skip(f"{path.name} has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


def _nodes(survey: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    return {(int(n["node"][0]), int(n["node"][1])): n for n in survey["nodes"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved(name: str) -> Path:
    return REPO / "data" / "raw" / name


def test_readme_10_67_is_the_generated_block() -> None:
    from src.evaluation.world_map_recheck_study import render_section

    payload = _load(METRICS)
    lines = render_section(payload)
    text = (REPO / "README.md").read_text(encoding="utf-8")
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.67 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.67 "), "the generator must start at the section heading"


def test_the_replay_reproduces_every_probe_record_of_the_published_route() -> None:
    """Same route, same answer - the survey is not a run that happened once."""
    published, replay = _load(PUBLISHED), _load(REPLAY)
    a, b = _nodes(published), _nodes(replay)
    assert set(a) == set(b), "the replay reached different cursor positions than the published run"
    assert sorted(a) == sorted(b)
    moved = [
        f"{node}.{field}"
        for node in a
        for field in FIELDS
        if a[node].get(field) != b[node].get(field)
    ]
    assert not moved, f"the replay did not reproduce: {moved}"
    assert [n["path"] for n in published["nodes"]] == [n["path"] for n in replay["nodes"]]


def test_the_published_savestates_are_the_bytes_a_replay_writes() -> None:
    """The strongest form of the reproducibility claim: the files themselves.

    The replay captured under its own prefix and those files are not kept, so the comparison is
    between the digest it recorded and the bytes of the state the published survey named.
    """
    published, replay = _load(PUBLISHED), _load(REPLAY)
    a, b = _nodes(published), _nodes(replay)
    checked = []
    for node, record in b.items():
        digest = (record.get("capture") or {}).get("sha256")
        target = a.get(node, {}).get("capture", {}).get("saved")
        if not digest or not target:
            continue
        path = _saved(target)
        assert path.is_file(), f"{target} is named by the published survey but not on disk"
        checked.append((node, digest, _sha(path)))
    assert len(checked) == len(a), f"only {len(checked)} of {len(a)} captures could be compared"
    mismatched = [node for node, recorded, actual in checked if recorded != actual]
    assert not mismatched, f"a fresh survey run writes different bytes at {mismatched}"


def test_the_byte_identity_check_fires_on_bytes_that_moved() -> None:
    """Positive control: the gate above is not a digest comparison that always passes."""
    from src.evaluation.world_map_recheck_study import byte_identity

    published, replay = _load(PUBLISHED), _load(REPLAY)
    honest = byte_identity(replay, published)
    assert honest["states_checked"] and not honest["mismatched_nodes"]
    tampered = json.loads(json.dumps(replay))
    first = next(n for n in tampered["nodes"] if (n.get("capture") or {}).get("sha256"))
    first["capture"]["sha256"] = "0" * 64
    caught = byte_identity(tampered, published)
    assert len(caught["mismatched_nodes"]) == 1, "a wrong digest must show up as a mismatch"
    assert caught["states_byte_identical"] == honest["states_byte_identical"] - 1


def test_the_two_routes_never_capture_the_same_console_state() -> None:
    """The finding the section is named for: entering a node is not the same as being at it."""
    replay, opened = _load(REPLAY), _load(OPENED)
    a, b = _nodes(replay), _nodes(opened)
    shared = sorted(set(a) & set(b))
    assert shared, "the two survey routes were expected to enter some of the same nodes"
    same = [
        node
        for node in shared
        if (a[node].get("capture") or {}).get("sha256")
        and (a[node].get("capture") or {}).get("sha256") == (b[node].get("capture") or {}).get(
            "sha256"
        )
    ]
    assert not same, f"the two routes captured identical bytes at {same}"


def test_the_position_the_clear_opened_is_a_path_tile_of_a_stage_already_counted() -> None:
    """(115,126) was the one node 10.66 left unanswered: walkable, entered, never classified."""
    published, opened = _load(PUBLISHED), _load(OPENED)
    record = _nodes(opened).get((115, 126))
    assert record is not None, "the opened route did not reach (115, 126)"
    twin = _nodes(published).get((116, 126))
    assert twin is not None, "the published route lost the position (115, 126) is compared with"
    assert record["translevel"] == twin["translevel"] == "0x28"
    assert record["timer_hundreds"] == twin["timer_hundreds"] == 0
    assert record.get("walkable") is True and record.get("level") is False
    assert (record.get("capture") or {}).get("x_at_entry") == (
        twin.get("capture") or {}
    ).get("x_at_entry")
    assert (record.get("terrain") or {}).get("terrain_px") == 256


def test_the_opened_map_adds_no_level_number_the_recorder_did_not_already_record() -> None:
    """The stage count is a property of the map, not of what has been cleared."""
    opened, stages = _load(OPENED), _load(STAGES)
    reached = sorted({str(n["translevel"]) for n in opened["nodes"] if n.get("walkable")})
    recorded = sorted(str(s["identity"]["translevel"]) for s in stages["stages"])
    assert reached == recorded, f"{reached} against the stages recorded {recorded}"


def test_the_walk_distances_the_section_prints_are_the_ones_the_probes_measured() -> None:
    """The route dependence is quoted as two numbers per node, and both are pinned."""
    published, opened = _load(PUBLISHED), _load(OPENED)
    a, b = _nodes(published), _nodes(opened)
    moved = {
        node: (a[node]["right_response_px"], b[node]["right_response_px"])
        for node in set(a) & set(b)
        if a[node].get("right_response_px") != b[node].get("right_response_px")
    }
    assert moved == {(56, 136): (15.8125, 17.0625), (68, 135): (15.8125, 17.0625)}, moved
    for first, second in moved.values():
        assert first > 6.0 and second > 6.0, "both routes still clear the walkability gate"
    text = (REPO / "README.md").read_text(encoding="utf-8")
    for first, second in moved.values():
        assert f"{first} against {second}" in text, "the pair must be printed as measured"


def test_only_the_published_survey_keeps_the_states_it_names() -> None:
    """The section says the replay and opened runs name states that are not in the repository."""
    published, replay, opened = _load(PUBLISHED), _load(REPLAY), _load(OPENED)
    kept = {
        name: sum(
            1
            for n in survey["nodes"]
            if (n.get("capture") or {}).get("saved")
            and _saved(n["capture"]["saved"]).is_file()
        )
        for name, survey in (
            ("published", published),
            ("replay", replay),
            ("opened", opened),
        )
    }
    metrics = _load(METRICS)
    assert kept == metrics["protocol"]["capture_files_kept"]
    assert kept["published"] == len(published["nodes"])
    assert kept["replay"] == 0 and kept["opened"] == 0
    assert "0 of them are on disk" in (REPO / "README.md").read_text(encoding="utf-8")


def test_the_replay_reports_no_field_difference_and_the_opened_route_only_three() -> None:
    """The two counts the prose leans on, recomputed rather than read from the payload."""
    published, replay, opened = _load(PUBLISHED), _load(REPLAY), _load(OPENED)
    a, b, c = _nodes(published), _nodes(replay), _nodes(opened)
    replay_moves = [
        (node, field)
        for node in set(a) & set(b)
        for field in FIELDS + ("path",)
        if a[node].get(field) != b[node].get(field)
    ]
    assert not replay_moves
    opened_moves = [
        (node, field)
        for node in set(a) & set(c)
        for field in FIELDS
        if a[node].get(field) != c[node].get(field)
    ]
    assert sorted(opened_moves) == [
        ((56, 136), "right_response_px"),
        ((68, 135), "right_response_px"),
        ((152, 136), "entry_frame"),
    ]
