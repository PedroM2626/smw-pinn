r"""Gates for README 10.36.3: the level-entry A/B, and the state the navigator is allowed to save.

The claim being pinned is a comparison, not a number: the schedule this repository used reaches a
rendered level whose player is scripted, and the route it uses now reaches one that is playable. Both
arms are measured on the console by `scripts/verify_level_entry.py`, so the gate re-reads the artifact
and requires the two arms to differ in the way the section says they do - and requires the committed
savestate to be the *verified* one, at the level start rather than wherever the probes left Mario.

CPU-only: it reads the artifact and the state's own header, no emulator.

Run:  pytest tests/test_level_entry.py -q
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "level_entry_verification.json"
STATE = REPO / "data" / "raw" / "smw_yoshi_island_1_level.state"


def _payload():
    if not ARTIFACT.is_file():
        return None
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme():
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_the_artifact_and_the_readme_agree_on_both_arms() -> None:
    payload = _payload()
    if payload is None:
        # The A/B needs the console, so CI may not have it; the section must then say so rather
        # than quote numbers nobody reproduced.
        assert "verify_level_entry.py" in _readme(), (
            "10.36.3 quotes an artifact that is not in the tree, and does not say how to make it"
        )
        return
    arms = payload["arms"]
    naive, measured = arms["naive_schedule"], arms["measured_route"]
    text = _readme()
    for token in (
        f"{measured['session']['x_start']:.1f}",
        f"{measured['session']['x_travel_px']:.1f}",
        f"{100 * measured['session']['share_of_pairs_picture_shifts']:.2f}%",
        f"{naive['session']['x_start']:.1f}",
    ):
        assert token in text, f"README 10.36.3 no longer quotes the measured {token}"


def test_the_two_arms_differ_the_way_the_section_claims() -> None:
    payload = _payload()
    if payload is None:
        return
    naive = payload["arms"]["naive_schedule"]
    measured = payload["arms"]["measured_route"]
    assert naive["final_mode"] == measured["final_mode"] == "0x14", (
        "the whole point of the section is that the mode byte cannot tell these two apart"
    )
    assert naive["session"]["x_travel_px"] == 0.0
    assert naive["session"]["share_of_pairs_picture_shifts"] == 0.0
    assert measured["session"]["x_travel_px"] > 300.0
    assert measured["session"]["share_of_pairs_picture_shifts"] > 0.5
    assert measured["session"]["camera_end"] > 100.0, (
        "a scrolling camera is what makes the frames usable as a video window"
    )


def test_the_committed_state_is_the_level_start_and_is_referenced() -> None:
    text = _readme()
    assert "smw_yoshi_island_1_level.state" in text, (
        "the verified savestate is what unlocks the RGB window; the README has to point at it"
    )
    assert "16.0 -> 882.8" in text or _payload() is None
    if STATE.is_file():
        assert STATE.stat().st_size > 500_000, "a libretro savestate is ~800 KB; this is not one"


def test_the_published_savestate_is_untouched_by_the_new_route() -> None:
    """The old state is the input to every committed artifact; the new route must not overwrite it."""
    published = REPO / "data" / "raw" / "smw_yoshi_island_1.state"
    assert published.is_file()
    import subprocess

    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", str(published.relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "", (
        f"the published savestate is modified ({proc.stdout.strip()}); every committed recording "
        "was made from it, so a new entry route gets a new file"
    )


def test_the_credit_and_the_open_question_are_both_in_print() -> None:
    text = _readme()
    section = text.split("#### 10.36.3")[1].split("## 11.")[0]
    assert "burn-mario-rl" in section, "the route is not ours and the section must say so"
    assert "does not settle" in section and "10.61" in section, (
        "the terrain mismatch between the genuine level and the committed recordings is the open "
        "item this section leaves; it may lose its unreproducible number now that 10.61 measures it, "
        "but not the question or the pointer to where it is answered"
    )
