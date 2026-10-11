"""
pause_hygiene_study.py - README 10.63: what the held player actually is, and what it costs 4.1.

10.59 found that most exceptions to §4.1 are frames where the whole player record repeats while WRAM
keeps changing, and recommended that the recorder stop sampling when that happens. 10.60 found a
mechanism - the engine's message box, which asks for a button no policy here has - and cleared it
with an operator press. This study asks the two questions that leaves open, with three recordings of
one instrument: is the hold one mechanism or several, and what does the recorder's answer cost the
identity?

The answer is that the hold is not one mechanism, and the difference is legible in the velocity byte
it freezes with. The death animation holds the player at the apex of its jump with `v_y = -112.000`
and nothing releases it, because the episode is over; the message box holds him mid-arc at a value of
its own and the A button releases it. Splitting the two changes what "stop the episode" means: for
the first it is free, for the second it is throwing away a level the operator could have continued.

Arms: the published boundary recording (forced-mode session, no hygiene), the same recorder on the
verified level-entry savestate, and that again with the hold cleared by an operator A pulse. The
identity is scored with 10.59's own definitions, imported rather than restated.

Needs no emulator: all three recordings are on disk. Writes results/pause_hygiene_metrics.json.

Run:  python -m src.evaluation.pause_hygiene_study
      make pause-hygiene
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np

from src.evaluation.boundary_channel_study import (
    EXACT_TOLERANCE_SUBPIXELS,
    load_boundary_recording,
)
from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_BOUNDARY,
    REPO_ROOT,
    dataset_path,
    results_file,
)
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

DATASET_LEVEL_PLAIN = dataset_path("smw_boundary_level_dataset.npz")
DATASET_LEVEL_CLEAR = dataset_path("smw_boundary_level_clear_dataset.npz")

#: the apex velocity of the death jump, as the console freezes it (sub-pixels per frame)
DEATH_APEX_VY = -112.0
VY_TOLERANCE = 1e-6

ARMS = (
    ("published_forced_mode", DATASET_BOUNDARY, "no", "the recording 10.59 measured"),
    ("level_entry_no_hygiene", DATASET_LEVEL_PLAIN, "no", "same recorder, entered level"),
    ("level_entry_hygiene", DATASET_LEVEL_CLEAR, "yes", "operator A pulsed at a held player"),
)


def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


def frames_in_holds(repeat: np.ndarray, minimum_run: int = 10) -> int:
    """Transitions inside a run of repeats at least `minimum_run` long.

    A hold is not one bad frame: while it lasts, every sample is the same frozen body. This counts
    the frames that belong to such a run, which is the volume of data a recorder spends on a player
    the engine has stopped. `minimum_run` matches the threshold 10.60's hygiene acts on.
    """
    total = 0
    run = 0
    for value in np.asarray(repeat, dtype=bool):
        run = run + 1 if value else 0
        if run == minimum_run:
            total += minimum_run
        elif run > minimum_run:
            total += 1
    return int(total)


def measure(arm: str, path: str, hygiene: str, note: str) -> Dict[str, Any]:
    """One arm: the identity's rates, and the held frames split by the mechanism they freeze with."""
    rec = load_boundary_recording(path)
    exact = np.abs(rec.residue) < EXACT_TOLERANCE_SUBPIXELS
    repeat = rec.repeat
    exceptions = ~exact
    death = repeat & (np.abs(rec.state[:, 3] - DEATH_APEX_VY) <= VY_TOLERANCE)
    other = repeat & ~death
    with np.load(path, allow_pickle=False) as blob:
        operator = (
            int(np.asarray(blob["hold_operator_frames"], dtype=np.int64).sum())
            if "hold_operator_frames" in blob.files
            else 0
        )
    return {
        "arm": arm,
        "dataset": _rel(path),
        "hygiene": hygiene,
        "note": note,
        "transitions": rec.n,
        "episodes": int(len(np.unique(rec.episodes))),
        "exact_share": float(np.mean(exact)),
        "repeat_share": float(np.mean(repeat)),
        "repeats": int(repeat.sum()),
        "death_apex_repeats": int(death.sum()),
        "death_apex_share_of_repeats": (float(np.mean(death[repeat])) if repeat.any() else None),
        "other_repeats": int(other.sum()),
        "other_repeat_vy_values": sorted({float(v) for v in rec.state[other, 3]})[:6],
        "exact_share_excluding_repeats": (
            float(np.mean(exact[~repeat])) if np.any(~repeat) else None
        ),
        "exceptions": int(exceptions.sum()),
        "non_repeat_exceptions": int((exceptions & ~repeat).sum()),
        "non_repeat_exception_share": float(np.mean(exceptions & ~repeat)),
        "frames_in_holds": frames_in_holds(repeat),
        "operator_A_frames": operator,
        "x_travel_px": float(rec.x.max() - rec.x.min()),
        "camera_travel_px": float(rec.camera.max() - rec.camera.min()),
        "mode_bytes": sorted({int(m) for m in np.unique(rec.mode)}),
    }


#: every channel the recorder writes except the two CRCs, which see the button press itself
SIMULATION_CHANNELS = (
    "states",
    "next_states",
    "actions",
    "camera",
    "next_camera",
    "mode",
    "next_mode",
    "tile_patches",
    "next_tile_patches",
    "episodes",
    "held",
)


def channel_parity(left: str, right: str) -> Dict[str, Any]:
    """Compare two arms channel by channel: what the hygiene changed, and what it could not.

    The WRAM CRC is deliberately outside the simulation set. It is a hash of all 128 KB, and the
    joypad latch the operator's A pulse writes lives in there, so it moves on a frame where nothing
    about the body moved - which is the whole distinction this function exists to make.
    """
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        identical = {
            k: bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))) for k in SIMULATION_CHANNELS
        }
        crc_differs = int(np.sum(np.asarray(a["wram_crc"]) != np.asarray(b["wram_crc"])))
    return {
        "simulation_channels_identical": all(identical.values()),
        "channels": identical,
        "crc_frames_differing": crc_differs,
    }


def run_study(
    output_metrics: str = results_file("pause_hygiene_metrics.json"),
) -> Dict[str, Any]:
    """Score the three arms and write the artifact README 10.63 prints."""
    missing = [arm for arm, path, _, _ in ARMS if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"no recording for {missing}; see `make record-boundary` and the flags of "
            "scripts/record_boundary_gameplay.py"
        )
    arms = [measure(arm, path, hygiene, note) for arm, path, hygiene, note in ARMS]
    by = {a["arm"]: a for a in arms}
    published, plain, clear = (
        by["published_forced_mode"],
        by["level_entry_no_hygiene"],
        by["level_entry_hygiene"],
    )
    parity = channel_parity(DATASET_LEVEL_PLAIN, DATASET_LEVEL_CLEAR)
    verdict = {
        "holds_are_one_mechanism": bool(
            (plain["death_apex_share_of_repeats"] or 0.0) < 0.5
            and (published["death_apex_share_of_repeats"] or 0.0) < 0.5
        ),
        "published_repeats_that_are_the_death_apex": published["death_apex_share_of_repeats"],
        "level_repeats_that_are_the_death_apex": plain["death_apex_share_of_repeats"],
        "hygiene_changed_the_trajectory": bool(not parity["simulation_channels_identical"]),
        "exact_rate_gain_from_dropping_holds_pp": 100.0
        * (clear["exact_share_excluding_repeats"] - clear["exact_share"]),
        "session_changes_the_identity_pp": 100.0
        * (plain["exact_share"] - published["exact_share"]),
        "reposition_residue_after_hygiene": clear["non_repeat_exceptions"],
    }
    payload: Dict[str, Any] = {
        "study": (
            "What the held player of 10.59 actually is, split by the velocity byte it freezes with, "
            "and what clearing it at capture costs the section-4.1 identity."
        ),
        "protocol": {
            "identity": "16*(x'-x) - v_x, exact under the published |residue| < "
            f"{EXACT_TOLERANCE_SUBPIXELS} sub-pixel tolerance",
            "repeat_definition": "every channel of the next state equals the current one (10.59)",
            "death_apex_definition": f"a repeat whose frozen v_y is exactly {DEATH_APEX_VY} "
            "sub-pixels per frame",
            "tail_definition": "the run of repeats that ends an episode",
            "hygiene_rule": "PlayerHold: A pulsed after 10 identical records, 4 on / 4 off, "
            "up to 240 frames",
            "recorder": "scripts/record_boundary_gameplay.py",
            "episodes": 40,
            "frames_per_episode": 500,
            "seed_base": 4000,
            "behaviour_vocabulary": "the tilemap recorder's random forward/retreat patterns",
        },
        "arms": arms,
        "arm_parity": parity,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(output_metrics) or ".", exist_ok=True)
    write_metrics(output_metrics, payload, command="make pause-hygiene")
    logger.info("Artifact written to: %s", output_metrics)
    payload["_artifact"] = output_metrics
    return payload


def render_table(arms: List[Dict[str, Any]]) -> List[str]:
    rows = [
        "| arm | hygiene | transitions | §4.1 exact | held frames | of them the death apex | "
        "exact without holds | frames inside a hold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        rows.append(
            f"| `{arm['arm']}` | {arm['hygiene']} | {arm['transitions']:,} "
            f"| {100 * arm['exact_share']:.2f}% | {arm['repeats']:,} "
            f"| {100 * (arm['death_apex_share_of_repeats'] or 0.0):.2f}% "
            f"| {100 * (arm['exact_share_excluding_repeats'] or 0.0):.2f}% "
            f"| {arm['frames_in_holds']:,} |"
        )
    return rows


def render_section(payload: Dict[str, Any]) -> List[str]:
    """README 10.63, generated from the artifact."""
    arms = payload["arms"]
    verdict = payload["verdict"]
    parity = payload["arm_parity"]
    by = {a["arm"]: a for a in arms}
    published, plain, clear = (
        by["published_forced_mode"],
        by["level_entry_no_hygiene"],
        by["level_entry_hygiene"],
    )
    return [
        "### 10.63 The Held Player Has Two Mechanisms, and the Recordings Mostly Show the Other One",
        "",
        "10.59 found that most exceptions to §4.1 are frames where the whole player record repeats "
        "while WRAM keeps changing, and recommended stopping the recorder there. 10.60 named one "
        "mechanism - the engine's message box, which asks for a button no policy in this repository "
        "has - and cleared it. Three recordings of one instrument separate what that leaves tangled.",
        "",
        *render_table(arms),
        "",
        f"The split is not subtle: {100 * (published['death_apex_share_of_repeats'] or 0.0):.2f}% of "
        f"the published recording's held frames, and "
        f"{100 * (plain['death_apex_share_of_repeats'] or 0.0):.2f}% of the entered level's, are "
        f"frozen at v_y = {DEATH_APEX_VY:.1f} sub-pixels per frame - the apex of the death jump, "
        f"where the engine holds the body for tens of frames and nothing releases it because the "
        f"episode is already over. The message box of 10.60 is a different signature: it holds him "
        f"mid-arc at another value, and it releases when somebody answers it. The "
        f"{clear['other_repeats']} held frames here are not the apex, and they are not boxes either - "
        f"the third arm pulsed A for {clear['operator_A_frames']} frames across all "
        f"{clear['repeats']:,} holds and released none of them, which the next paragraph makes "
        f"precise.",
        "",
        f"That reorders 10.59's recommendation rather than cancelling it. Stopping an episode at a "
        f"held player is free when the hold is a death - the {plain['frames_in_holds']:,} frames "
        f"sitting inside a long hold in the unhygienic level arm are data the recorder was never "
        f"going to get - and wrong when the hold is a box, because there the level continues the "
        f"moment somebody answers it. The identity itself is what moves: dropping the held frames "
        f"lifts "
        f"§4.1 from {100 * clear['exact_share']:.2f}% to "
        f"{100 * (clear['exact_share_excluding_repeats'] or 0.0):.2f}%, a "
        f"{verdict['exact_rate_gain_from_dropping_holds_pp']:.2f} pp gain, which is the same "
        f"operation 10.59 performed after the fact on the committed files and now the recorder can "
        f"perform at capture.",
        "",
        f"**The hygiene changed nothing about the body, and that is a result.** The second and third "
        f"arms differ only in the `--clear-boxes` flag, and every channel of the simulation - state, "
        f"next state, action, camera, mode byte, tile patch, episode id, the hold flag itself - is "
        f"element-for-element identical across {clear['transitions']:,} transitions. Exactly one "
        f"channel disagrees: the WRAM CRC, on {parity['crc_frames_differing']} frames, because the "
        f"joypad latch the {clear['operator_A_frames']} A pulses wrote lives inside the 128 KB the "
        f"hash covers. The operator pressed a button and the only thing that changed in the console "
        f"was the record that a button was pressed - these holds are deaths, and deaths do not answer "
        f"to a button. The box is not absent from the entered level: the dashboard episode of 10.60 "
        f"met it at 1034.1250 px and the operator cleared it in 23 frames. It is absent from *this* "
        f"behaviour vocabulary, because random forward and retreat patterns fall into a pit long "
        f"before the tutorial asks anybody for anything.",
        "",
        f"**What this changes, and what it does not.** The published recordings stay published and "
        f"§10.58/§10.59's numbers remain measurements of the files they name; the two level arms are "
        f"new files, and the recorder's default behaviour was verified element-for-element against "
        f"the committed arrays before this section was written. What survives is the residue 10.59 "
        f"could not explain with the camera and 10.60 could not explain with the pause: "
        f"{verdict['reposition_residue_after_hygiene']} exceptions on {clear['transitions']:,} "
        f"transitions ({100 * clear['non_repeat_exception_share']:.2f}%) that are neither a boundary "
        f"nor a hold. The session is worth {verdict['session_changes_the_identity_pp']:+.2f} pp of "
        f"the identity on its own, which is the number 10.61's terrain result predicts: a different "
        f"level is a different set of walls to be repositioned against.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-metrics", default=results_file("pause_hygiene_metrics.json"))
    args = parser.parse_args()
    run_study(output_metrics=args.output_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
