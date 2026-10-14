"""
dashboard_study.py - README section 10.60: the hero dashboard and the two instruments it settled.

The dashboard is the repository's first artifact whose purpose is to be *looked at*, so the section
is written from measurements rather than impressions: which actor flies the episode (probed, not
chosen), what the engine's own pause costs a policy whose action set has no A button, and what the
frame buffer turned out to hold - a core that requests a 32-bit surface and delivers a 16-bit one
garbles every picture the repository has ever exported, which is a fact about older claims, not
about this video.

The section is emitted from `results/dashboard_study_metrics.json`, which gathers the three
artifacts the dashboard pipeline writes, and the evidence figure is cut out of the committed
recording rather than captured separately.

Run:  python -m src.evaluation.dashboard_study
      make dashboard-study
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np

from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_DASHBOARD,
    REPO_ROOT,
    RESULTS_DIR,
    figure_file,
    results_file,
)
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

RUN_ARTIFACT = results_file("dashboard_run_metrics.json")
SELECTION_ARTIFACT = results_file("dashboard_controller_selection.json")
VIDEO_ARTIFACT = results_file("dashboard_video_metrics.json")
EVIDENCE_FIGURE = figure_file("dashboard_message_box.png")
#: nearest multiple of two that keeps the 256x224 source's pixel grid intact when enlarging
EVIDENCE_SCALE = 2


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _md(path: str) -> str:
    """Forward slashes, because the string is going into a markdown link or a `code` span."""
    return path.replace("\\", "/")


def _rel(path: str) -> str:
    """Repo-relative and forward-slashed, whether the artifact stored it absolute or not."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / path).resolve()
    return _md(str(resolved.relative_to(REPO_ROOT)))


def _read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)
    return payload


def load_inputs() -> Dict[str, Any]:
    """The three artifacts the dashboard pipeline writes, or an empty dict if any is missing."""
    paths = (RUN_ARTIFACT, SELECTION_ARTIFACT, VIDEO_ARTIFACT)
    if not all(os.path.isfile(p) for p in paths):
        return {}
    return {
        "run": _read(RUN_ARTIFACT),
        "selection": _read(SELECTION_ARTIFACT),
        "video": _read(VIDEO_ARTIFACT),
    }


def evidence_frame(run: Dict[str, Any]) -> int:
    """The last held frame before the operator's first A press, where the box is fully printed."""
    pauses: List[Dict[str, Any]] = list(run.get("pauses", []))
    if not pauses:
        raise KeyError("the recording has no pause to cut an evidence frame from")
    first = min(pauses, key=lambda p: int(p["frame"]))
    return int(first["frame"]) - int(first["operator_frames"]) - 1


def extract_evidence_figure(
    dataset_path: str = DATASET_DASHBOARD,
    run: Dict[str, Any] | None = None,
    out_png: str = EVIDENCE_FIGURE,
) -> Dict[str, Any]:
    """Write the message-box frame out of the recording, at an integer scale."""
    if run is None:
        run = _read(RUN_ARTIFACT)
    with np.load(dataset_path, allow_pickle=False) as blob:
        frame = evidence_frame(run)
        rgb = np.asarray(blob["rgb"][frame])
        states = np.asarray(blob["states"], dtype=np.float64)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    big = np.repeat(np.repeat(rgb, EVIDENCE_SCALE, axis=0), EVIDENCE_SCALE, axis=1)
    imageio.imwrite(out_png, big)
    return {
        "figure": _rel(out_png),
        "frame": frame,
        "progress_px": float(states[frame, 0]) - float(states[0, 0]),
        "source_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        "scale": EVIDENCE_SCALE,
        "bytes": os.path.getsize(out_png),
    }


def render_arms_table(selection: Dict[str, Any]) -> List[str]:
    """Every probed arm, including the one that did not make the video."""
    rows = [
        "| arm | published overrides | frames | progress px | most frequent action | reflex frames |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in list(selection["arms"]) + list(selection.get("override_control", [])):
        top = arm["most_frequent_actions"][0] if arm["most_frequent_actions"] else ["-", 0]
        rows.append(
            f"| `{arm['arm']}` | {'yes' if arm['published_overrides'] else 'no'} "
            f"| {arm['frames']} | {arm['progress_px']:.4f} "
            f"| `{top[0]}` x{top[1]} | {arm['frames_where_reflex_overrode']} |"
        )
    return rows


def render_pause_table(run: Dict[str, Any]) -> List[str]:
    rows = [
        "| pause | frame | progress px | held frames | operator A frames | resumed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, event in enumerate(run.get("pauses", []), start=1):
        rows.append(
            f"| {i} | {event['frame']} | {event['progress_px']:.4f} | {event['held_frames']} "
            f"| {event['operator_frames']} | {'yes' if event['resumed'] else 'no'} |"
        )
    return rows


def forced_mode_picture_check(
    frames: int = 150, every: int = 30, max_shift: int = 64
) -> Dict[str, Any]:
    """Re-measure 10.31.1's picture claim now that the decoder is fixed.

    The forced-mode savestate of 10.36.3 has engine physics but is said to draw the menu. That is a
    claim about pixels, and every pixel claim this repository made before 10.60 was decoded as 32-bit,
    so it is measured again rather than trusted: drive the session, compare frame pairs by the best
    horizontal shift, and report whether the picture translates while the state does.
    """
    from src.environment.snes_emulator import SnesLibretroEmulator
    from src.evaluation.pixel_frame_probe import best_shift
    from src.utils.paths import CORE_PATH, ROM_PATH, STATE_YOSHI_ISLAND_1

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    emu.enable_frame_capture(True)
    with open(STATE_YOSHI_ISLAND_1, "rb") as fh:
        emu.start_episode(fh.read())
    shots: List[np.ndarray] = []
    xs: List[float] = []
    last: np.ndarray | None = None
    try:
        for frame in range(frames):
            emu.set_input({"RIGHT": True, "Y": True})
            emu.step_frame()
            if frame % every == 0:
                last = emu.get_frame()
                shots.append(np.asarray(last, dtype=np.float64).mean(axis=2))
                xs.append(emu.get_smw_state()["x"])
    finally:
        emu.close()
    shifts = [best_shift(shots[0], shot, max_shift)[0] for shot in shots[1:]]
    decoded = int(emu.frame_pixel_format)
    requested = int(emu.requested_pixel_format)
    figure = figure_file("dashboard_forced_mode.png")
    os.makedirs(os.path.dirname(figure) or ".", exist_ok=True)
    assert last is not None
    imageio.imwrite(
        figure, np.repeat(np.repeat(last.astype(np.uint8), EVIDENCE_SCALE, 0), EVIDENCE_SCALE, 1)
    )
    return {
        "savestate": os.path.basename(STATE_YOSHI_ISLAND_1),
        "frames_driven": frames,
        "state_dx_px": round(float(xs[-1] - xs[0]), 4),
        "pairs_compared": len(shifts),
        "pairs_that_translated": int(sum(1 for s in shifts if s != 0)),
        "shifts_px": [int(s) for s in shifts],
        "figure": _rel(figure),
        "requested_pixel_format": requested,
        "decoded_as_pixel_format": decoded,
    }


def gather(
    inputs: Dict[str, Any],
    evidence: Dict[str, Any],
    forced_mode: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run, selection, video = inputs["run"], inputs["selection"], inputs["video"]
    return {
        "study": (
            "The README hero dashboard: the arm probe that chose which actor flies it, the episode "
            "the recording holds, the engine's message box as the cause of 10.59's frozen player "
            "record, and the pixel format the frame buffer actually carried."
        ),
        "protocol": {
            "recorder": "scripts/record_dashboard_run.py",
            "renderer": "src/evaluation/dashboard_video.py",
            "savestate": run["protocol"]["savestate"],
            "controller": run["protocol"]["controller"],
            "checkpoint": run["protocol"]["checkpoint"],
            "value_critic": run["protocol"]["value_critic"],
            "operator_rule": run["protocol"]["operator_rule"],
            "operator_delay_frames": run["protocol"]["operator_delay_frames"],
            "pause_trigger_frames": run["protocol"]["pause_trigger_frames"],
            "frame_capture": run["protocol"]["frame_capture"],
            "probe_frames": selection["protocol"]["probe_frames"],
            "learned_arm_rule": selection["protocol"]["learned_arm_rule"],
            "featured_rule": selection["protocol"]["featured_rule"],
        },
        "arms": selection["arms"],
        "override_control": selection.get("override_control", []),
        "learned_arm_selected": selection.get("learned_arm_selected"),
        "featured": selection.get("featured"),
        "episode": run["run"],
        "pauses": run.get("pauses", []),
        "video": {**video["video"], "mp4": _rel(video["video"]["mp4"])},
        "gif": {**video["gif"], "gif": _rel(video["gif"]["gif"])},
        "caption": video["caption"],
        "evidence_figure": evidence,
        "forced_mode_check": forced_mode,
        "dataset": run["dataset"],
        "levels": _levels(),
    }


def _levels() -> Dict[str, Any] | None:
    """How many stages the map turns out to give, and whether any has been finished.

    The hero is one recording of one stage, and a reader who likes it will ask what else there is.
    The answer belongs next to the picture rather than forty sections away, and it belongs only once
    it has been measured - so this reads the study's artifact when there is one and says nothing when
    there is not. The clearance count comes from the campaign that settled what a clear is, not from
    the run that mistook a game over for one.
    """
    path = results_file("level_dashboards_cleared.json")
    if not os.path.isfile(path):
        path = results_file("level_dashboards.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    stages = payload.get("stages", [])
    clearance_path = results_file("world_map_clearance.json")
    cleared = None
    if os.path.isfile(clearance_path):
        with open(clearance_path, encoding="utf-8") as handle:
            cleared = int(json.load(handle).get("cleared_rounds", 0))
    return {
        "stages": len(stages),
        "translevels": sorted({str(s["identity"]["translevel"]) for s in stages}),
        "cleared_rounds": cleared,
        "artifact": os.path.basename(path),
    }


def render_caption(payload: Dict[str, Any]) -> List[str]:
    """The README's hero block: the GIF, and the sentence of numbers a gate can check."""
    ep = payload["episode"]
    gif = _rel(payload["gif"]["gif"])
    mp4 = _rel(payload["video"]["mp4"])
    modes = ", ".join(f"`0x{m:02X}`" for m in ep["mode_bytes_seen"])
    levels = payload.get("levels")
    elsewhere = ""
    if levels and levels["stages"] > 1:
        others = ", ".join(f"`{t}`" for t in levels["translevels"])
        cleared = levels.get("cleared_rounds")
        finish = (
            f", {cleared} of which the clearance controller has now finished on the console's own "
            f"terms"
            if cleared
            else ", none of which anything in this repository has finished"
        )
        elsewhere = (
            f" It is one of {levels['stages']} stages - {others}{finish} - that the world map "
            f"actually hands over, and 10.66 shows the rest and says what a clear takes."
        )
    return [
        "## What the Actor Sees, and What It Believed",
        "",
        "![One clearance-draw episode on the live console: the emulator's own RGB picture filling "
        "the left, and on the right the tile window the level-conditioned models read, what the "
        "actor thinks as eight named actions, the buttons the console actually received, the "
        "terminal-value critic's prediction, and along the bottom the twelve numbers the actor "
        "sees and the stage progress bar](" + gif + ")",
        "",
        f"*One episode of the `{payload['protocol']['controller']}` controller, recorded frame by frame from "
        f"the verified level entry of 10.36.3: {ep['frames']} frames, {ep['progress_px']:.4f} px of "
        f"forward progress, ending as `{ep['ended']}`. The engine mode byte reads {modes} on the "
        f"frames the player was live; the mean top button probability is {ep['mean_max_button_probability']:.4f}; "
        f"the section-4.1 identity holds exactly on {_pct(ep['residue_exact_share'])} of the live "
        f"frames and the exceptions are the {len(payload['pauses'])} intervals where the engine holds the player "
        f"still, which 10.60 "
        f"identifies as the level's own message box. Full quality: "
        f"[`{mp4}`]({mp4}) ({payload['video']['bytes'] / 1e6:.2f} MB, "
        f"{payload['video']['duration_s']:.2f} s) - a download, not a player: GitHub serves a "
        f"committed `.mp4` as a file, which is why the inline copy above is a GIF. 10.60 says "
        f"which controller flies this, and why.{elsewhere}*",
        "",
    ]


def render_section(payload: Dict[str, Any]) -> List[str]:
    r"""The whole of README section 10.60, generated from the artifact."""
    ep = payload["episode"]
    proto = payload["protocol"]
    capture = proto["frame_capture"]
    arms = {
        a["arm"] + ("_no_overrides" if not a["published_overrides"] else ""): a
        for a in list(payload["arms"]) + list(payload["override_control"])
    }
    ppo = arms["ppo"]
    dag = arms["dagger"]
    raw = arms.get("dagger_no_overrides")
    evidence = payload["evidence_figure"]
    forced = payload["forced_mode_check"]
    gif_mb = payload["gif"]["bytes"] / 1e6
    mp4_mb = payload["video"]["bytes"] / 1e6
    void_share = payload["caption"]["tile_void_cell_share"]

    lines: List[str] = [
        "### 10.60 The Dashboard at the Top of This File, and the Two Instruments It Needed",
        "",
        f"The repository had telemetry plots and a schematic GIF, and neither could be looked at "
        f"while the console ran. The hero above is the fix: one episode, the emulator's own RGB "
        f"picture filling the left half and, down the right, the tile window the level-conditioned "
        f"models read, what the actor thinks as eight named actions, the buttons the console actually "
        f"received drawn on a controller, and the critic's $V(s)$; along the bottom, the twelve "
        f"numbers the actor really is given and the stage progress bar. "
        f"`scripts/record_dashboard_run.py` records it, `src/evaluation/dashboard_video.py` renders it "
        f"({mp4_mb:.2f} MB MP4, {gif_mb:.2f} MB GIF), and every number in this section is read out of "
        f"the artifacts those two write.",
        "",
        "**Two things the first version of that picture got wrong.** The tile window was drawn with "
        "`extent=(0, 1, 1, 0)`, which puts the array's first row - the tiles *above* Mario - at the "
        "bottom of the panel, so the committed video showed sky under his feet and ground above his "
        "head. And the class patch cannot distinguish air from a cell the buffer does not have: "
        "`get_local_tilemap_patch` stores both as 0, because that is what the level-conditioned "
        "models are given. The panel now asks `wram.tilemap_addr` for the address of every cell and "
        f"draws the ones that have none in a fifth colour - {_pct(void_share)} of the cells it paints "
        "for this episode, all of them below the ground, because Mario runs along the floor of a "
        "27-row subscreen. Both are the same defect as everywhere else in this file: a picture is a "
        "claim, and a claim drawn from an array has to say which cells the array actually has.",
        "",
        "**Which controller flies it.** Two learned checkpoints claim to be the repository's "
        "controller, so both were probed on the verified level-entry savestate for the same "
        f"{proto['probe_frames']}-frame horizon, and the rule was written down before the run: "
        "the arm that reaches the largest forward progress is the better of the two. That "
        "comparison is reported below either way - but the episode the README opens with is a "
        "different question, and 10.68 changed its answer: the hero shows a level being "
        "completed, so it is flown by the one controller this repository has that finishes the "
        "stage - the 10.66 seed-7 clearance draw, run until the engine itself ends it and kept "
        "recording through the exit sequence.",
        "",
        *render_arms_table(
            {"arms": payload["arms"], "override_control": payload["override_control"]}
        ),
        "",
        f"The PPO primitive actor answers `RUN_LEFT` on all {ppo['frames']} frames and ends "
        f"{abs(ppo['progress_px']):.4f} px behind where it started, at a mean top-1 probability of "
        f"{ppo['mean_top_probability']:.4f} - a distribution that has not decided anything. The "
        f"distilled DAgger actor traverses {dag['progress_px']:.4f} px of the level with a mean top-1 "
        f"button probability of {dag['mean_top_probability']:.4f}, so it stays the better of the "
        f"two learned arms - and neither is the featured episode, because neither finishes the "
        f"stage within any horizon this repository has flown it. The third row is the same actor "
        f"with the published reflex vaults and jump pulse removed: "
        + (
            f"it reaches {raw['progress_px']:.4f} px, "
            f"{raw['progress_px'] - dag['progress_px']:+.4f} px further than the arm that keeps them, "
            "which is the cost of the override layer and the reason the dashboard shows the buttons "
            "the console actually received rather than the ones the actor proposed."
            if raw is not None
            else "no control arm was recorded."
        ),
        "",
        f"**The episode the video shows.** {ep['frames']} frames, "
        f"{ep['progress_px']:.4f} px of forward progress, ending as `{ep['ended']}`, with the mode "
        f"byte reading {', '.join(f'0x{m:02X}' for m in ep['mode_bytes_seen'])} - `0x14` for every "
        f"frame the player was live, then the exit sequence's own march once the engine delivered "
        f"its verdict, which the outro keeps recording. The "
        f"camera travels {ep['camera_travel_px']:.0f} px, Mario's screen position stays inside "
        f"[{ep['screen_x_range_px'][0]:.2f}, {ep['screen_x_range_px'][1]:.2f}] px of the 256 px "
        f"window, and the critic's prediction moves from {ep['value_start_px']:.2f} px to "
        f"{ep['value_end_px']:.2f} px of discounted future progress over the episode.",
        "",
        "**The pause, at last identified.** 10.59 found whole player records repeating on a console "
        "whose WRAM was still changing and called it a paused simulation without saying what paused "
        "it. With a picture attached to the recording, it says so: the engine opened its own message "
        "box, which reads *To do a spin jump, press the A Button*, and the box waits for A.",
        "",
        *render_pause_table({"pauses": payload["pauses"]}),
        "",
        "A is not one of the six channels in `ACTION_BUTTONS`, so no policy in this repository can "
        "dismiss the box it just walked into - the action space has no answer to the level's "
        "instruction. The recorder therefore lets an operator press it"
        + (
            f", holding the frozen player for {proto['operator_delay_frames']} extra frames first so "
            f"the text finishes printing, and"
            if int(proto["operator_delay_frames"]) > 0
            else " - immediately, with no extra hold, because the featured episode is the 10.66 "
            "clearance draw and its own hygiene layer waits only for the frozen record - and"
        )
        + f" marks every one of those {ep['frames_under_operator_A']} frames in the recording and on "
        f"the dashboard. The frame the video holds on is cut out of the committed recording as "
        f"`{_md(evidence['figure'])}` (frame {evidence['frame']}, "
        f"{evidence['progress_px']:.4f} px into the level). The episode itself ends the way the "
        f"featured controller's episodes now end: the mode byte leaves `0x14` for the engine's own "
        f"exit sequence, and the outro keeps recording through it - the mode march and the map "
        f"settling - so the video shows the level being completed rather than freezing on the "
        f"flipped byte. The residue is no longer drawn as a panel; "
        f"it is in the recording, and it satisfies the per-row identity exactly - "
        f"`residue = 16 (x_(i+1) - x_i) - vx_i` - on every frame the player was live. The frames "
        f"where the reading is not zero are the frames the engine itself did not move the position "
        f"at the velocity's rate: {_pct(ep['residue_exact_share'])} of the live frames move exactly "
        f"as §4.1 says, and the exceptions are the holds and the goal tape's half-rate walk - which "
        f"read exactly minus the velocity, because the position stood still while the speed byte "
        f"did not - plus, at the clearance draw's run speed, the whole-pixel re-anchoring 10.59 "
        f"found at the dagger's walk speed and called a reposition: at run speed the pixel byte "
        f"lags the subpixel accumulator, and the reading comes back a whole pixel off until the "
        f"carry lands. 10.58 called the residue a clamp with a memory; here it is a picture that "
        f"does not change.",
        "",
        f"![The engine's message box, held by the player record]({_rel(EVIDENCE_FIGURE)})",
        "",
        "**The frame buffer was not what the core asked for.** Building the video required reading "
        "the pictures, and the first exported frame was a half-width magenta smear. The cause is in "
        "the capture path rather than the level: the core answers "
        "`RETRO_ENVIRONMENT_SET_PIXEL_FORMAT` with "
        f"{capture['requested_pixel_format']} (RGB888, four bytes per pixel) and then hands back "
        f"{capture['decoded_as_pixel_format']} (RGB565, two) - pitch 2048 for a 256-px row, the "
        "second half of every row zero, and the lane a 0x00RRGGBB word keeps at zero carrying "
        "picture data. `refine_pixel_format` now checks the request against those two facts a "
        "genuine buffer cannot violate, and `tests/test_pixel_perception.py` holds both the positive "
        "control (a 16-bit row under a 32-bit request is demoted) and the negative one (a real "
        "RGB888 row is not).",
        "",
        "**What this changes, and what it does not.** It does not move any telemetry result: every "
        "state, residue, camera and mode number in this repository comes from WRAM, and WRAM was "
        "never decoded through the pixel path. What it does touch is any claim this file has made "
        "about what a frame *shows*. The two such claims are the pixel probe of 10.31 and the "
        "picture check of 10.36.3, both run on frames decoded as 32-bit, so both were measured again "
        "with the fixed decoder on the same forced-mode savestate they were made on: driven for "
        f"{forced['frames_driven']} frames the state moves {forced['state_dx_px']:.4f} px while "
        f"{forced['pairs_that_translated']} of {forced['pairs_compared']} frame pairs translate at "
        "all, and the picture is the file-select menu - "
        f"![The forced-mode session draws the menu while the physics runs]({_md(forced['figure'])}). "
        "Both sections' readings were therefore right and for the right reason, and the shift "
        "measurement survives the old decode anyway: the same squeeze applied to every frame cannot "
        "invent a horizontal translation. What the fix does change is this repository's ability to "
        "look at anything it records, which is what the dashboard above is for.",
        "",
    ]
    return lines


def run_study(
    output_metrics: str = results_file("dashboard_study_metrics.json"),
    dataset_path: str = DATASET_DASHBOARD,
    write_figure: bool = True,
    verify_forced_mode: bool = True,
) -> Dict[str, Any]:
    """Gather the three artifacts, cut the evidence figure, and write the study artifact."""
    inputs = load_inputs()
    if not inputs:
        raise FileNotFoundError(
            "the dashboard study needs the recording, the arm probe and the render artifacts: "
            "run `make record-dashboard` and `make dashboard-video` first"
        )
    run = inputs["run"]
    forced = forced_mode_picture_check() if verify_forced_mode else None
    evidence = (
        extract_evidence_figure(dataset_path, run)
        if write_figure
        else {"figure": _rel(EVIDENCE_FIGURE)}
    )
    payload = gather(inputs, evidence, forced)
    os.makedirs(os.path.dirname(output_metrics) or RESULTS_DIR, exist_ok=True)
    write_metrics(output_metrics, payload, command="make dashboard-study")
    logger.info("Artifact written to: %s", output_metrics)
    payload["_artifact"] = output_metrics
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-metrics", default=results_file("dashboard_study_metrics.json"))
    parser.add_argument("--dataset", default=DATASET_DASHBOARD)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--no-forced-mode-check",
        action="store_true",
        help="skip the emulator run that re-measures the forced-mode picture",
    )
    args = parser.parse_args()
    run_study(
        output_metrics=args.output_metrics,
        dataset_path=args.dataset,
        write_figure=not args.no_figure,
        verify_forced_mode=not args.no_forced_mode_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
