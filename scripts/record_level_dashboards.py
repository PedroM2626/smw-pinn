"""
record_level_dashboards.py - one dashboard episode per stage the map actually gives.

10.60 built the hero on a single stage, and 10.66 then surveyed the map: seven cursor positions are
reachable from a fresh file, and the engine hands over a playable session at three distinct level
numbers. This records the same channels 10.60 records - the actor's probabilities, the critic's V(s),
the buttons the console received, the tile window, an RGB frame - at each of those stages, and
measures two more numbers per stage that the dashboard's bottom line needs:

* the *walk-right frontier*: how far a held RIGHT+Y run gets before the episode ends. It is not the
  level's length; it is the furthest this repository has any evidence that the terrain goes, and it
  gives the picture a denominator that was measured rather than assumed.
* what the clearance controller does there: N attempts, the furthest one, and the mode byte the
  engine ended on. 10.66 showed a return to the map at the same node is a game over, not a clear, so
  nothing here claims a clear until the cursor moves.

The selected episode is re-run with the frames kept, and the re-run's progress is compared with the
attempt it replaces: the actor and the console are both deterministic, so a difference would mean one
of them is not, and that is worth a line in the artifact either way.

Run:  python -m scripts.record_level_dashboards
      python -m scripts.record_level_dashboards --mpc-attempts 3 --frames 1268
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.record_dashboard_run import record  # noqa: E402
from scripts.unlock_world_map import build_mpc  # noqa: E402
from src.environment import wram  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import CORE_PATH, ROM_PATH, dataset_path, state_path  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

logger = get_logger(__name__)

#: a rightward walk is over once the player has not advanced for this many frames
STALL_FRAMES = 90
FRONTIER_BUDGET = 3000
#: the published clearance benchmark's own goal distance, used only as a labelled reference
GOAL_PROXY_PX = 1900.0


def walk_right(emu: SnesLibretroEmulator, state_bytes: bytes, budget: int) -> Dict[str, Any]:
    """Hold RIGHT+Y from the stage start and report where the episode ends.

    The B pulse every 24 frames is the same edge-trigger rule the recorders use, because a held B
    never re-triggers a jump; the walk is meant to be as far as a rightward run can go, not as short
    as a stuck button.
    """
    from src.environment.episode_hygiene import PlayerHold

    emu.load_state(state_bytes)
    hold = PlayerHold(limit=240)
    start_x: Optional[float] = None
    best = 0.0
    stalled = 0
    frame = 0
    for frame in range(budget):
        state = emu.get_smw_state()
        if start_x is None:
            start_x = float(state["x"])
        vector = np.asarray(
            [
                state[k]
                for k in ("x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right")
            ],
            dtype=np.float64,
        )
        hold.observe(vector)
        progress = float(state["x"]) - start_x
        stalled = stalled + 1 if progress <= best + 1e-9 else 0
        best = max(best, progress)
        if stalled >= STALL_FRAMES or hold.exhausted() or float(state["y"]) > wram.DEATH_Y:
            break
        buttons = {"RIGHT": True, "Y": True, "B": frame % 24 == 0}
        emu.set_input(hold.buttons(buttons))
        emu.step_frame()
    final = emu.get_smw_state()
    return {
        "frames": frame + 1,
        "frontier_px": float(best),
        "ended": (
            "death"
            if float(final["y"]) > wram.DEATH_Y
            else "hold"
            if hold.exhausted()
            else "stall"
            if stalled >= STALL_FRAMES
            else "budget"
        ),
        "final_mode": f"0x{emu.get_game_mode():02X}",
        "operator_A_frames": int(hold.operator_frames),
    }


def mpc_attempts(
    emu: SnesLibretroEmulator,
    controller: Any,
    state_bytes: bytes,
    attempts: int,
    max_frames: int,
) -> Dict[str, Any]:
    """Fly the clearance controller N times from the same state and keep what each draw reached."""
    from scripts.unlock_world_map import fly

    results: List[Dict[str, Any]] = []
    for attempt in range(attempts):
        emu.load_state(state_bytes)
        report = fly(emu, controller, max_frames)
        report["attempt"] = attempt
        results.append(report)
        logger.info(f"  mpc attempt {attempt}: {report['max_progress_px']:.1f} px")
    best = max(results, key=lambda r: r["max_progress_px"]) if results else None
    return {
        "attempts": results,
        "best_progress_px": float(best["max_progress_px"]) if best else None,
        "end_modes": sorted({str(r["final_mode"]) for r in results}),
    }


def stage_identity(emu: SnesLibretroEmulator, state_bytes: bytes) -> Dict[str, Any]:
    emu.load_state(state_bytes)
    emu.step_frame()
    return {
        "mode_at_start": f"0x{emu.get_game_mode():02X}",
        "translevel": f"0x{emu.read_wram_u8(wram.ADDR_TRANSLEVEL):02X}",
        "timer_hundreds": int(emu.read_wram_u8(wram.ADDR_TIMER_HUNDREDS)),
        "x_at_start": float(emu.get_smw_state()["x"]),
        "y_at_start": float(emu.get_smw_state()["y"]),
    }


def candidates(survey: str) -> List[Dict[str, Any]]:
    """Every distinct level number the map survey captured a state for.

    Two cursor positions can hand over the same stage - the survey found Yoshi's Island 1 at both
    (152,136) and (141,135) - and a second recording of the same bytes would be a duplicate episode
    presented as a second stage. The first node reached for each translevel is the one recorded.
    """
    with open(survey, encoding="utf-8") as handle:
        payload = json.load(handle)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for node in payload["nodes"]:
        capture = node.get("capture") or {}
        if not capture.get("saved"):
            continue
        level = str(node.get("translevel"))
        if level in seen:
            continue
        seen.add(level)
        out.append(
            {
                "state": capture["saved"],
                "node": node["node"],
                "survey_translevel": level,
                "survey_right_response_px": node.get("right_response_px"),
                "capture": capture,
            }
        )
    if not seen:
        out.append({"state": "smw_yoshi_island_1_level.state", "node": None, "capture": {}})
    return out


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Record a dashboard episode per stage.")
    parser.add_argument("--survey", default="results/world_map_survey.json")
    parser.add_argument("--frames", type=int, default=1268)
    parser.add_argument("--mpc-attempts", type=int, default=3)
    parser.add_argument("--mpc-frames", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/level_dashboards.json")
    args = parser.parse_args()

    controller = build_mpc(torch.device(args.device))
    stages: List[Dict[str, Any]] = []
    for entry in candidates(args.survey):
        name = os.path.basename(entry["state"])
        tag = name[: -len(".state")]
        path = state_path(entry["state"])
        with open(path, "rb") as handle:
            state_bytes = handle.read()
        record_out = dataset_path(f"smw_dashboard_run_{tag}.npz")
        logger.info(f"stage {entry['state']}")

        emu = SnesLibretroEmulator(CORE_PATH)
        emu.load_rom(ROM_PATH)
        identity = stage_identity(emu, state_bytes)
        frontier = walk_right(emu, state_bytes, FRONTIER_BUDGET)
        emu.close()
        logger.info(f"  frontier {frontier['frontier_px']:.1f} px ({frontier['ended']})")

        emu = SnesLibretroEmulator(CORE_PATH)
        emu.load_rom(ROM_PATH)
        mpc = mpc_attempts(emu, controller, state_bytes, args.mpc_attempts, args.mpc_frames)
        emu.close()

        cheap = record(
            "dagger",
            path,
            args.frames,
            args.device,
            256,
            224,
            keep_frames=False,
            output_dir=os.path.join("results", "level_dashboards", tag),
        )
        kept = record(
            "dagger",
            path,
            args.frames,
            args.device,
            256,
            224,
            keep_frames=True,
            dataset_path=record_out,
            savestate_route=(
                "entered through the world map; the survey's own capture step saved the state "
                "180 frames after the fade in, before the handoff probes"
            ),
            output_dir=os.path.join("results", "level_dashboards", tag),
        )
        delta = abs(float(kept["run"]["progress_px"]) - float(cheap["run"]["progress_px"]))
        with np.load(record_out) as blob:
            recorded = blob["states"]
        # The recorder's own `progress_px` is the position reached by the last step it executed; the
        # dashboard draws the states it stored, and those two differ by exactly one frame whenever
        # the episode ends on the frame budget rather than on a hold. The picture and the table must
        # quote the same number, so the table quotes the picture's.
        episode_progress = float(recorded[-1, 0] - recorded[0, 0])
        frontier_px = float(frontier["frontier_px"])
        stages.append(
            {
                "savestate": entry["state"],
                "map_node": entry["node"],
                "identity": identity,
                "survey": {
                    "right_response_px": entry["capture"].get("right_response_px"),
                    "handoff_gate_of_10_36_3": entry["capture"].get("handoff_gate_of_10_36_3"),
                },
                "walk_right_frontier": frontier,
                "mpc": mpc,
                "episode": {
                    "dataset": os.path.relpath(record_out, os.getcwd()).replace("\\", "/"),
                    "frames": kept["run"]["frames"],
                    "progress_px": episode_progress,
                    "progress_after_last_step_px": float(kept["run"]["progress_px"]),
                    "ended": kept["run"]["ended"],
                    "share_of_frontier": episode_progress / frontier_px
                    if frontier_px > 0
                    else None,
                    "repeats_the_measured_attempt_px": round(delta, 6),
                },
                "goal_proxy_px": GOAL_PROXY_PX,
            }
        )
        logger.info(f"  episode {episode_progress:.1f} px of {frontier_px:.1f} px frontier")

    payload = {
        "study": (
            "One dashboard episode per stage the world map hands over, with the distance a rightward "
            "walk reaches in that stage and what the clearance controller does there."
        ),
        "protocol": {
            "recorder": "scripts/record_dashboard_run.py, through scripts/record_level_dashboards.py",
            "controller": "dagger",
            "frames_per_episode": args.frames,
            "frontier_rule": f"RIGHT+Y with the B pulse, stopped at {STALL_FRAMES} frames of no "
            "advance, a spent hold, death, or the frame budget",
            "mpc_attempts": args.mpc_attempts,
            "mpc_frames": args.mpc_frames,
            "goal_proxy_px": GOAL_PROXY_PX,
            "clear_definition": "none claimed: 10.66 showed the engine returns to the map at the same "
            "node when the run ends, which is a game over",
            "device": args.device,
        },
        "stages": stages,
        "stages_recorded": len(stages),
    }
    write_metrics(args.output, payload, command="python -m scripts.record_level_dashboards")
    return payload


if __name__ == "__main__":
    main()
