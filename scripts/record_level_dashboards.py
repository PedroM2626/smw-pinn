"""
record_level_dashboards.py - one dashboard episode per stage, flown by the agent that can finish it.

10.60 built the hero on a single stage, and 10.66 then surveyed the map: three distinct level
numbers, plus the blank session behind the Yoshi's House node. 10.68 re-reads the tile map the
way the engine reads it (16-bit Map16 ids, behaviours measured off the console), and this
recorder now gives each stage the agent that can actually finish it:

* Yoshi's Island 1 (0x2A) - the published clearance controller, seed 7, the draw that cleared it
  in 10.66, flown until the engine itself ends the stage: the mode byte leaving 0x14, which is
  the clear discriminator of 10.66 rather than a distance proxy;
* the Yoshi's House session (0x28) - the 10.68 rule-based runner walking back out through the
  door it came in by, the only way a session with no goal tape ends;
* the 0x27 and 0x29 sessions - the distilled actor for a fixed horizon, because 10.68 measured
  those sessions to be unplayable: their floors stop holding (a measured decor row over air) and
  their entrances capture the player in an animation that never completes, so no agent finishes
  them and the episode documents what the engine does instead.

Every channel 10.60 records is recorded here too - and the two channels 10.68 added: the full
16-bit tile ids of the 33x33 window (the panel classifies them through the measured behaviour
table) and every live sprite slot with its id, so the panel draws each one in the class the
engine was measured to give it.

Run:  python -m scripts.record_level_dashboards
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
from scripts.survey_world_map import terrain_width  # noqa: E402
from scripts.unlock_world_map import DEATH_MODE  # noqa: E402
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

#: which agent flies each stage, and why: the one stage that can be finished is finished by the
#: controller that finished it in 10.66; every other session is documented by the distilled actor
#: for a fixed horizon, because 10.68 measured them unplayable - the house session's own exit is
#: a warp this core never completes, and the 0x27/0x29 floors stop holding
EPISODES = {
    "0x2A": {
        "kind": "mpc",
        "seed": 7,
        "frames": 9000,
        "goal_px": None,
        "operator_delay": 0,
        "warmup_frames": 0,
        "render_stride": 8,
    },
    "0x28": {"kind": "dagger", "seed": None, "frames": 1268, "goal_px": GOAL_PROXY_PX},
    "0x27": {"kind": "dagger", "seed": None, "frames": 1268, "goal_px": GOAL_PROXY_PX},
    "0x29": {"kind": "dagger", "seed": None, "frames": 1268, "goal_px": GOAL_PROXY_PX},
}


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


def candidates(survey: str, progression: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every distinct level number the map has handed over - surveyed, or unlocked by a clear.

    Two cursor positions can hand over the same stage - the survey found Yoshi's Island 1 at both
    (152,136) and (141,135) - and a second recording of the same bytes would be a duplicate episode
    presented as a second stage, so the list is keyed by the translevel byte and the first state seen
    wins. The clearance campaign's round states are included because that is how a stage that only
    exists after another stage was finished gets here.
    """
    with open(survey, encoding="utf-8") as handle:
        payload = json.load(handle)
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def add(state: str, node: Any, translevel: str, capture: Dict[str, Any]) -> None:
        if translevel in seen:
            return
        seen.add(translevel)
        out.append({"state": state, "node": node, "translevel": translevel, "capture": capture})

    for node in payload["nodes"]:
        capture = node.get("capture") or {}
        if not capture.get("saved"):
            continue
        add(
            capture["saved"],
            node["node"],
            str(node.get("translevel")),
            {**capture, "source": "the map survey of 10.66"},
        )
    if progression and os.path.isfile(progression):
        with open(progression, encoding="utf-8") as handle:
            campaign = json.load(handle)
        for round_record in campaign.get("rounds", []):
            state = round_record.get("savestate")
            if not state or not round_record.get("entered"):
                continue
            add(
                state,
                round_record.get("node"),
                str(round_record.get("translevel")),
                {
                    "saved": state,
                    "source": "the clearance campaign",
                    "cleared_by_the_seeded_run": bool(round_record.get("cleared")),
                    "terrain_px_from_campaign": (round_record.get("terrain") or {}).get(
                        "terrain_px"
                    ),
                },
            )
    if not seen:
        add(
            "smw_yoshi_island_1_level.state",
            None,
            "0x2A",
            {"saved": True, "source": "the level entry of 10.36.3"},
        )
    return out


def clear_evidence(episode: Dict[str, Any]) -> Dict[str, Any]:
    """The 10.66 discriminator, read off the recorded mode channel of the episode.

    A stage is cleared when the engine left interactive mode for something that is not the death
    animation: 0x14 -> 0x0B is a death, 0x14 -> anything else is the stage's own exit sequence.
    """
    modes = episode.get("mode_bytes_seen") or []
    exits = [m for m in modes if m != wram.GAME_MODE_INTERACTIVE]
    if not exits:
        reason = (
            "the player fell past the death line before the mode byte moved"
            if str(episode.get("ended")) == "death"
            else "the engine never left 0x14"
        )
        return {"cleared": False, "exit_mode": None, "reason": reason}
    exit_mode = int(exits[0])
    return {
        "cleared": exit_mode != DEATH_MODE,
        "exit_mode": f"0x{exit_mode:02X}",
        "reason": (
            "the engine ended the stage without the death mode"
            if exit_mode != DEATH_MODE
            else "the engine entered the death mode"
        ),
    }


def clearance_record(
    translevel: str, progression: Dict[str, Any], episode: Dict[str, Any]
) -> Dict[str, Any]:
    """The stage's clearance numbers, from wherever they were actually measured.

    The episode of 0x2A *is* the seed-7 clearance draw re-flown with its frames kept, so its own
    progress is the best attempt. The campaign of 10.66 published two flights at 0x27. The other
    two sessions have never had a clearance flight that could finish, and saying so is the number.
    """
    if translevel == "0x2A" and bool((episode.get("clear_evidence") or {}).get("cleared")):
        return {
            "clears": 1,
            "tries": 1,
            "best_progress_px": float(episode["progress_px"]),
            "source": "the recorded episode itself: the seed-7 clearance draw of 10.66, re-flown with its frames kept",
        }
    for round_record in progression.get("rounds", []):
        if str(round_record.get("translevel")) == translevel and round_record.get("attempts"):
            best = max(float(a["max_progress_px"]) for a in round_record["attempts"])
            return {
                "clears": sum(1 for a in round_record["attempts"] if a.get("cleared")),
                "tries": len(round_record["attempts"]),
                "best_progress_px": best,
                "source": "the published clearance campaign of 10.66",
            }
    return {
        "clears": 0,
        "tries": 0,
        "best_progress_px": None,
        "source": "no clearance flight has ever finished a run in this session; 10.68 measured it unplayable",
    }


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Record a dashboard episode per stage.")
    parser.add_argument("--survey", default="results/world_map_survey.json")
    parser.add_argument(
        "--progression",
        default="results/world_map_clearance.json",
        help="the clearance campaign's artifact, whose round states add unlocked stages",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/level_dashboards_cleared.json")
    args = parser.parse_args()

    stages: List[Dict[str, Any]] = []
    for entry in candidates(args.survey, args.progression):
        translevel = entry["translevel"]
        plan = EPISODES.get(
            translevel, {"kind": "dagger", "seed": None, "frames": 1268, "goal_px": GOAL_PROXY_PX}
        )
        name = os.path.basename(entry["state"])
        tag = name[: -len(".state")]
        path = state_path(entry["state"])
        with open(path, "rb") as handle:
            state_bytes = handle.read()
        record_out = dataset_path(f"smw_dashboard_run_{tag}.npz")
        logger.info(f"stage {entry['state']} ({translevel}): {plan['kind']}")
        emu = SnesLibretroEmulator(CORE_PATH)
        emu.load_rom(ROM_PATH)
        identity = stage_identity(emu, state_bytes)
        emu.load_state(state_bytes)
        emu.step_frame()
        terrain = terrain_width(emu)
        frontier = walk_right(emu, state_bytes, FRONTIER_BUDGET)
        emu.close()
        logger.info(
            f"  terrain {terrain['terrain_px']} px, frontier {frontier['frontier_px']:.1f} px ({frontier['ended']})"
        )

        episode = record(
            kind=plan["kind"],
            state_path=path,
            max_frames=plan["frames"],
            device=args.device,
            rgb_width=256,
            rgb_height=224,
            keep_frames=True,
            goal_px=plan["goal_px"],
            seed=plan.get("seed"),
            runner_direction=plan.get("direction", "RIGHT"),
            apply_overrides=plan["kind"] != "mpc",
            operator_delay=int(plan.get("operator_delay", 120)),
            warmup_frames=int(plan.get("warmup_frames", 5)),
            savestate_route=(
                "entered through the world map; the survey's own capture step saved the state "
                "180 frames after the fade in, before the handoff probes"
            ),
            dataset_path=record_out,
            output_dir=os.path.join("results", "level_dashboards", tag),
        )
        evidence = clear_evidence(episode["run"])
        progression = (
            json.load(open(args.progression, encoding="utf-8"))
            if os.path.isfile(args.progression)
            else {}
        )
        mpc = clearance_record(
            translevel,
            progression,
            {"progress_px": episode["run"]["progress_px"], "clear_evidence": evidence},
        )
        logger.info(
            f"  episode {episode['run']['frames']} frames, ended {episode['run']['ended']} -> "
            f"cleared={evidence['cleared']} ({evidence['reason']})"
        )
        stages.append(
            {
                "savestate": entry["state"],
                "map_node": entry["node"],
                "identity": identity,
                "entry_evidence": {
                    "source": entry["capture"].get("source"),
                    "right_response_px": entry["capture"].get("right_response_px"),
                    "handoff_gate_of_10_36_3": entry["capture"].get("handoff_gate_of_10_36_3"),
                    "cleared_by_the_seeded_run": entry["capture"].get("cleared_by_the_seeded_run"),
                },
                "walk_right_frontier": frontier,
                "terrain": terrain,
                "episode": {
                    "dataset": os.path.relpath(record_out, os.getcwd()).replace("\\", "/"),
                    "controller": episode["protocol"]["controller"],
                    "controller_kind": episode["protocol"]["controller_kind"],
                    "planner_seed": episode["protocol"]["planner_seed"],
                    "episode_end_rule": episode["protocol"]["episode_end_rule"],
                    "frames": episode["run"]["frames"],
                    "progress_px": episode["run"]["progress_px"],
                    "ended": episode["run"]["ended"],
                    "mode_bytes_seen": episode["run"]["mode_bytes_seen"],
                    "wall_clock_s": episode["run"]["wall_clock_s"],
                    "frames_under_operator_A": episode["run"]["frames_under_operator_A"],
                },
                "clear_evidence": evidence,
                "mpc": mpc,
                "render_stride": int(plan.get("render_stride", 2)),
                "goal_proxy_px": GOAL_PROXY_PX,
            }
        )

    payload = {
        "study": (
            "One dashboard episode per stage the world map hands over, flown by the agent that "
            "can finish that stage, with the 10.66 mode-byte clear discriminator and the two "
            "perception channels 10.68 added: 16-bit tile ids and every live sprite slot."
        ),
        "protocol": {
            "recorder": "scripts/record_dashboard_run.py, through scripts/record_level_dashboards.py",
            "episode_plans": {k: {kk: vv for kk, vv in v.items()} for k, v in EPISODES.items()},
            "clear_definition": (
                "the engine left interactive mode (0x14) for anything other than the death mode "
                "(0x0B) - the discriminator 10.66 read off the byte and 10.67 re-checked"
            ),
            "unplayable_sessions": (
                "0x27 and 0x29: 10.68 measured their floors to stop holding (a measured decor "
                "row over air) and their entrances to capture the player in an animation that "
                "never completes; no agent finishes them, and the episode documents what the "
                "engine does"
            ),
            "frontier_rule": f"RIGHT+Y with the B pulse, stopped at {STALL_FRAMES} frames of no "
            "advance, a spent hold, death, or the frame budget",
            "goal_proxy_px": GOAL_PROXY_PX,
            "device": args.device,
        },
        "stages": stages,
        "stages_recorded": len(stages),
    }
    write_metrics(args.output, payload, command="python -m scripts.record_level_dashboards")
    return payload


if __name__ == "__main__":
    main()
