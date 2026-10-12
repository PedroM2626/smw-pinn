"""
unlock_world_map.py - clear a stage on the engine's own terms and see what the map opens.

The survey of 10.66 found seven cursor positions reachable from a fresh file and one level the
engine hands over cleanly. That is not a limitation of the harness: Super Mario World opens the rest
of the island only after the previous stage is *cleared*, and clearing means the engine leaves the
level on its own, not that a distance proxy is crossed. The published clearance benchmark of 10.37
stops at `x >= 1900 px`, which is a proxy this repository has never checked against the engine's own
signal - and 10.64 showed the session that proxy was measured in is the title screen's own.

So this flies the MPC controller at the level entered through the world map and records the mode byte
every frame. A clear is the engine walking the level out and putting the cursor back on the map
(0x0E); a death is the mode leaving 0x14 and coming back. Either way the answer is a mode sequence,
not an interpretation, and the map is re-surveyed afterwards so whatever it opened is written down
with the same gate as before.

Run:  python -m scripts.unlock_world_map
      python -m scripts.unlock_world_map --attempts 2 --max-frames 3000
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from scripts.navigate_to_level import (  # noqa: E402
    _plausible,
    walk_to_overworld,
)
from scripts.survey_world_map import (  # noqa: E402
    cursor,
    discover_nodes,
    hold,
    press_a_until_level,
    replay,
)
from src.environment import wram  # noqa: E402
from src.environment.episode_hygiene import PlayerHold  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.evaluation.evaluate_full_level_clearance import (  # noqa: E402
    action_vector_to_dict,
    extract_12d_vector,
)
from src.models.pinn_hard_residual import HardResidualPINNDynamics  # noqa: E402
from src.models.pinn_multi_entity import MultiEntityPINNDynamics  # noqa: E402
from src.planning.mpc_planner import ModelPredictiveController, TrajectoryObjective  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CORE_PATH,
    ROM_PATH,
    checkpoint_file,
    data_dir,
    results_file,
)
from src.utils.provenance import write_metrics  # noqa: E402

logger = get_logger(__name__)

#: frames spent on the clear sequence and the walk back to the map
EXIT_BUDGET = 1800
#: how long the operator keeps trying to dismiss whatever is holding the player
OPERATOR_LIMIT = 240


def build_mpc(device: torch.device) -> ModelPredictiveController:
    """The published clearance controller, built the way 10.37 builds it."""
    base_pinn = HardResidualPINNDynamics(state_dim=8, action_dim=6)
    world_model = MultiEntityPINNDynamics(base_pinn=base_pinn).to(device)
    ckpt = checkpoint_file("pinn_multi_entity_best.pt")
    world_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    world_model.eval()
    objective = TrajectoryObjective(
        weight_progress=3.5,
        weight_velocity=0.6,
        pit_penalty=1200.0,
        death_y=450.0,
        hazard_penalty=850.0,
        leap_bonus=400.0,
    )
    return ModelPredictiveController(
        world_model=world_model,
        device=device,
        horizon=16,
        num_candidates=256,
        cem_iterations=3,
        objective=objective,
    )


def fly(
    emu: SnesLibretroEmulator, controller: ModelPredictiveController, max_frames: int
) -> Dict[str, Any]:
    """Drive one attempt and let the engine decide how it ended.

    The mode byte is sampled every frame and only the transitions are kept, because the question this
    script exists to answer is what the engine does at the end of a stage - a claim that has to be
    read off the byte, not inferred from a distance. The hygiene of 10.63 is applied so a message box
    cannot be mistaken for a clear that never happens.
    """
    start_x = float(emu.get_smw_state()["x"])
    hold_state = PlayerHold(limit=OPERATOR_LIMIT)
    modes: List[Tuple[int, int]] = []
    previous: int | None = None
    progress = 0.0
    seconds = 0.0
    for frame in range(max_frames):
        state = emu.get_smw_state()
        vector = extract_12d_vector(emu.get_smw_extended_state())
        mode = emu.get_game_mode()
        if mode != previous:
            modes.append((frame, int(mode)))
            previous = mode
        progress = max(progress, float(state["x"]) - start_x)
        if mode != wram.GAME_MODE_INTERACTIVE:
            break
        index, _ = controller.plan(vector)
        buttons = action_vector_to_dict(index)
        hold_state.observe(np_vector(state))
        if hold_state.exhausted():
            break
        emu.set_input(hold_state.buttons(buttons))
        t0 = time.perf_counter()
        emu.step_frame()
        seconds += time.perf_counter() - t0
    return {
        "frames": len(modes) and frame + 1,
        "mode_transitions": [{"frame": f, "mode": f"0x{m:02X}"} for f, m in modes],
        "max_progress_px": float(progress),
        "final_mode": f"0x{emu.get_game_mode():02X}",
        "on_map_at_exit": emu.get_game_mode() == wram.GAME_MODE_WORLD_MAP,
        "wall_seconds_per_frame": round(seconds / max(1, frame + 1), 5),
    }


def np_vector(state: Dict[str, float]):
    """The eight-channel record `PlayerHold` watches, in the order the recorders use."""
    import numpy as np

    keys = ("x", "y", "vx", "vy", "c_ground", "c_ceiling", "c_left", "c_right")
    return np.asarray([state[k] for k in keys], dtype=np.float64)


def wait_for_map(emu: SnesLibretroEmulator) -> Dict[str, Any]:
    """Sit through whatever the engine does after the level ends until the cursor is back on the map.

    The mode byte that follows 0x14 is not 0x0E: a cleared stage walks its own exit sequence first,
    and this records that sequence rather than guessing which of its members means a clear.
    """
    modes: List[str] = []
    for frame in range(EXIT_BUDGET):
        mode = f"0x{emu.get_game_mode():02X}"
        if not modes or modes[-1] != mode:
            modes.append(mode)
        if emu.get_game_mode() == wram.GAME_MODE_WORLD_MAP:
            return {"on_map": True, "frames": frame, "modes": modes}
        emu.set_input({})
        emu.step_frame()
    return {"on_map": False, "frames": EXIT_BUDGET, "modes": modes}


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Clear a stage and survey what the map opens.")
    parser.add_argument("--rounds", type=int, default=4, help="levels to try to clear in sequence")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--max-nodes", type=int, default=40)
    parser.add_argument("--output", default=results_file("world_map_progression.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = build_mpc(device)

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    trace: Dict[str, Any] = {"target_level": 1, "node_round": None, "entry_frame": None}
    for _ in range(220):
        emu.step_frame()
    if not walk_to_overworld(emu, trace):
        raise RuntimeError(f"the world map was never reached (mode 0x{emu.get_game_mode():02X})")
    hold(emu, {}, 60)
    map_state = emu.save_state()
    start = cursor(emu)
    before = discover_nodes(emu, map_state, start, args.max_nodes)
    known = {tuple(p) for p in before["positions"]}
    logger.info(f"map before: {len(known)} positions, cursor at {start}")

    rounds: List[Dict[str, Any]] = []
    for round_index in range(args.rounds):
        replay(emu, map_state, [])
        node = cursor(emu)
        frame, _ = press_a_until_level(emu)
        if frame < 0:
            rounds.append({"round": round_index, "node": list(node), "entered": False})
            break
        hold(emu, {}, 180)
        entry_state = emu.get_smw_state()
        translevel = f"0x{emu.read_wram_u8(wram.ADDR_TRANSLEVEL):02X}"
        level_start = emu.save_state()
        tag = translevel.lower().replace("0x", "t")
        name = f"smw_map_round_{round_index}_{node[0]}_{node[1]}_{tag}.state"
        state_path = data_dir() / name
        state_path.write_bytes(level_start)
        record: Dict[str, Any] = {
            "round": round_index,
            "node": list(node),
            "entered": True,
            "entry_frame": frame,
            "translevel": translevel,
            "timer_hundreds": int(emu.read_wram_u8(wram.ADDR_TIMER_HUNDREDS)),
            "x_at_entry": float(entry_state["x"]),
            "y_at_entry": float(entry_state["y"]),
            "plausible_at_entry": bool(_plausible(emu)),
            "savestate": state_path.name,
            "attempts": [],
            "cleared": None,
        }
        for attempt in range(args.attempts):
            emu.load_state(level_start)
            report = fly(emu, controller, args.max_frames)
            report["attempt"] = attempt
            record["attempts"].append(report)
            logger.info(
                f"round {round_index} attempt {attempt}: {report['max_progress_px']:.1f} px, "
                f"final mode {report['final_mode']}, on map at exit={report['on_map_at_exit']}"
            )
            if report["final_mode"] != "0x14":
                waited = wait_for_map(emu)
                report["after_exit"] = waited
                after = list(cursor(emu)) if waited["on_map"] else None
                report["map_cursor"] = after
                if waited["on_map"] and after != list(node):
                    record["cleared"] = {
                        "attempt": attempt,
                        "map_wait_frames": waited["frames"],
                        "exit_modes": waited["modes"],
                        "cursor_after": after,
                        "transitions": report["mode_transitions"],
                    }
                    map_state = emu.save_state()
                    break
                if waited["on_map"]:
                    record["returned_to_map_same_node"] = {
                        "attempt": attempt,
                        "exit_modes": waited["modes"],
                    }
        if record["cleared"] is None:
            rounds.append(record)
            break
        rounds.append(record)

    final = discover_nodes(emu, map_state, cursor(emu), args.max_nodes)
    new_positions = [p for p in final["positions"] if tuple(p) not in known]
    logger.info(
        f"{sum(1 for r in rounds if r.get('cleared'))} rounds cleared; "
        f"the map now reaches {len(final['positions'])} positions, {len(new_positions)} new"
    )
    payload: Dict[str, Any] = {
        "study": (
            "What the engine does when the clearance controller is flown at the level entered through "
            "the world map, and what the map reaches afterwards."
        ),
        "protocol": {
            "controller": "mpc",
            "checkpoint": "pinn_multi_entity_best.pt",
            "rounds": args.rounds,
            "attempts_per_round": args.attempts,
            "max_frames_per_attempt": args.max_frames,
            "exit_budget_frames": EXIT_BUDGET,
            "clear_definition": (
                "the engine reached the world map (0x0E) with the cursor on a node other than the one "
                "entered; reaching 0x0E at the same node is a game over, which is what 0x0B -> 0x0C "
                "-> 0x0D -> 0x0E is"
            ),
            "savestate_dir": "data/raw",
            "device": str(device),
        },
        "map_before": {
            "cursor": list(start),
            "positions": before["positions"],
            "paths": before["paths"],
        },
        "rounds": rounds,
        "cleared_rounds": sum(1 for r in rounds if r.get("cleared")),
        "map_after": {
            "positions": final["positions"],
            "paths": final["paths"],
            "new_positions": new_positions,
        },
    }
    emu.close()
    write_metrics(args.output, payload, command="python -m scripts.unlock_world_map")
    return payload


if __name__ == "__main__":
    main()
