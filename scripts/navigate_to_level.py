"""
navigate_to_level.py
Drives the Super Mario World boot sequence through the emulator, selects file 1,
enters a target level (Yoshi's Island 1 by default) and saves the level-entry savestate.

For `--level 2` the attempt always leaves machine-readable evidence behind in
`results/yi2_capture_attempt.json` (see README section 10.36), whether it succeeds
or is blocked by the post-entry message box.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import wram
from src.environment.snes_emulator import SnesLibretroEmulator
from src.utils.paths import CORE_PATH, ROM_PATH, STATE_YOSHI_ISLAND_1, STATE_YOSHI_ISLAND_2


def _record_attempt(trace: dict, outcome: str) -> None:
    """Persist the capture diagnostics so a blocked claim stays checkable."""
    if trace.get("target_level") != 2:
        return  # the artifact documents the Yoshi's Island 2 frontier only
    from src.utils.paths import results_file
    from src.utils.provenance import write_metrics

    write_metrics(
        results_file("yi2_capture_attempt.json"),
        {"outcome": outcome, **trace},
        command=f"python -m scripts.navigate_to_level --level {trace['target_level']}",
    )


def boot_and_enter_level(
    rom_path: str = ROM_PATH,
    core_path: str = CORE_PATH,
    state_out_path: str = STATE_YOSHI_ISLAND_1,
    level: int = 1,
):
    print(f"Booting emulator with ROM: {rom_path}...")
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    trace: dict = {
        "target_level": level,
        "node_round": None,
        "entry_frame": None,
        "mode_at_entry": None,
        "x_at_entry": None,
        "y_at_entry": None,
        "air_state_at_entry": None,
        "post_message_state": None,
        "stable_frames": 0,
        "movement_dx_px": None,
    }

    # Step 1: advance to the title screen (~200 frames)
    print("Waiting for the Nintendo logo and the title screen...")
    for f in range(220):
        emu.step_frame()

    # Steps 2-3: leave the title screen (0x07) and the file selector (0x08).
    # START now actually reaches the emulator; alternate START/B with polling
    # until both modes are exited (world map / intro reached).
    print("Leaving the title screen / file selector...")
    for _ in range(15):
        mode = emu.get_game_mode()
        if mode not in (wram.GAME_MODE_TITLE, wram.GAME_MODE_FILE_SELECT):
            break
        for _ in range(20):
            emu.set_input({"START": True})
            emu.step_frame()
        mode = emu.get_game_mode()
        if mode not in (wram.GAME_MODE_TITLE, wram.GAME_MODE_FILE_SELECT):
            break
        for _ in range(20):
            emu.set_input({"B": True})
            emu.step_frame()
    for _ in range(60):
        emu.set_input({})
        emu.step_frame()

    # Step 4: on the world map an intro message box may appear
    print("Advancing past the world-map introduction...")
    for _ in range(180):
        emu.set_input({"B": True})  # B closes the text box
        emu.step_frame()

    # Step 5: move to the target level point and enter it (press B or START on the node)
    # level=1: Yoshi's Island 1 (starting node, with a safety LEFT input).
    # level=2: Yoshi's Island 2 (one node to the RIGHT on the map, reachable
    # without clearing level 1). Experimental: verified by mode/coordinates below.
    print("Navigating the world map towards the target level point...")
    if level == 2:
        # On the map, Yoshi's Island 2 sits to the right of Yoshi's House. Since
        # entering with B stops at the FIRST node (the House), travel past it
        # first (~500 frames walking right clear the House node) and only then
        # attempt the B entry.
        print("Passing the Yoshi's House node towards level 2...")
        for _ in range(500):
            emu.set_input({"RIGHT": True})
            emu.step_frame()
        for _ in range(20):
            emu.set_input({})
            emu.step_frame()
        entered = False
        for round_idx in range(20):
            for _ in range(30):
                emu.set_input({"RIGHT": True})
                emu.step_frame()
            for _ in range(10):
                emu.set_input({})
                emu.step_frame()
            for _ in range(10):
                emu.set_input({"B": True})
                emu.step_frame()
            if emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE:
                entered = True
                trace["node_round"] = round_idx
                print(f"Level 2 node reached on round {round_idx}.")
                break
        if not entered:
            print("Warning: level 2 node not confirmed; proceeding to the mode wait.")
    else:
        for _ in range(60):
            emu.set_input({"LEFT": True, "B": True})
            emu.step_frame()

    # Wait for the level transition ($7E:0100 == 0x14 is Level Game Mode)
    print("Waiting for the level-entry transition ($7E:0100 == 0x14)...")
    max_frames = 1500
    level_loaded = False

    for frame_idx in range(max_frames):
        game_mode = emu.get_game_mode()
        # If the level has not started yet, press B to confirm entry
        if game_mode != wram.GAME_MODE_INTERACTIVE:
            emu.set_input({"B": True})
        else:
            emu.set_input({})

        emu.step_frame()

        if game_mode == wram.GAME_MODE_INTERACTIVE:
            # Check that Mario already has valid in-level coordinates
            x = emu.read_wram_u16_le(wram.ADDR_PLAYER_X)
            y = emu.read_wram_u16_le(wram.ADDR_PLAYER_Y)
            if x > 0 and y > 0:
                trace.update(
                    {
                        "entry_frame": frame_idx,
                        "mode_at_entry": hex(game_mode),
                        "x_at_entry": x,
                        "y_at_entry": y,
                        "air_state_at_entry": emu.read_wram_u8(wram.ADDR_AIR_STATE),
                    }
                )
                print(
                    f"Level loaded on frame {frame_idx}! Mode: 0x{game_mode:02X}, Mario X={x}, Y={y}"
                )
                if level == 2:
                    # Yoshi's Island 2: the entry slide opens a message box that
                    # only dismisses with a Y pulse (edge on the $7E:0016 latch;
                    # held B/A/X do not dismiss it). After dismissal there is a
                    # fade (mode leaves 0x14 and returns); trust only once stable.
                    print("Dismissing the level 2 message box with Y pulses...")
                    for _ in range(240):
                        emu.set_input({})
                        emu.step_frame()
                    for _ in range(12):
                        for _ in range(10):
                            emu.set_input({"Y": True})
                            emu.step_frame()
                        for _ in range(40):
                            emu.set_input({})
                            emu.step_frame()
                        if emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE and emu.read_wram_u8(
                            wram.ADDR_AIR_STATE
                        ) in (0, 1, 2):
                            break
                    st = emu.get_smw_state()
                    trace["post_message_state"] = {k: float(v) for k, v in st.items()}
                    print(f"Post-message state: {st}")

                    # Stability: zeroed WRAM during transitions passes naive
                    # checks (air==0 on all-zero memory). Require 60 consecutive
                    # plausible frames before trusting the state.
                    def _plausible() -> bool:
                        s = emu.get_smw_state()
                        return (
                            emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE
                            and 0.0 < s["x"] < 4000.0
                            and 0.0 < s["y"] < 450.0
                            and emu.read_wram_u8(wram.ADDR_AIR_STATE) in (0, 1, 2)
                        )

                    stable = 0
                    for _ in range(1200):
                        emu.set_input({})
                        emu.step_frame()
                        stable = stable + 1 if _plausible() else 0
                        if stable >= 60:
                            break
                    trace["stable_frames"] = stable
                    if stable < 60:
                        _record_attempt(
                            trace,
                            outcome="blocked: WRAM never stabilised after the message box",
                        )
                        raise RuntimeError("Level 2: state never stabilised. Savestate NOT saved.")
                    # Honesty gate: save only if Mario responds to input
                    # and ends in a plausible state.
                    x_before = emu.get_smw_state()["x"]
                    for _ in range(90):
                        emu.set_input({"RIGHT": True, "Y": True})
                        emu.step_frame()
                    st_after = emu.get_smw_state()
                    dx = st_after["x"] - x_before
                    trace["movement_dx_px"] = float(dx)
                    print(f"Movement verification: dx={dx:.1f}px, state={st_after}")
                    if dx < 10.0 or not _plausible():
                        _record_attempt(
                            trace, outcome=f"blocked: movement verification failed (dx={dx:.1f} px)"
                        )
                        raise RuntimeError(
                            f"Level {level}: verification failed (dx={dx:.1f}). "
                            "Savestate NOT saved."
                        )
                # Wait a few frames for the fade-in to settle
                for _ in range(60):
                    emu.set_input({})
                    emu.step_frame()
                level_loaded = True
                if level == 2:
                    _record_attempt(trace, outcome="captured: playable handoff verified")
                break

    if not level_loaded:
        game_mode = emu.get_game_mode()
        _record_attempt(
            trace,
            outcome=f"blocked: never entered interactive mode (final mode 0x{game_mode:02X})",
        )
        raise RuntimeError(
            f"Level {level}: interactive gameplay mode "
            f"0x{wram.GAME_MODE_INTERACTIVE:02X} never reached "
            f"(final mode 0x{game_mode:02X}). Savestate NOT saved."
        )

    state_bytes = emu.save_state()
    with open(state_out_path, "wb") as f:
        f.write(state_bytes)

    final_state = emu.get_smw_state()
    print(f"Savestate saved successfully to {state_out_path} ({len(state_bytes)} bytes)!")
    print(f"Mario initial state: {final_state}")
    emu.close()
    return final_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Boot SMW and capture a level savestate.")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--state-out-path", default=None)
    args = parser.parse_args()
    out = args.state_out_path or (STATE_YOSHI_ISLAND_2 if args.level == 2 else STATE_YOSHI_ISLAND_1)
    boot_and_enter_level(state_out_path=out, level=args.level)
