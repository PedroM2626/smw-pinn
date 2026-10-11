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
    """Persist the capture diagnostics so a blocked claim stays checkable.

    Both stages now have a frontier to document: Yoshi's Island 2 has never been entered at all
    (§10.36), and Yoshi's Island 1's *graphical* entry turned out to be unverified - the honesty gate
    below refuses a state whose player does not walk, and the only 0x14 the boot reaches is the title
    screen's attract demo, whose Mario stops responding after one scripted stride.
    """
    from src.utils.paths import results_file
    from src.utils.provenance import write_metrics

    name = (
        "yi2_capture_attempt.json" if trace.get("target_level") == 2 else "yi1_capture_attempt.json"
    )
    write_metrics(
        results_file(name),
        {"outcome": outcome, **trace},
        command=f"python -m scripts.navigate_to_level --level {trace['target_level']}",
    )


def _plausible(emu: SnesLibretroEmulator) -> bool:
    """Mode 0x14 with non-degenerate coordinates.

    The animation byte used to veto this check as well (excluding 0, 0x0B and 0x0C), on the strength
    of the Yoshi's Island 2 campaign. That exclusion is now known to be wrong: a Mario standing on
    solid ground in a level that responds to input reads 0. What distinguishes a frozen frame from a
    playable one is not any static byte but whether the world moves under a held button, which is
    what `verify_control_handoff` measures, so the byte is recorded and no longer decides anything.
    """
    s = emu.get_smw_state()
    return (
        emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE
        and 20.0 < s["x"] < 4000.0
        and 0.0 < s["y"] < 450.0
    )


def _right_response(emu: SnesLibretroEmulator) -> float:
    """How far Mario advances under 30 frames of RIGHT: the only test that distinguishes a live
    simulation from a frozen one, since a savestate taken during the entry slide or behind the
    welcome text box reports plausible coordinates and integrates nothing."""
    x_before = emu.get_smw_state()["x"]
    for _ in range(30):
        emu.set_input({"RIGHT": True})
        emu.step_frame()
    for _ in range(12):
        emu.set_input({})
        emu.step_frame()
    return emu.get_smw_state()["x"] - x_before


#: the first node past Yoshi's House on a fresh file, measured by polling $7E:1F17/$7E:1F19 while
#: walking the map (the cursor lands there after the house exit and the path settles).
MAP_X_FIRST_LEVEL = 152
MAP_Y_FIRST_LEVEL = 136


def _hold(emu: SnesLibretroEmulator, buttons: dict, frames: int) -> None:
    for _ in range(frames):
        emu.set_input(buttons)
        emu.step_frame()


def _drive(
    emu: SnesLibretroEmulator,
    buttons: dict,
    until,
    budget: int,
    on: int = 5,
    off: int = 7,
) -> int:
    """Pulse a button with release gaps until `until()` holds; -1 when the budget runs out.

    The release gap is not decoration: a held START never leaves the title screen, while 5-on/7-off
    pulses walk the mode byte through the boot states. Measured 2026-10-11 by polling $7E:0100.
    """
    for f in range(budget):
        if until():
            return f
        pulsing = (f % (on + off)) < on
        _hold(emu, buttons if pulsing else {}, on if pulsing else off)
    return -1


def walk_to_overworld(emu: SnesLibretroEmulator, trace: dict) -> bool:
    """Play the menus from power-on until the world map (mode 0x0E) is live.

    The route is: sit out the black intro, pulse START until the map appears, enter Yoshi's House
    with A, walk out of it with RIGHT, and let the path settle. Pressing B where A belongs, or
    treating only 0x07/0x08 as "still in the menu", ends up in the title screen's attract demo
    instead - a rendered level whose player is scripted and stops responding after one stride.
    """
    _hold(emu, {}, 600)  # the ROM intro is black for about five seconds
    to_map = _drive(
        emu, {"START": True}, lambda: emu.get_game_mode() == wram.GAME_MODE_WORLD_MAP, 3000
    )
    if to_map < 0:
        trace["overworld_frames"] = None
        return False
    _hold(emu, {}, 60)
    into_house = _drive(
        emu, {"A": True}, lambda: emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE, 900
    )
    _hold(emu, {}, 30)
    out_of_house = _drive(
        emu,
        {"RIGHT": True},
        lambda: emu.get_game_mode() == wram.GAME_MODE_WORLD_MAP,
        1500,
        on=20,
        off=4,
    )
    _hold(emu, {}, 300)  # the path to the first real node opens after the house exit
    trace["overworld_frames"] = to_map
    trace["house_entry_frames"] = into_house
    trace["house_exit_frames"] = out_of_house
    _hold(emu, {"RIGHT": True}, 40)  # step off the house node onto the path
    _hold(emu, {}, 200)  # let the path to the first level finish opening
    return emu.get_game_mode() == wram.GAME_MODE_WORLD_MAP


def warp_on_map(emu: SnesLibretroEmulator, map_x: int, map_y: int) -> bool:
    """Park the overworld cursor on a node and enter it with A.

    The game derives the level from wherever the cursor is on the frame the button is pressed, so
    writing the translevel byte does nothing; the cursor write is what works. Nodes that are not
    unlocked (or not level tiles) never leave the map, or load a blank level with a zero timer.
    """
    if emu.get_game_mode() != wram.GAME_MODE_WORLD_MAP:
        return False
    emu.write_wram_u16_le(wram.ADDR_MAP_X, map_x)
    emu.write_wram_u16_le(wram.ADDR_MAP_Y, map_y)
    entered = _drive(
        emu, {"A": True}, lambda: emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE, 900
    )
    if entered < 0:
        return False
    _hold(emu, {}, 120)  # let the fade in and Mario land
    return emu.get_game_mode() == wram.GAME_MODE_INTERACTIVE


def verify_control_handoff(
    emu: SnesLibretroEmulator, trace: dict, level: int, dismiss_text_box: bool = False
) -> bytes:
    """Refuse to save a state the engine is not actually playing, and return the pre-probe state.

    The returned bytes are the machine state as it stood *before* the probes walked Mario: a
    savestate taken after the verification starts the episode hundreds of pixels into the level,
    which is not what "the level start" means in this repository's recordings.

    A pure idle settle reaches a stable-but-frozen frame - vy constant, x unchanged under RIGHT -
    because the engine is not stepping Mario's physics: either the entry slide is still running, or
    (Yoshi's Island 1, measured 2026-10-11) the welcome text box owns the screen. Both look like a
    level to every static read of WRAM, so the gate is behavioural: Mario has to walk.

    For level 2 the prior campaign also found that a Y *edge* on the entry slide fires a
    0x14 -> 0x0C transition back to the world map, so nothing here pulses Y.
    """
    if dismiss_text_box:
        print("Dismissing the level-introduction text box (B pulses)...")
        for _ in range(30):
            if _right_response(emu) > 6.0:
                break
            for _ in range(8):
                emu.set_input({"B": True})
                emu.step_frame()
            for _ in range(12):
                emu.set_input({})
                emu.step_frame()
    else:
        for _ in range(180):
            emu.set_input({})
            emu.step_frame()

    # Everything after this point walks Mario down the stage, so remember where the level starts.
    level_start_state = emu.save_state()
    responsive = _right_response(emu)
    for _ in range(3):
        if _plausible(emu) and responsive > 6.0:
            break
        for _ in range(6):
            emu.set_input({"START": True})
            emu.step_frame()
        for _ in range(6):
            emu.set_input({})
            emu.step_frame()
        responsive = _right_response(emu)
    st = emu.get_smw_state()
    trace["post_message_state"] = {k: float(v) for k, v in st.items()}
    trace["right_response_px"] = float(responsive)
    print(f"Post-settle state: {st}, right-press dx={responsive:.1f}")
    if not (_plausible(emu) and responsive > 6.0):
        _record_attempt(
            trace,
            outcome=(
                f"blocked: no control handoff (right dx={responsive:.1f}, air={st['air_state']:.0f})"
            ),
        )
        raise RuntimeError(
            f"Level {level}: engine never handed control (dx={responsive:.1f} px). Savestate NOT saved."
        )
    trace["stable_frames"] = 60  # control response verified just above
    x_before = emu.get_smw_state()["x"]
    for _ in range(90):
        emu.set_input({"RIGHT": True, "Y": True})
        emu.step_frame()
    st_after = emu.get_smw_state()
    dx = st_after["x"] - x_before
    trace["movement_dx_px"] = float(dx)
    print(f"Movement verification: dx={dx:.1f}px, state={st_after}")
    if dx < 10.0 or not _plausible(emu):
        _record_attempt(trace, outcome=f"blocked: movement verification failed (dx={dx:.1f} px)")
        raise RuntimeError(
            f"Level {level}: verification failed (dx={dx:.1f}). Savestate NOT saved."
        )
    emu.load_state(level_start_state)
    return level_start_state


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

    if level == 1:
        # The measured route from power-on: sit out the intro, pulse START with release gaps until
        # the world map is live, enter and leave Yoshi's House, then enter the first node. The
        # schedule below (START/B holds, B to "close" the map intro) walks the mode byte into the
        # title screen's attract demo instead, which renders a level whose player stops responding.
        print("Playing the menus to the world map...")
        if not walk_to_overworld(emu, trace):
            _record_attempt(trace, outcome="blocked: world map (0x0E) never reached")
            raise RuntimeError(
                f"Level {level}: the world map was never reached "
                f"(final mode 0x{emu.get_game_mode():02X}). Savestate NOT saved."
            )
        print(f"World map live at frame {trace['overworld_frames']}.")
        if not warp_on_map(emu, MAP_X_FIRST_LEVEL, MAP_Y_FIRST_LEVEL):
            _record_attempt(
                trace,
                outcome=f"blocked: node ({MAP_X_FIRST_LEVEL},{MAP_Y_FIRST_LEVEL}) did not enter a level",
            )
            raise RuntimeError(
                f"Level {level}: the first node past the house entered nothing "
                f"(final mode 0x{emu.get_game_mode():02X}). Savestate NOT saved."
            )
        trace["entry_frame"] = 0
        trace["mode_at_entry"] = hex(emu.get_game_mode())
        trace["x_at_entry"] = int(emu.read_wram_u16_le(wram.ADDR_PLAYER_X))
        trace["y_at_entry"] = int(emu.read_wram_u16_le(wram.ADDR_PLAYER_Y))
        trace["air_state_at_entry"] = emu.read_wram_u8(wram.ADDR_AIR_STATE)
        trace["translevel"] = f"0x{emu.read_wram_u8(wram.ADDR_TRANSLEVEL):02X}"
        print(
            f"Level entered from cursor ({MAP_X_FIRST_LEVEL},{MAP_Y_FIRST_LEVEL}): "
            f"mode 0x{emu.get_game_mode():02X}, translevel {trace['translevel']}, "
            f"X={trace['x_at_entry']}, Y={trace['y_at_entry']}"
        )
        state_bytes = verify_control_handoff(emu, trace, level, dismiss_text_box=False)
        with open(state_out_path, "wb") as f:
            f.write(state_bytes)
        print(f"Savestate saved successfully to {state_out_path} ({len(state_bytes)} bytes)!")
        final_state = emu.get_smw_state()
        print(f"Mario initial state: {final_state}")
        emu.close()
        return final_state

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

    # Wait for the level transition ($7E:0100 == 0x14 is Level Game Mode)
    print("Waiting for the level-entry transition ($7E:0100 == 0x14)...")
    max_frames = 1500
    level_loaded = False
    verified_start: bytes | None = None

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
                verified_start = verify_control_handoff(
                    emu, trace, level, dismiss_text_box=(level == 1)
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

    # The handoff probes walk Mario down the stage; save the state taken before them.
    state_bytes = verified_start if verified_start is not None else emu.save_state()
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
