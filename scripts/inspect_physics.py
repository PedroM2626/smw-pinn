"""inspect_physics.py
Print a raw WRAM physics sample from the running game to eyeball the fixed-point
kinematics (dx per frame vs. v_x / 16) on real console memory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment.snes_emulator import SnesLibretroEmulator


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emu = SnesLibretroEmulator(
        os.path.join(repo_root, "src", "environment", "bin", "snes9x_libretro.dll")
    )
    emu.load_rom(os.path.join(repo_root, "data", "raw", "smw_usa.sfc"))

    print("Advancing to frame 420 (game mode 0x07)...")
    for _ in range(420):
        emu.step_frame()

    print(f"Game mode: 0x{emu.get_game_mode():02X}")
    print("\n--- 30-FRAME SAMPLE OF THE GAME'S GENUINE PHYSICS ---")
    print(
        f"{'Frame':>6} | {'X':>8} | {'Y':>8} | {'v_x':>6} | {'v_y':>6} | {'Ground':>6} | {'dx (pix)':>8} | {'v_x / 16':>8}"
    )
    print("-" * 65)

    prev_x = None
    for i in range(30):
        s = emu.get_smw_state()
        dx = (s["x"] - prev_x) if prev_x is not None else 0.0
        expected_dx = s["vx"] / 16.0
        print(
            f"{420 + i:>6} | {s['x']:>8.2f} | {s['y']:>8.2f} | {s['vx']:>6.1f} | {s['vy']:>6.1f} | {int(s['c_ground']):>6} | {dx:>8.2f} | {expected_dx:>8.2f}"
        )
        prev_x = s["x"]
        emu.step_frame()

    emu.close()


if __name__ == "__main__":
    main()
