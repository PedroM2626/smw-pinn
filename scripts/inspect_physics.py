import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.environment.snes_emulator import SnesLibretroEmulator


def main():
    emu = SnesLibretroEmulator(r"src\environment\bin\snes9x_libretro.dll")
    emu.load_rom(r"data\raw\smw_usa.sfc")

    print("Avançando para frame 420 (Modo de jogo 0x07)...")
    for _ in range(420):
        emu.step_frame()

    print(f"Modo de jogo: 0x{emu.read_wram_u8(0x0100):02X}")
    print("\n--- AMOSTRA DE 30 QUADROS DA FÍSICA GENUÍNA DO JOGO ---")
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
            f"{420+i:>6} | {s['x']:>8.2f} | {s['y']:>8.2f} | {s['vx']:>6.1f} | {s['vy']:>6.1f} | {int(s['c_ground']):>6} | {dx:>8.2f} | {expected_dx:>8.2f}"
        )
        prev_x = s["x"]
        emu.step_frame()

    emu.close()


if __name__ == "__main__":
    main()
