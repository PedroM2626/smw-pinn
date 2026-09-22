"""
diagnose_obstacle_1000.py
Formal diagnosis of the extended-navigation bottleneck at X ~= 1000 px.

Runs scripted gameplay from the Yoshi's Island 1 savestate until Mario
reaches X >= 900, then records, for X in [850, 1200]:
  - local WRAM tile patches ($7E:C800) and ground profile,
  - the 12-slot sprite table ($7E:14C8 / $7E:00E4 / $7E:00D8) per frame,
  - Mario kinematics (x, y, vx, vy).

Outputs `results/obstacle_1000_diagnosis.json` + `.png`: ground profile with
hazard encounters marked, plus the minimum takeoff velocity needed to clear
each detected ground gap under SMW projectile physics
(v0 in [-80, -56], g_held = +3, g_fall = +6 subpx/frame^2, term +64).
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from src.environment.snes_emulator import SnesLibretroEmulator
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

X_START_SCAN = 850.0
X_END_SCAN = 1200.0
GRAV_HOLD = 3.0
GRAV_FALL = 6.0
VY_JUMP = -72.0


def min_takeoff_vx(gap_width_px: float, vy0: float = VY_JUMP) -> float:
    """Minimum horizontal speed (subpx/frame) to clear a gap of given width.

    Airtime from projectile motion with asymmetric gravity: ascent with
    g_hold while vy < 0, descent with g_fall. Solves t_air(vy0), then
    vx_min = gap_px * 16 / t_air.
    """
    t_up = -vy0 / GRAV_HOLD
    y_apex = vy0 * t_up + 0.5 * GRAV_HOLD * t_up * t_up  # negative (up)
    # Fall back to launch height: t_down solves 0.5*g*t^2 = -y_apex
    t_down = math.sqrt(max(0.0, -2.0 * y_apex / GRAV_FALL))
    t_air = t_up + t_down
    if t_air <= 0:
        return float("inf")
    return gap_width_px * 16.0 / t_air


def run_diagnosis(
    core_path: str = "src/environment/bin/snes9x_libretro.dll",
    rom_path: str = "data/raw/smw_usa.sfc",
    state_path: str = "data/raw/smw_yoshi_island_1.state",
    output_dir: str = "results",
    seed: int = 7,
) -> dict:
    set_global_seed(seed)
    emu = SnesLibretroEmulator(core_path)
    emu.load_rom(rom_path)
    with open(state_path, "rb") as f:
        emu.load_state(f.read())
    emu.wram_buffer[0x0100] = 0x14
    for _ in range(5):
        emu.step_frame()

    # 1. Scripted drive to the scan window (hold run + periodic jumps).
    for frame in range(3000):
        st = emu.get_smw_state()
        if st["x"] >= X_START_SCAN:
            break
        emu.set_input({"RIGHT": True, "Y": True, "B": frame % 50 < 25})
        emu.step_frame()
    else:
        raise RuntimeError("Scripted drive never reached the scan window.")

    # 2. Slow scan with full telemetry.
    samples = []
    for _ in range(1200):
        st = emu.get_smw_state()
        if st["x"] > X_END_SCAN:
            break
        sprites = emu.get_active_sprites()
        patch = emu.get_local_tilemap_patch(st["x"], st["y"], radius=3)
        samples.append(
            {
                "x": st["x"], "y": st["y"], "vx": st["vx"], "vy": st["vy"],
                "c_ground": st["c_ground"],
                "hazards": [
                    {"id": s["id"], "x": s["x"], "y": s["y"], "vx": s["vx"]}
                    for s in sprites
                ],
                "ground_below": bool((patch[4:, 3] == 1).any()),
            }
        )
        emu.set_input({"RIGHT": True, "Y": True, "B": st["c_ground"] > 0.5})
        emu.step_frame()
    emu.close()
    if not samples:
        raise RuntimeError("No samples collected in scan window.")

    # 3. Ground-gap detection: stretches without ground below while airborne.
    # NOTE (honest limitation): "gap" = no solid tile in the 3 rows below
    # Mario's *current* position. A jump arc over flat ground also reads as a
    # gap, so each record carries grounding context + vertical drop to tell
    # jump arcs (grounded at entry, y recovers) from true pits (y falls).
    xs = np.array([s["x"] for s in samples])
    ys = np.array([s["y"] for s in samples])
    grounded = np.array([s["ground_below"] for s in samples], dtype=bool)
    gaps = []
    i = 0
    while i < len(xs):
        if not grounded[i]:
            j = i
            while j < len(xs) and not grounded[j]:
                j += 1
            k = min(j, len(xs) - 1)
            width = float(xs[k] - xs[i])
            if width > 8.0:
                gaps.append(
                    {
                        "x_start": float(xs[i]),
                        "x_end": float(xs[k]),
                        "width_px": width,
                        "y_start": float(ys[i]),
                        "y_end": float(ys[k]),
                        "y_drop_px": float(ys[k] - ys[i]),
                        "min_takeoff_vx_subpx": float(min_takeoff_vx(width)),
                    }
                )
            i = j
        else:
            i += 1

    # 4. Hazard encounters inside the window.
    encounters = []
    for s in samples:
        for hz in s["hazards"]:
            dx = hz["x"] - s["x"]
            if -16.0 < dx < 96.0:
                encounters.append({"mario_x": s["x"], "id": hz["id"], "dx": dx})
                break

    payload = {
        "num_samples": len(samples),
        "x_range": [float(xs[0]), float(xs[-1])],
        "ground_gaps": gaps,
        "num_hazard_encounters": len(encounters),
        "hazard_encounters": encounters[:40],
        "rex_present": any(e["id"] == 0x05 for e in encounters),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "obstacle_1000_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 5. Figure: ground profile + gaps + hazard markers.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xs, [1.0 if g else 0.0 for g in grounded], lw=1.5, label="ground below")
    for g in gaps:
        ax.axvspan(g["x_start"], g["x_end"], color="red", alpha=0.3)
        ax.text(
            (g["x_start"] + g["x_end"]) / 2, 0.5,
            f"gap {g['width_px']:.0f}px\nvx>={g['min_takeoff_vx_subpx']:.0f}",
            ha="center", fontsize=8,
        )
    for e in encounters[:: max(1, len(encounters) // 30)]:
        ax.plot(e["mario_x"], 0.9, "ro", ms=3)
    ax.set_xlabel("Mario X (px)")
    ax.set_ylabel("ground contact")
    ax.set_title("Obstacle diagnosis: ground gaps + hazard encounters (X 850-1200)")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "obstacle_1000_diagnosis.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Diagnosis: {len(gaps)} gaps, {len(encounters)} encounters -> {fig_path}")
    return payload


if __name__ == "__main__":
    run_diagnosis()
