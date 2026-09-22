"""
render_level_clearance_video.py
Renders a publication-grade animated video (MP4) and GIF of Mario's full level
clearance trajectory on authentic SNES hardware, featuring live WRAM telemetry HUD:
- Spatial Subscreen map (Subscreens 0 to 7)
- Altitude profile with parabolic jump arcs and collision ground line
- Telemetry HUD: Coordinates (X, Y), Velocities (vx, vy), Active Hazard distance, Joypad inputs
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional
import imageio
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("."))


def render_trajectory_video(
    metrics_path: str = "results/full_level_clearance_metrics.json",
    trajectory_data_path: str = "results/full_level_trajectory_log.json",
    output_mp4: str = "results/figures/full_level_clearance.mp4",
    output_gif: str = "results/figures/full_level_clearance.gif",
    fps: int = 30,
    stride: int = 2,  # Subsample frames (e.g. 60 FPS -> 30 FPS video)
):
    print("====================================================================")
    print("  RENDERING FULL LEVEL CLEARANCE VIDEO & TELEMETRY HUD ANIMATION    ")
    print("====================================================================")

    if not os.path.exists(trajectory_data_path):
        print(f"Error: Trajectory log file not found at: {trajectory_data_path}")
        return

    with open(trajectory_data_path, "r") as f:
        traj_data = json.load(f)

    frames = traj_data["frames"]
    x_vals = traj_data["x"]
    y_vals = traj_data["y"]
    vx_vals = traj_data["vx"]
    vy_vals = traj_data["vy"]
    hazard_dx = traj_data.get("hazard_dx", [999.0] * len(frames))
    actions_log = traj_data.get("actions", [{"RIGHT": True}] * len(frames))

    total_frames = len(frames)
    indices = list(range(0, total_frames, stride))
    print(f"Total hardware frames: {total_frames} | Sampled video frames: {len(indices)} | Output FPS: {fps}")

    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    temp_frames = []

    fig, (ax_map, ax_alt) = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"height_ratios": [1.5, 1.2]})

    t0 = time.time()
    for i, idx in enumerate(indices):
        ax_map.clear()
        ax_alt.clear()

        curr_frame = frames[idx]
        curr_x = x_vals[idx]
        curr_y = y_vals[idx]
        curr_vx = vx_vals[idx]
        curr_vy = vy_vals[idx]
        curr_hdx = hazard_dx[idx]
        curr_act = actions_log[idx]

        # Top Panel: Level Progression Map across Subscreens
        ax_map.set_facecolor("#1a1a24")
        ax_map.set_xlim(-50, 2050)
        ax_map.set_ylim(-10, 50)

        # Draw subscreen demarcations
        for s in range(8):
            ax_map.axvline(x=s * 256, color="#444455", linestyle=":", alpha=0.6)
            ax_map.text(s * 256 + 10, 42, f"Subscreen {s}", color="#8888aa", fontsize=8, fontweight="bold")

        # Ground representation
        ax_map.axhline(y=0, color="#228b22", linewidth=6)
        ax_map.axvline(x=1900, color="#ffd700", linestyle="-.", linewidth=2.0)
        ax_map.text(1905, 20, "GOAL TAPE", color="#ffd700", fontsize=9, fontweight="bold", rotation=90)

        # Trajectory trail
        past_x = x_vals[:idx+1]
        past_y_map = [5.0] * len(past_x)
        ax_map.plot(past_x, past_y_map, color="#00ffff", linewidth=2.0, alpha=0.7)

        # Mario marker
        ax_map.scatter([curr_x], [5.0], color="#ff3333", s=120, edgecolors="white", linewidths=1.5, zorder=5)

        # Active Rex marker if in proximity
        if curr_hdx < 300.0:
            enemy_x = curr_x + curr_hdx
            ax_map.scatter([enemy_x], [5.0], color="#ff9900", s=90, marker="s", edgecolors="black", zorder=4)

        ax_map.set_title(
            f"Super Mario World Hardware Trajectory | Frame: {curr_frame:4d} | X: {curr_x:6.1f} px | "
            f"vx: {curr_vx:4.1f} subpx/f | Subscreen: {int(curr_x)//256}",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        ax_map.set_xticks(range(0, 2049, 256))
        ax_map.set_yticks([])

        # Bottom Panel: Parabolic Jump Arc and Altitude Y
        ax_alt.set_facecolor("#111118")
        ax_alt.set_xlim(max(0, curr_frame - 180), curr_frame + 20)
        ax_alt.set_ylim(260, 440)
        ax_alt.invert_yaxis()

        past_frames = frames[:idx+1]
        past_y = y_vals[:idx+1]

        ax_alt.plot(past_frames, past_y, color="#39ff14", linewidth=2.0, label="Mario Altitude Y")
        ax_alt.axhline(y=384.0, color="#8b4513", linestyle="--", linewidth=1.5, label="Solid Ground (Y=384)")
        ax_alt.scatter([curr_frame], [curr_y], color="red", s=70, zorder=5)

        # Telemetry Text Box
        btn_str = " ".join([k for k, v in curr_act.items() if v])
        telemetry_txt = (
            f"Telemetria WRAM:\n"
            f"Pos X: {curr_x:6.1f} px\n"
            f"Pos Y: {curr_y:6.1f} px\n"
            f"Vel vx: {curr_vx:5.1f}\n"
            f"Vel vy: {curr_vy:5.1f}\n"
            f"Dist Rex: {curr_hdx:5.1f} px\n"
            f"Inputs: [{btn_str}]"
        )
        ax_alt.text(
            0.02, 0.95, telemetry_txt,
            transform=ax_alt.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
            color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#222233", alpha=0.8, edgecolor="#555577"),
        )

        ax_alt.set_xlabel("Hardware Simulation Frames (60 Hz)", fontsize=9, fontweight="bold", color="white")
        ax_alt.set_ylabel("Altitude Y (px)", fontsize=9, fontweight="bold", color="white")
        ax_alt.legend(loc="upper right", facecolor="#222233", edgecolor="#555577", labelcolor="white")
        ax_alt.tick_params(colors="white")
        ax_map.tick_params(colors="white")

        fig.patch.set_facecolor("#0a0a0f")
        plt.tight_layout()

        # Canvas to numpy RGB array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        temp_frames.append(rgb)

    plt.close(fig)
    print(f"Generated {len(temp_frames)} video frames in {time.time() - t0:.1f}s. Encoding video...")

    # Write MP4
    try:
        imageio.mimwrite(output_mp4, temp_frames, fps=fps, quality=8)
        print(f"MP4 Video saved successfully to: {output_mp4}")
    except Exception as e:
        print(f"MP4 encoding notice: {e}")

    # Write GIF (subsampled for compactness)
    gif_frames = temp_frames[::2]  # Subsample for lightweight GIF
    imageio.mimsave(output_gif, gif_frames, fps=fps//2, loop=0)
    print(f"Animated GIF saved successfully to: {output_gif}")


if __name__ == "__main__":
    render_trajectory_video()
