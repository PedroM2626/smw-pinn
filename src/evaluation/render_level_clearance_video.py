"""
render_level_clearance_video.py
High-impact publication-grade telemetry video & animated GIF renderer.
Features:
1. Dynamic Side-Scrolling Camera that tracks Mario in (X, Y) level space
   with scrolling terrain blocks, Rex hazard proximity, and parabolic jump arcs.
2. Prominent Real-Time Distance Gauge (0 -> 833.5 px, Subscreens 0 to 3, Milestones).
3. Continuous Horizontal Progress Curve X(t) with moving time cursor.
4. Live Joypad input HUD with illuminated buttons (B: Jump, Y: Dash, RIGHT).
"""

import json
import os
import time

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging import get_logger
from src.utils.paths import (
    figure_file,
    results_file,
)

logger = get_logger(__name__)


def render_dynamic_telemetry_video(
    trajectory_data_path: str = results_file("full_level_trajectory_log.json"),
    output_mp4: str = figure_file("full_level_clearance.mp4"),
    output_gif: str = figure_file("full_level_clearance.gif"),
    fps: int = 30,
    stride: int = 3,
    max_render_frame: int = 3000,  # Full stage clearance
):
    logger.info("====================================================================")
    logger.info("  RENDERING DYNAMIC SIDE-SCROLLING TELEMETRY VIDEO & GIF ANIMATION  ")
    logger.info("====================================================================")

    if not os.path.exists(trajectory_data_path):
        logger.info(f"Error: Trajectory log not found: {trajectory_data_path}")
        return

    with open(trajectory_data_path, "r") as f:
        traj = json.load(f)

    all_frames = traj["frames"]
    all_x = traj["x"]
    all_y = traj["y"]
    all_vx = traj["vx"]
    all_vy = traj["vy"]
    all_hdx = traj.get("hazard_dx", [999.0] * len(all_frames))
    all_actions = traj.get("actions", [{}] * len(all_frames))

    # Slice up to max_render_frame to capture the active forward progress
    end_idx = min(len(all_frames), max_render_frame)
    frames = all_frames[:end_idx]
    x_vals = all_x[:end_idx]
    y_vals = all_y[:end_idx]
    vx_vals = all_vx[:end_idx]
    vy_vals = all_vy[:end_idx]
    hdx_vals = all_hdx[:end_idx]
    actions = all_actions[:end_idx]

    indices = list(range(0, len(frames), stride))
    logger.info(
        f"Rendering {len(indices)} frames at {fps} FPS (active run: 0 -> {x_vals[-1]:.1f} px)..."
    )

    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    rendered_frames = []

    # Setup 3-panel figure:
    # 1. Top: Dynamic Side-Scrolling World View (Follows Mario)
    # 2. Middle: Real-Time Cumulative Progress X(t) with time cursor
    # 3. Bottom: Telemetry Dashboard (Speedometer, Buttons, Status)
    fig = plt.figure(figsize=(13, 8), facecolor="#0d0e15")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.3, 0.8], hspace=0.35)
    ax_cam = fig.add_subplot(gs[0])
    ax_prog = fig.add_subplot(gs[1])
    ax_hud = fig.add_subplot(gs[2])

    t_start = time.time()

    for step_i, idx in enumerate(indices):
        ax_cam.clear()
        ax_prog.clear()
        ax_hud.clear()

        curr_frame = frames[idx]
        curr_x = x_vals[idx]
        curr_y = y_vals[idx]
        curr_vx = vx_vals[idx]
        curr_vy = vy_vals[idx]
        curr_hdx = hdx_vals[idx]
        curr_act = actions[idx]

        # -------------------------------------------------------------
        # PANEL 1: DYNAMIC SIDE-SCROLLING VIEWPORT (TRACKING CAMERA)
        # -------------------------------------------------------------
        ax_cam.set_facecolor("#151824")
        cam_x_min = curr_x - 120
        cam_x_max = curr_x + 280
        ax_cam.set_xlim(cam_x_min, cam_x_max)
        ax_cam.set_ylim(440, 240)  # Inverted Y (SNES coordinates)

        # Draw level ground and terrain blocks across the level
        ax_cam.axhline(y=384, color="#4a7c59", linewidth=8, label="Base Ground Level (Y=384)")
        ax_cam.fill_between([cam_x_min - 50, cam_x_max + 50], 384, 450, color="#2d5037", alpha=0.9)

        # Elevated platforms across Yoshi's Island 1:
        # Platform 1: X ~ 380 - 850 (Y = 352)
        ax_cam.fill_between([380, 850], 352, 384, color="#8b5a2b", alpha=0.6)
        # Platform 2: X ~ 850 - 1024 (High Ridge Y = 272)
        ax_cam.fill_between([850, 1024], 272, 384, color="#6b4423", alpha=0.6)
        # Platform 3: X ~ 1250 - 1450 (Middle Ridge Y = 304)
        ax_cam.fill_between([1250, 1450], 304, 384, color="#8b5a2b", alpha=0.6)

        # Subscreen grid markers (Subscreen 0 to 7)
        for s in range(9):
            sx = s * 256
            if cam_x_min - 50 <= sx <= cam_x_max + 50:
                ax_cam.axvline(x=sx, color="#445577", linestyle=":", linewidth=1.5, alpha=0.7)
                ax_cam.text(
                    sx + 5, 255, f"Subscreen {s}", color="#7799cc", fontsize=9, fontweight="bold"
                )

        # Stage Milestones
        stage_milestones = [
            (250, "250px"),
            (500, "500px"),
            (782, "782px Rex Evasion"),
            (1000, "1000px Plateau Drop"),
            (1250, "1250px Pipe Valley"),
            (1500, "1500px Upper Slopes"),
            (1750, "1750px Final Stretch"),
            (1950, "GOAL TAPE CLEAR"),
        ]
        for m, m_name in stage_milestones:
            if cam_x_min - 50 <= m <= cam_x_max + 50:
                ax_cam.axvline(x=m, color="#f39c12", linestyle="-.", linewidth=1.5)
                ax_cam.text(
                    m + 4, 275, f"[{m_name}]", color="#f39c12", fontsize=8, fontweight="bold"
                )

        # Goal Tape Structure at X ~ 1950 px
        if cam_x_min - 50 <= 1950 <= cam_x_max + 50:
            ax_cam.plot([1945, 1945], [260, 384], color="#ffffff", linewidth=4.0, zorder=4)
            ax_cam.plot([1965, 1965], [260, 384], color="#ffffff", linewidth=4.0, zorder=4)
            ax_cam.fill_between([1945, 1965], 275, 288, color="#ffdd00", alpha=0.9, zorder=5)
            ax_cam.text(
                1955,
                270,
                "GOAL",
                color="#ffdd00",
                fontsize=9,
                fontweight="bold",
                ha="center",
                zorder=6,
            )

        # Mario's historical jump trajectory ribbon
        trail_start = max(0, idx - 80)
        ax_cam.plot(
            x_vals[trail_start : idx + 1],
            y_vals[trail_start : idx + 1],
            color="#00ffcc",
            linewidth=2.5,
            alpha=0.85,
            label="Trajectory Ribbon",
        )

        # Draw Mario Agent
        ax_cam.scatter(
            [curr_x], [curr_y], color="#ff3344", s=180, edgecolors="white", linewidths=2.0, zorder=6
        )
        ax_cam.text(
            curr_x - 15,
            curr_y - 14,
            "MARIO",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=7,
        )

        # Stage Clearance Victory Banner if Mario has crossed Goal Tape
        if curr_x >= 1920.0:
            ax_cam.text(
                cam_x_min + 20,
                260,
                "★ STAGE CLEARED! LEVEL FINISHED ★",
                color="#f1c40f",
                fontsize=11,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#1a1c29",
                    edgecolor="#f1c40f",
                    linewidth=1.8,
                ),
                zorder=10,
            )

        # Draw Dynamic Hazard (Rex) if in camera range
        if curr_hdx < 350.0:
            enemy_x = curr_x + curr_hdx
            if cam_x_min - 50 <= enemy_x <= cam_x_max + 50:
                ax_cam.scatter(
                    [enemy_x],
                    [368],
                    color="#ff9900",
                    s=150,
                    marker="D",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )
                ax_cam.text(
                    enemy_x - 12, 350, "REX", color="#ffcc00", fontsize=8, fontweight="bold"
                )
                # Distance arrow
                ax_cam.annotate(
                    f"{curr_hdx:.1f}px",
                    xy=(enemy_x, 335),
                    xytext=(curr_x, 335),
                    arrowprops=dict(arrowstyle="<->", color="#ffaa00", lw=1.2),
                    color="#ffcc00",
                    fontsize=8,
                    ha="center",
                )

        ax_cam.set_title(
            f"LIVE TRACKING VIEWPORT (Follows Mario) | Yoshi's Island 1 | Frame: {curr_frame:4d} (60Hz)",
            fontsize=11,
            fontweight="bold",
            color="white",
            pad=8,
        )
        ax_cam.set_ylabel("Altitude Y (px)", color="#cccccc", fontsize=9, fontweight="bold")
        ax_cam.tick_params(colors="#888899", labelsize=8)
        ax_cam.legend(
            loc="upper right",
            facecolor="#1e2233",
            edgecolor="#334466",
            labelcolor="white",
            fontsize=8,
        )

        # -------------------------------------------------------------
        # PANEL 2: HORIZONTAL PROGRESS CURVE X(t) WITH MOVING CURSOR
        # -------------------------------------------------------------
        ax_prog.set_facecolor("#151824")
        ax_prog.set_xlim(0, frames[-1] + 10)
        ax_prog.set_ylim(-20, max(x_vals) + 60)

        # Full progress curve in background
        ax_prog.plot(frames, x_vals, color="#334466", linewidth=1.5, linestyle="--", alpha=0.6)
        # Active progress curve up to current frame
        ax_prog.plot(
            frames[: idx + 1],
            x_vals[: idx + 1],
            color="#2ecc71",
            linewidth=2.8,
            label="Real SNES Progress X(t)",
        )

        # Moving time cursor
        ax_prog.axvline(x=curr_frame, color="#ff3344", linestyle=":", linewidth=2.0)
        ax_prog.scatter([curr_frame], [curr_x], color="#ff3344", s=90, zorder=5)

        # Milestone horizontal levels
        for m, m_name in [(500, "500px"), (1000, "1000px"), (1500, "1500px"), (1950, "Goal Tape")]:
            ax_prog.axhline(y=m, color="#e67e22", linestyle=":", alpha=0.5)
            ax_prog.text(5, m + 8, f"{m} px Milestone", color="#e67e22", fontsize=7.5)

        ax_prog.set_title(
            f"Horizontal Level Progression: {curr_x:6.1f} px / {x_vals[-1]:.1f} px | Velocity vx: {curr_vx:4.1f} subpx/f",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            pad=6,
        )
        ax_prog.set_ylabel("Progress X (px)", color="#cccccc", fontsize=9, fontweight="bold")
        ax_prog.set_xlabel(
            "Hardware Simulation Frames (60 Hz)", color="#cccccc", fontsize=9, fontweight="bold"
        )
        ax_prog.tick_params(colors="#888899", labelsize=8)
        ax_prog.legend(
            loc="upper left",
            facecolor="#1e2233",
            edgecolor="#334466",
            labelcolor="white",
            fontsize=8,
        )

        # -------------------------------------------------------------
        # PANEL 3: TELEMETRY DASHBOARD & JOYPAD BUTTON INDICATORS
        # -------------------------------------------------------------
        ax_hud.set_facecolor("#0a0b10")
        ax_hud.set_xlim(0, 10)
        ax_hud.set_ylim(0, 2)
        ax_hud.axis("off")

        # Telemetry metrics boxes
        telemetry_boxes = [
            ("POSITION X", f"{curr_x:6.1f} px", "#3498db"),
            ("ALTITUDE Y", f"{curr_y:5.1f} px", "#2ecc71"),
            ("VELOCITY vx", f"{curr_vx:4.1f} subpx", "#9b59b6"),
            ("VELOCITY vy", f"{curr_vy:4.1f} subpx", "#e67e22"),
            ("SUBSCREEN", f"{int(curr_x) // 256} / 7", "#1abc9c"),
        ]

        for b_i, (label, val, col) in enumerate(telemetry_boxes):
            bx = 0.2 + b_i * 1.55
            rect = patches.FancyBboxPatch(
                (bx, 0.2),
                1.4,
                1.5,
                boxstyle="round,pad=0.1",
                facecolor="#151824",
                edgecolor=col,
                linewidth=1.5,
            )
            ax_hud.add_patch(rect)
            ax_hud.text(
                bx + 0.7, 1.3, label, color="#8888aa", fontsize=7, fontweight="bold", ha="center"
            )
            ax_hud.text(
                bx + 0.7, 0.55, val, color="white", fontsize=9, fontweight="bold", ha="center"
            )

        # Joypad Button Lights
        btn_x_start = 8.1
        ax_hud.text(
            btn_x_start + 0.9,
            1.45,
            "JOYPAD 60Hz",
            color="#8888aa",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
        )
        joy_buttons = [
            ("B", curr_act.get("B", False)),
            ("Y", curr_act.get("Y", False)),
            ("R", curr_act.get("RIGHT", False)),
        ]
        for btn_i, (btn_name, active) in enumerate(joy_buttons):
            bx = btn_x_start + btn_i * 0.6
            btn_col = "#e74c3c" if active else "#2c3e50"
            text_col = "white" if active else "#556677"
            circ = patches.Circle(
                (bx + 0.25, 0.7),
                0.24,
                facecolor=btn_col,
                edgecolor="white" if active else "#445566",
                linewidth=1.2,
            )
            ax_hud.add_patch(circ)
            ax_hud.text(
                bx + 0.25,
                0.7,
                btn_name,
                color=text_col,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
            )

        # Render canvas to RGB buffer
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()
        rendered_frames.append(rgb)

    plt.close(fig)
    logger.info(
        f"Rendered {len(rendered_frames)} dynamic frames in {time.time() - t_start:.1f}s. Encoding video..."
    )

    # Write MP4 Video
    try:
        imageio.mimwrite(output_mp4, rendered_frames, fps=fps, quality=8)
        logger.info(f"MP4 Video saved to: {output_mp4}")
    except Exception as e:
        logger.info(f"MP4 encoding notice: {e}")

    # Write Animated GIF using Pillow save_all for guaranteed frame delay and animation
    from PIL import Image

    gif_imgs = [
        Image.fromarray(f).resize((780, 480), Image.Resampling.BILINEAR)
        for f in rendered_frames[::2]
    ]
    gif_imgs[0].save(
        output_gif,
        save_all=True,
        append_images=gif_imgs[1:],
        duration=66,  # 15 FPS playback (66 ms per frame)
        loop=0,
    )
    logger.info(f"Animated GIF saved to: {output_gif} ({len(gif_imgs)} frames, looping)")


if __name__ == "__main__":
    render_dynamic_telemetry_video()
