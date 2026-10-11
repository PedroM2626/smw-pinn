"""
dashboard_video.py - render the README's hero dashboard from a recorded controller episode.

`scripts/record_dashboard_run.py` records what the actor believed on every frame; this turns that
recording into the two-window picture the repository opens with: the emulator's own RGB output on
one side and the tilemap the level-conditioned models read on the other, with the actor's button
probabilities, the terminal-value critic's prediction, the section-4.1 identity residue, the engine
mode byte and the stage progress bar around them.

Every number drawn is read from the recording, and the caption numbers the README quotes are
written to the artifact next to the video so a gate can compare the two.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from src.planning.mpc_planner import ACTION_BUTTONS
from src.utils.logging import get_logger
from src.utils.paths import DATASET_DASHBOARD, REPO_ROOT, figure_file, results_file
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

#: the published clearance benchmark calls the level done at this distance
GOAL_PX = 1900.0
SUBSCREEN_PX = 256.0
STATE_NAMES = (
    "x",
    "y",
    "vx",
    "vy",
    "c_ground",
    "c_ceiling",
    "c_left",
    "c_right",
    "dx_enemy",
    "dy_enemy",
    "vx_enemy",
    "hazard",
)
TILE_COLOURS = ("#101418", "#8a5a2b", "#c8322b", "#c8a02b")
INK = "#e8ecf4"
PANEL = "#151a24"
BG = "#0b0d13"
ACCENT = "#4fc3f7"
WARN = "#ffb74d"


def load_recording(path: str = DATASET_DASHBOARD) -> Dict[str, np.ndarray]:
    """Read the dashboard recording, keeping the keys this renderer needs."""
    with np.load(path) as data:
        missing = {"states", "button_probs", "executed", "values", "camera", "patches", "rgb"}
        have = set(data.files)
        if missing - have:
            raise KeyError(f"{path} is missing {sorted(missing - have)}")
        return {key: data[key] for key in data.files}


def _style(ax: Any, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, color=INK, fontsize=8.5, loc="left", pad=3)


def build_figure(data: Dict[str, np.ndarray], level_label: str) -> Dict[str, Any]:
    """Create the panel layout once; `draw_frame` then only updates artists."""
    n = int(data["states"].shape[0])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    header = fig.add_axes([0.012, 0.955, 0.976, 0.035], facecolor=BG)
    header.axis("off")
    bar = fig.add_axes([0.012, 0.888, 0.976, 0.052])
    rgb_ax = fig.add_axes([0.012, 0.335, 0.315, 0.53])
    tile_ax = fig.add_axes([0.345, 0.335, 0.185, 0.53])
    obs_ax = fig.add_axes([0.548, 0.335, 0.185, 0.53])
    prob_ax = fig.add_axes([0.752, 0.335, 0.236, 0.53])
    btn_ax = fig.add_axes([0.012, 0.045, 0.315, 0.265])
    val_ax = fig.add_axes([0.345, 0.045, 0.40, 0.265])
    res_ax = fig.add_axes([0.752, 0.045, 0.236, 0.265])

    _style(rgb_ax, "emulator RGB output (the console's own picture)")
    _style(tile_ax, "tilemap $7E:C800 (level models' view)")
    _style(obs_ax, "actor observation (12D)")
    _style(prob_ax, "actor's button probabilities")
    _style(btn_ax, "buttons the emulator received")
    bar.set_facecolor(PANEL)
    bar.set_xticks([])
    bar.set_yticks([])
    for side in ("top", "bottom", "left", "right"):
        bar.spines[side].set_visible(False)

    rgb_img = rgb_ax.imshow(np.zeros(data["rgb"][0].shape, np.uint8), interpolation="nearest")
    tile_img = tile_ax.imshow(
        np.zeros(data["patches"][0].shape, np.int8),
        cmap=ListedColormap(TILE_COLOURS),
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
    )
    tile_ax.plot(
        data["patches"].shape[2] // 2,
        data["patches"].shape[1] // 2,
        marker="o",
        color=INK,
        markersize=3,
        fillstyle="none",
    )

    obs_text = obs_ax.text(
        0.02,
        0.94,
        "",
        color=INK,
        fontsize=8,
        family="monospace",
        verticalalignment="top",
        linespacing=1.55,
    )
    obs_ax.set_xlim(0, 1)
    obs_ax.set_ylim(0, 1)
    prob_ax.set_xlim(0, 1)
    prob_ax.set_ylim(0, 1)
    prob_bars = [
        prob_ax.barh(
            1.0 - (i + 0.5) / len(ACTION_BUTTONS),
            0.0,
            height=0.7 / len(ACTION_BUTTONS),
            color=ACCENT,
        )
        for i in range(len(ACTION_BUTTONS))
    ]
    prob_labels = [
        prob_ax.text(
            0.01,
            1.0 - (i + 0.5) / len(ACTION_BUTTONS),
            ACTION_BUTTONS[i],
            color=INK,
            fontsize=8,
            family="monospace",
            verticalalignment="center",
        )
        for i in range(len(ACTION_BUTTONS))
    ]
    for label in prob_labels:
        label.set_x(0.99)
        label.set_ha("right")

    btn_ax.set_xlim(0, 1)
    btn_ax.set_ylim(0, 1)
    btn_pips = [
        btn_ax.add_patch(
            Rectangle(
                (i / len(ACTION_BUTTONS) + 0.012, 0.30),
                1.0 / len(ACTION_BUTTONS) - 0.024,
                0.55,
                facecolor="#20262f",
                edgecolor="#39414f",
                linewidth=0.8,
            )
        )
        for i in range(len(ACTION_BUTTONS))
    ]
    btn_texts = [
        btn_ax.text(
            (i + 0.5) / len(ACTION_BUTTONS),
            0.575,
            ACTION_BUTTONS[i],
            color="#4a5364",
            fontsize=11,
            family="monospace",
            ha="center",
            va="center",
        )
        for i in range(len(ACTION_BUTTONS))
    ]
    btn_state = [
        btn_ax.text(
            (i + 0.5) / len(ACTION_BUTTONS),
            0.16,
            "",
            color=INK,
            fontsize=7.5,
            family="monospace",
            ha="center",
            va="center",
        )
        for i in range(len(ACTION_BUTTONS))
    ]
    operator_text = btn_ax.text(
        0.985,
        0.94,
        "",
        color=WARN,
        fontsize=8.5,
        family="monospace",
        ha="right",
        va="center",
    )

    x = np.asarray(data["states"][:, 0], dtype=np.float64)
    camera = np.asarray(data["camera"], dtype=np.float64)
    progress = x - float(x[0])
    bar.set_xlim(0, GOAL_PX)
    bar.set_ylim(0, 1)
    bar.axvspan(0, GOAL_PX, color="#20262f")
    for tick in range(0, int(GOAL_PX) + 1, int(SUBSCREEN_PX)):
        bar.axvline(tick, color="#39414f", linewidth=0.6)
    fill = bar.barh(0.5, 0.0, height=0.9, color=ACCENT, alpha=0.85)
    marker = bar.plot([0.0], [0.5], marker="|", color=INK, markersize=18)[0]
    bar_text = bar.text(
        0.004,
        0.5,
        "",
        color=INK,
        fontsize=9,
        family="monospace",
        va="center",
        bbox=dict(facecolor=BG, edgecolor="none", alpha=0.72, pad=2.0),
    )

    values = np.asarray(data["values"], dtype=np.float64)
    residue = np.asarray(data["residue_subpixels"], dtype=np.float64)
    val_ax.set_xlim(0, max(1, n - 1))
    val_ax.plot(np.arange(n), values, color=ACCENT, linewidth=1.0)
    val_cursor = val_ax.plot([0], [values[0]], marker="o", color=INK, markersize=4)[0]
    val_ax.set_ylim(float(values.min()) - 5.0, float(values.max()) + 5.0)
    val_ax.set_facecolor(PANEL)
    val_ax.tick_params(colors=INK, labelsize=7)
    val_ax.set_title(
        "terminal-value critic V(s), px of discounted future progress",
        color=INK,
        fontsize=8.5,
        loc="left",
        pad=3,
    )
    for side in ("top", "right"):
        val_ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        val_ax.spines[side].set_color("#39414f")

    res_ax.set_xlim(0, max(1, n - 1))
    res_ax.plot(np.arange(n), np.zeros(n), color="#39414f", linewidth=0.6)
    res_ax.step(np.arange(n), residue, color="#7ee787", linewidth=1.0)
    res_cursor = res_ax.plot([0], [residue[0]], marker="o", color=INK, markersize=4)[0]
    res_ax.set_ylim(float(residue.min()) - 2.0, float(residue.max()) + 2.0)
    res_ax.set_facecolor(PANEL)
    res_ax.tick_params(colors=INK, labelsize=7)
    res_ax.set_title(
        "section-4.1 identity residue 16*(x'-x)-v_x, sub-pixels",
        color=INK,
        fontsize=8.5,
        loc="left",
        pad=3,
    )
    for side in ("top", "right"):
        res_ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        res_ax.spines[side].set_color("#39414f")

    header_text = header.text(0.0, 0.5, "", color=INK, fontsize=10, family="monospace", va="center")

    return {
        "fig": fig,
        "header": header_text,
        "level_label": level_label,
        "n": n,
        "progress": progress,
        "fill": fill,
        "marker": marker,
        "bar_text": bar_text,
        "rgb_img": rgb_img,
        "tile_img": tile_img,
        "obs_text": obs_text,
        "prob_bars": prob_bars,
        "btn_pips": btn_pips,
        "btn_texts": btn_texts,
        "btn_state": btn_state,
        "operator_text": operator_text,
        "val_cursor": val_cursor,
        "res_cursor": res_cursor,
        "values": values,
        "residue": residue,
        "camera": camera,
        "frames": data["rgb"],
        "patches": data["patches"],
        "states": data["states"],
        "button_probs": data["button_probs"],
        "executed": data["executed"],
        "operator": data.get("operator", np.zeros(n)),
        "mode": data.get("mode", np.zeros(n)),
        "live": data.get("live", np.ones(n)),
    }


def draw_frame(art: Dict[str, Any], i: int) -> None:
    """Point every artist at frame `i`."""
    state = np.asarray(art["states"][i], dtype=np.float64)
    probs = np.asarray(art["button_probs"][i], dtype=np.float64)
    executed = np.asarray(art["executed"][i], dtype=np.float64)
    operator = float(art["operator"][i])
    progress = float(art["progress"][i])
    camera = float(art["camera"][i])
    mode = int(art["mode"][i])

    art["rgb_img"].set_data(art["frames"][i])
    art["tile_img"].set_data(art["patches"][i])
    art["obs_text"].set_text(
        "\n".join(
            f"{name:>9s} {state[k]:10.3f}" if k > 3 else f"{name:>9s} {state[k]:10.0f}"
            for k, name in enumerate(STATE_NAMES)
        )
    )
    for k, rect in enumerate(art["prob_bars"]):
        rect[0].set_width(float(probs[k]))
        rect[0].set_color(ACCENT if probs[k] >= 0.5 else "#3d5a75")
    for k, txt in enumerate(art["btn_texts"]):
        on = bool(executed[k] > 0.5)
        txt.set_color(INK if on else "#4a5364")
        art["btn_pips"][k].set_facecolor("#3d5a75" if on else "#20262f")
        art["btn_state"][k].set_text("pressed" if on else "")
    art["operator_text"].set_text("OPERATOR HOLDING A - not in the action set" if operator else "")
    art["fill"][0].set_width(max(0.0, progress))
    art["marker"].set_xdata([progress])
    art["bar_text"].set_text(
        f" {progress:8.2f} px of {GOAL_PX:.0f}   subscreen "
        f"{int(max(0.0, progress) // SUBSCREEN_PX)}   "
        f"Mario at screen x={state[0] - camera:6.2f} "
    )
    art["val_cursor"].set_data([i], [float(art["values"][i])])
    art["res_cursor"].set_data([i], [float(art["residue"][i])])
    art["header"].set_text(
        f"{art['level_label']}   frame {i:5d}/{art['n'] - 1}   mode 0x{mode:02X}   cam {camera:7.2f} px   "
        f"V(s) {float(art['values'][i]):7.2f} px   res {float(art['residue'][i]):+6.2f} sp   "
        f"{'LIVE' if art['live'][i] else 'PLAYER HELD STILL'}"
    )


def render_video(
    art: Dict[str, Any], indices: List[int], output_mp4: str, fps: float
) -> Dict[str, Any]:
    """Draw the selected frames and encode them straight to disk (no frame stack in memory)."""
    os.makedirs(os.path.dirname(output_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(
        output_mp4,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    fig = art["fig"]
    try:
        for step, i in enumerate(indices):
            draw_frame(art, i)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(np.ascontiguousarray(buf[:, :, :3]))
            if (step + 1) % 100 == 0:
                logger.info(f"dashboard: rendered {step + 1}/{len(indices)} frames")
    finally:
        writer.close()
    plt.close(art["fig"])
    return {
        "mp4": output_mp4,
        "bytes": os.path.getsize(output_mp4),
        "frames_written": len(indices),
        "fps": fps,
        "duration_s": round(len(indices) / fps, 2),
    }


def render_gif(
    output_mp4: str, output_gif: str, gif_fps: float, width: int, max_colors: int
) -> Dict[str, Any]:
    """Derive the README's inline GIF from the MP4 through the bundled ffmpeg."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filt = (
        f"fps={gif_fps},scale={width}:-1:flags=lanczos,"
        f"split[s0][s1];[s0]palettegen=max_colors={max_colors}[p];"
        f"[s1][p]paletteuse=dither=none"
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", output_mp4, "-vf", filt, output_gif],
        check=True,
        capture_output=True,
    )
    return {
        "gif": output_gif,
        "bytes": os.path.getsize(output_gif),
        "gif_fps": gif_fps,
        "gif_width_px": width,
        "gif_palette_colors": max_colors,
    }


def run_dashboard_render(
    dataset_path: str = DATASET_DASHBOARD,
    output_mp4: str = figure_file("dashboard_run.mp4"),
    output_gif: str = figure_file("dashboard_run.gif"),
    stride: int = 2,
    fps: float = 30.0,
    gif_fps: float = 12.0,
    gif_width: int = 720,
    gif_colors: int = 128,
    level_label: str = "Yoshi's Island 1 - world-map entry, control handoff verified",
    output_metrics: str = results_file("dashboard_video_metrics.json"),
) -> Dict[str, Any]:
    """Render the video and the GIF, then write the artifact the README caption is gated against."""
    data = load_recording(dataset_path)
    n = int(data["states"].shape[0])
    indices = list(range(0, n, max(1, stride)))
    art = build_figure(data, level_label)
    video = render_video(art, indices, output_mp4, fps)
    gif = render_gif(output_mp4, output_gif, gif_fps, gif_width, gif_colors)

    x = np.asarray(data["states"][:, 0], dtype=np.float64)
    progress = x - float(x[0])
    probs = np.asarray(data["button_probs"], dtype=np.float64)
    residue = np.asarray(data["residue_subpixels"], dtype=np.float64)
    operator = np.asarray(data.get("operator", np.zeros(n)), dtype=np.float64)
    payload: Dict[str, Any] = {
        "study": (
            "The README hero dashboard, rendered from the recorded episode: the console's own RGB "
            "picture beside the tilemap the level-conditioned models read, with the actor's button "
            "probabilities, the terminal-value critic's prediction, the section-4.1 identity residue, "
            "the engine mode byte and the stage progress bar."
        ),
        "protocol": {
            "dataset": os.path.relpath(dataset_path, REPO_ROOT),
            "frames_in_episode": n,
            "frames_rendered": len(indices),
            "stride": stride,
            "level_label": level_label,
            "goal_px": GOAL_PX,
            "subscreen_px": SUBSCREEN_PX,
        },
        "video": video,
        "gif": gif,
        "caption": {
            "episode_frames": n,
            "progress_px": float(progress[-1]),
            "camera_travel_px": float(np.ptp(np.asarray(data["camera"], dtype=np.float64))),
            "mean_max_button_probability": float(np.mean(probs.max(axis=1))),
            "residue_exact_share": float(np.mean(np.abs(residue) < 1e-6)),
            "operator_A_frames": int(operator.sum()),
            "value_first_px": float(np.asarray(data["values"], dtype=np.float64)[0]),
            "value_last_px": float(np.asarray(data["values"], dtype=np.float64)[-1]),
            "subscreens_reached": int(progress[-1] // SUBSCREEN_PX) + 1,
        },
    }
    write_metrics(output_metrics, payload, command="make dashboard-video")
    logger.info(f"dashboard video: {video}")
    logger.info(f"dashboard gif: {gif}")
    return payload


if __name__ == "__main__":
    run_dashboard_render()
