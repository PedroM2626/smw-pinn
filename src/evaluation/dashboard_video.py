"""
dashboard_video.py - render the README's hero dashboard from a recorded controller episode.

`scripts/record_dashboard_run.py` records what the actor believed on every frame; this turns that
recording into the picture the repository opens with: the console's own output filling the left half,
and on the right the four things a reader asks of a policy - what the level models see, what the
actor thinks, which buttons the console received, and what the critic predicts the episode is worth -
with the stage progress and the measured clear rate along the bottom.

Two honesty rules are baked into the layout. The action bars show the actor's own distribution when
it has one (the primitive head), and the independent-Bernoulli joint of its button marginals when it
does not (the distilled actor), labelled as derived rather than shown as if the network emitted it.
And the controller graphic lights the buttons the emulator actually received - after the published
reflex vaults, the jump pulse and any operator A press - because a dashboard that showed only the
actor's proposal would credit the policy with the override layer's progress.

Every number drawn is read from the recording, and the caption numbers the README quotes are written
to the artifact next to the video so a gate can compare the two.

Run:  python -m src.evaluation.dashboard_video
      make dashboard-video
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from src.environment import wram
from src.planning.mpc_planner import ACTION_BUTTONS, ACTION_MATRIX
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
#: the same twelve channels, abbreviated to fit the strip under the game window
OBS_SHORT = (
    "x",
    "y",
    "vx",
    "vy",
    "gnd",
    "ceil",
    "lft",
    "rgt",
    "dx_e",
    "dy_e",
    "vx_e",
    "haz",
)
#: the eight planner primitives in the order ACTION_MATRIX stacks them, in words a reader can use
ACTION_WORDS = (
    "idle",
    "walk right",
    "run right",
    "walk + jump",
    "run + jump",
    "jump in place",
    "walk left",
    "run left",
)
#: the four terrain classes the level buffer stores, plus the one it cannot: a cell that is not there
TILE_COLOURS = ("#0d1420", "#8a5a2b", "#c8322b", "#c8a02b", "#243352")
TILE_LEGEND = ("air", "floor / wall", "hazard", "slope", "no cell in the buffer")
VOID_CLASS = 4


def _rgb(colour: str) -> Tuple[int, int, int]:
    value = colour.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


#: the drawn colours a gate can look for in the rendered still, as raw RGB triples
FLOOR_RGB = _rgb(TILE_COLOURS[1])
VOID_RGB = _rgb(TILE_COLOURS[4])
BG = "#0d1220"
PANEL = "#161d2e"
PANEL_EDGE = "#26314a"
INK = "#e8eef8"
MUTED = "#8794ab"
ACCENT = "#f2a541"
BLUE = "#5aa9e6"
GREEN = "#63d68a"
RED = "#ff6b5e"
IDLE_PIP = "#2f3a52"

#: the layout, in figure fractions of a 1280x720 canvas. The two picture panels are sized so that
#: their box aspect equals the array aspect (256x224 frames, 33x33 square tiles), which is what lets
#: `imshow` fill the box with `aspect="auto"` instead of letterboxing inside it.
RECT_TITLE = [0.010, 0.945, 0.978, 0.040]
RECT_RGB = [0.010, 0.200, 0.463, 0.720]
RECT_PROGRESS = [0.010, 0.150, 0.463, 0.035]
RECT_TIME = [0.010, 0.078, 0.463, 0.064]
RECT_TILE = [0.490, 0.565, 0.186, 0.330]
RECT_LEGEND = [0.690, 0.565, 0.298, 0.330]
RECT_THINK = [0.490, 0.235, 0.330, 0.280]
RECT_PAD = [0.835, 0.235, 0.153, 0.280]
RECT_REWARD = [0.490, 0.090, 0.498, 0.095]
RECT_CLEAR = [0.490, 0.030, 0.498, 0.045]


#: the one stage this repository can name from its own evidence: 10.36.3 built the savestate by
#: walking the map from the node a fresh file starts on, and 10.66 reads that node's translevel as
#: 0x2A. Anything else is drawn by the identity the engine gave it, not by a guessed title.
LEVEL_NAMES = {"yoshi_island_1": "Yoshi's Island 1", "0x2a": "Yoshi's Island 1"}
DEFAULT_TITLE = "Reinforcement learning in Super Mario World"


def level_label(key: str) -> str:
    """The name of a stage when the repository has evidence for one, its identifier otherwise."""
    return LEVEL_NAMES.get(str(key).lower(), str(key))


def context_from_run(
    run_metrics: str = results_file("dashboard_run_metrics.json"),
    level_count: int = 1,
    level_index: int = 1,
    clear_line: str = "",
) -> Dict[str, Any]:
    """The title-bar strings for one recording, read from that recording's own protocol."""
    import json

    with open(run_metrics, encoding="utf-8") as handle:
        run = json.load(handle)
    protocol = run.get("protocol", {})
    key = str(protocol.get("translevel") or protocol.get("savestate", ""))
    key = key.split("/")[-1].replace("smw_", "").replace(".state", "").replace("_level", "")
    return {
        "title": DEFAULT_TITLE,
        "level_label": level_label(key),
        "level_index": level_index,
        "level_count": level_count,
        "clear_line": clear_line,
    }


def load_recording(path: str = DATASET_DASHBOARD) -> Dict[str, np.ndarray]:
    """Read the dashboard recording, keeping the keys this renderer needs."""
    with np.load(path) as data:
        missing = {"states", "button_probs", "executed", "values", "camera", "patches", "rgb"}
        have = set(data.files)
        if missing - have:
            raise KeyError(f"{path} is missing {sorted(missing - have)}")
        return {key: data[key] for key in data.files}


def action_probabilities(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """The distribution to draw, and where it came from.

    A primitive-headed actor emits one directly. A button-headed actor does not: the honest
    substitute is the joint its own independent marginals imply, which is a derivation and is
    labelled as one on the panel rather than presented as the network's opinion.
    """
    if "primitive_probs" in data:
        return np.asarray(data["primitive_probs"], dtype=np.float64), "the actor's own distribution"
    marginals = np.asarray(data["button_probs"], dtype=np.float64)
    matrix = np.asarray(ACTION_MATRIX, dtype=np.float64)
    joint = marginals @ matrix.T + (1.0 - marginals) @ (1.0 - matrix).T
    joint /= np.clip(joint.sum(axis=1, keepdims=True), 1e-12, None)
    return joint, "joint of the actor's independent button marginals"


def patch_view(patches: np.ndarray, states: np.ndarray) -> np.ndarray:
    """The recorded class patch with a fifth class added: the cells the level buffer does not have.

    `get_local_tilemap_patch` stores out-of-range and air in the same byte, because that is what the
    level-conditioned models are given. A picture drawn from the patch alone would therefore call the
    rows below the subscreen's last one empty space. The mask is recomputed here from the shared
    address formula rather than read from the patch - the point being that the model really is shown
    air there, and the panel can say both things at once.
    """
    radius = int(patches.shape[1] // 2)
    view = np.array(patches, dtype=np.int8)
    offsets = range(-radius, radius + 1)
    for i in range(view.shape[0]):
        centre_x = int(states[i, 0]) // wram.TILE_SIZE_PX
        centre_y = int(states[i, 1]) // wram.TILE_SIZE_PX
        for dy in offsets:
            for dx in offsets:
                if wram.tilemap_addr(centre_x + dx, centre_y + dy) is None:
                    view[i, dy + radius, dx + radius] = VOID_CLASS
    return view


def _panel(fig: Any, rect: List[float], title: str = "", card: bool = True) -> Any:
    """One dashboard panel: dark card, thin edge, small-caps title above it."""
    ax = fig.add_axes(rect, facecolor=PANEL if card else BG, zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(card)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(1.0)
    if title:
        ax.text(
            0.0,
            1.0 + (9.0 / (rect[3] * 720.0)),
            title.upper(),
            color=ACCENT,
            fontsize=9.5,
            va="bottom",
            transform=ax.transAxes,
        )
    return ax


def build_figure(data: Dict[str, np.ndarray], context: Dict[str, Any]) -> Dict[str, Any]:
    """Create the panel layout once; `draw_frame` then only updates artists."""
    states = np.asarray(data["states"], dtype=np.float64)
    n = int(states.shape[0])
    values = np.asarray(data["values"], dtype=np.float64)
    actions, action_source = action_probabilities(data)
    progress = states[:, 0] - float(states[0, 0])
    views = patch_view(np.asarray(data["patches"]), states)
    #: the progress bar's denominator: the published goal where one exists, the furthest anything
    #: measured otherwise, because a bar scaled to a goal a stage has never been shown to have is a
    #: claim about the level rather than a picture of the episode
    goal_px = float(context.get("goal_px", GOAL_PX))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    title_ax = _panel(fig, RECT_TITLE, card=False)
    title_text = title_ax.text(0.0, 0.5, "", color=INK, fontsize=13, va="center")
    level_text = title_ax.text(1.0, 0.5, "", color=ACCENT, fontsize=11.5, va="center", ha="right")

    rgb = fig.add_axes(RECT_RGB, facecolor="#000000")
    rgb.set_xticks([])
    rgb.set_yticks([])
    for spine in rgb.spines.values():
        spine.set_color(PANEL_EDGE)
    rgb_img = rgb.imshow(
        np.zeros(data["rgb"][0].shape, np.uint8), aspect="auto", interpolation="nearest"
    )

    tile = _panel(fig, RECT_TILE, "what the level models see")
    tile_img = tile.imshow(
        views[0],
        cmap=ListedColormap(TILE_COLOURS),
        vmin=-0.5,
        vmax=4.5,
        aspect="auto",
        interpolation="nearest",
        extent=(0, 1, 0, 1),
        origin="upper",
    )
    tile.plot([0.5], [0.5], marker="s", color=INK, markersize=7, fillstyle="none")
    tile.text(0.5, 0.44, "Mario", color=INK, fontsize=7.5, ha="center", va="top")
    enemy_mark = tile.plot([], [], marker="o", color=RED, markersize=6)[0]

    legend = _panel(fig, RECT_LEGEND, card=False)
    for i, (colour, label) in enumerate(zip(TILE_COLOURS, TILE_LEGEND)):
        y = 0.96 - 0.085 * i
        legend.add_patch(Rectangle((0.0, y - 0.032), 0.055, 0.064, color=colour))
        legend.text(0.095, y, label, color=INK, fontsize=9, va="center")
    y_mario = 0.96 - 0.085 * 5
    legend.plot([0.027], [y_mario], marker="s", color=INK, markersize=7, fillstyle="none")
    legend.text(0.095, y_mario, "Mario (window centre)", color=INK, fontsize=9, va="center")
    y_enemy = 0.96 - 0.085 * 6
    legend.plot([0.027], [y_enemy], marker="o", color=RED, markersize=6)
    legend.text(0.095, y_enemy, "enemy", color=INK, fontsize=9, va="center")
    legend.text(
        0.0,
        0.38,
        "33x33 tile window from $7E:C800 - the array the\n"
        "level models read. The fifth class marks cells the\n"
        "buffer does not have: the model is shown air there.\n"
        "The actor sees no picture: only the twelve numbers.",
        color=MUTED,
        fontsize=8,
        va="top",
        family="monospace",
        linespacing=1.6,
    )

    think = _panel(fig, RECT_THINK, "what it thinks")
    rows = len(ACTION_WORDS)
    think_bars: List[Any] = []
    think_values: List[Any] = []
    row_span, row_top = 0.80, 0.97
    for i in range(rows):
        y = row_top - (i + 0.5) * row_span / rows
        think.add_patch(
            Rectangle((0.40, y - 0.34 / rows), 0.44, 0.68 / rows, color="#101726", zorder=2)
        )
        think_bars.append(think.barh(y, 0.0, height=0.68 / rows, left=0.40, zorder=3)[0])
        think.text(
            0.01, y, ACTION_WORDS[i], color=INK, fontsize=9, ha="left", va="center", zorder=4
        )
        think_values.append(
            think.text(
                0.865,
                y,
                "",
                color=MUTED,
                fontsize=9,
                ha="left",
                va="center",
                family="monospace",
                zorder=4,
            )
        )
    think.text(0.01, 0.035, action_source, color=MUTED, fontsize=8, va="bottom")

    pad = _panel(fig, RECT_PAD, "buttons")
    pad.add_patch(
        FancyBboxPatch(
            (0.04, 0.40),
            0.92,
            0.34,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=0,
            facecolor="#222a3d",
            zorder=2,
        )
    )
    pips: Dict[str, Any] = {}
    for button, (cx, cy, w, h) in (
        ("UP", (0.20, 0.68, 0.10, 0.13)),
        ("DOWN", (0.20, 0.46, 0.10, 0.13)),
        ("LEFT", (0.11, 0.57, 0.10, 0.13)),
        ("RIGHT", (0.29, 0.57, 0.10, 0.13)),
    ):
        pips[button] = pad.add_patch(
            Rectangle((cx - w / 2, cy - h / 2), w, h, color=IDLE_PIP, zorder=3)
        )
        pad.text(cx, cy - 0.005, button[0], color=MUTED, fontsize=6, ha="center", va="center")
    for button, cx, cy in (("Y", 0.62, 0.685), ("B", 0.76, 0.50), ("A", 0.90, 0.685)):
        pips[button] = pad.add_patch(Circle((cx, cy), 0.075, color=IDLE_PIP, zorder=3))
        pad.text(cx, cy, button, color=INK, fontsize=8, ha="center", va="center", zorder=4)
    pad.text(
        0.50,
        0.26,
        "lit = the console got it",
        color=MUTED,
        fontsize=8,
        ha="center",
        va="center",
    )
    pad.text(
        0.50,
        0.11,
        "A is outside the action\nset: only the operator\npresses it (10.60)",
        color=MUTED,
        fontsize=7.8,
        ha="center",
        va="center",
    )

    reward = _panel(
        fig, RECT_REWARD, "reward prediction   V(s) in px of discounted future progress"
    )
    reward.set_xlim(0, max(1, n - 1))
    reward.plot(np.arange(n), values, color=GREEN, linewidth=1.1)
    reward_cursor = reward.plot([0], [values[0]], marker="o", color=INK, markersize=4)[0]
    reward.set_ylim(float(values.min()) - 5.0, float(values.max()) + 5.0)
    reward.tick_params(colors=MUTED, labelsize=6.5)
    reward.set_xticks([])
    reward_value = reward.text(
        1.0,
        1.0 + 9.0 / (RECT_REWARD[3] * 720.0),
        "",
        color=INK,
        fontsize=9.5,
        family="monospace",
        ha="right",
        va="bottom",
        transform=reward.transAxes,
    )

    bar = _panel(fig, RECT_PROGRESS)
    bar.set_facecolor("#101726")
    bar.set_xlim(0, goal_px)
    bar.set_ylim(0, 1)
    for tick in range(0, int(goal_px) + 1, int(SUBSCREEN_PX)):
        bar.plot([tick, tick], [0, 1], color=PANEL_EDGE, linewidth=0.6, zorder=2)
    fill = bar.barh(0.5, 0.0, height=0.84, color=ACCENT, zorder=3)[0]
    marker = bar.plot([0.0], [0.5], marker="|", color=INK, markersize=18, zorder=5)[0]
    bar.text(
        0.995,
        0.5,
        "one tick per 256 px subscreen",
        color=MUTED,
        fontsize=8,
        family="monospace",
        ha="right",
        va="center",
        zorder=6,
        transform=bar.transAxes,
    )

    time_ax = _panel(fig, RECT_TIME, card=False)
    time_text = time_ax.text(
        0.0, 0.74, "", color=MUTED, fontsize=8.5, family="monospace", va="center"
    )
    obs_text = time_ax.text(
        0.0, 0.26, "", color=MUTED, fontsize=7.5, family="monospace", va="center"
    )

    clear_ax = _panel(fig, RECT_CLEAR, card=False)
    clear_text = clear_ax.text(
        1.0,
        0.5,
        context.get("clear_line", ""),
        color=ACCENT,
        fontsize=10.5,
        ha="right",
        va="center",
    )

    return {
        "fig": fig,
        "n": n,
        "context": context,
        "title_text": title_text,
        "level_text": level_text,
        "rgb_img": rgb_img,
        "tile_img": tile_img,
        "enemy_mark": enemy_mark,
        "think_bars": think_bars,
        "think_values": think_values,
        "pips": pips,
        "fill": fill,
        "marker": marker,
        "time_text": time_text,
        "clear_text": clear_text,
        "reward_cursor": reward_cursor,
        "reward_value": reward_value,
        "obs_text": obs_text,
        "values": values,
        "residue": np.asarray(data["residue_subpixels"], dtype=np.float64),
        "camera": np.asarray(data["camera"], dtype=np.float64),
        "actions": actions,
        "progress": progress,
        "states": states,
        "executed": np.asarray(data["executed"], dtype=np.float64),
        "operator": np.asarray(data.get("operator", np.zeros(n)), dtype=np.float64),
        "mode": np.asarray(data.get("mode", np.zeros(n)), dtype=np.float64),
        "live": np.asarray(data.get("live", np.ones(n)), dtype=np.float64),
        "frames": data["rgb"],
        "patches": data["patches"],
        "views": views,
        "goal_px": goal_px,
    }


def draw_frame(art: Dict[str, Any], i: int) -> None:
    """Point every artist at frame `i`."""
    ctx = art["context"]
    state = art["states"][i]
    progress = float(art["progress"][i])
    mode = int(art["mode"][i])
    executed = art["executed"][i]
    actions = art["actions"][i]

    art["rgb_img"].set_data(art["frames"][i])
    art["tile_img"].set_data(art["views"][i])

    art["title_text"].set_text(ctx.get("title", ""))
    art["level_text"].set_text(
        f"level {ctx.get('level_index', '?')} of {ctx.get('level_count', '?')}"
        f"    {ctx.get('level_label', '')}    engine mode 0x{mode:02X}"
    )

    top = int(np.argmax(actions))
    for k, rect in enumerate(art["think_bars"]):
        rect.set_width(float(actions[k]) * 0.44)
        rect.set_color(ACCENT if k == top else BLUE)
        art["think_values"][k].set_text(f"{100.0 * actions[k]:.0f}%")

    for k, button in enumerate(ACTION_BUTTONS):
        on = bool(executed[k] > 0.5)
        art["pips"][button].set_facecolor(ACCENT if on else IDLE_PIP)
    art["pips"]["A"].set_facecolor(GREEN if art["operator"][i] > 0.5 else IDLE_PIP)

    art["fill"].set_width(max(0.0, progress))
    art["marker"].set_xdata([progress])
    seconds = i / 60.0
    art["time_text"].set_text(
        f"time {int(seconds // 60)}:{seconds % 60:04.1f}   frame {i}/{art['n'] - 1}"
        f"   {progress:.1f} / {art['goal_px']:.0f} px"
        f"   subscreen {int(max(0.0, progress) // SUBSCREEN_PX)}"
        f"   {'live' if art['live'][i] else 'held by the engine'}"
    )
    art["reward_cursor"].set_data([i], [float(art["values"][i])])
    art["reward_value"].set_text(f"V(s) = {float(art['values'][i]):.2f} px")
    art["obs_text"].set_text(
        "state: " + " ".join(f"{name} {state[k]:.0f}" for k, name in enumerate(OBS_SHORT))
    )

    half = art["patches"].shape[1] // 2
    span = (half + 0.5) * 16.0
    dx_enemy, dy_enemy = float(state[8]), float(state[9])
    if float(state[11]) > 0.5 and abs(dx_enemy) <= span and abs(dy_enemy) <= span:
        # The panel is drawn with the array's first row at the top, so a positive `dy_enemy` - an
        # enemy below Mario - goes to a smaller axis height, not a larger one.
        art["enemy_mark"].set_data([0.5 + 0.5 * dx_enemy / span], [0.5 - 0.5 * dy_enemy / span])
    else:
        art["enemy_mark"].set_data([], [])


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
    plt.close(fig)
    return {
        "mp4": _rel(output_mp4),
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
        "gif": _rel(output_gif),
        "bytes": os.path.getsize(output_gif),
        "gif_fps": gif_fps,
        "gif_width_px": width,
        "gif_palette_colors": max_colors,
    }


def _rel(path: str) -> str:
    resolved = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    return os.path.relpath(resolved, REPO_ROOT).replace("\\", "/")


def still_preview(art: Dict[str, Any], frame: int, out_png: str) -> str:
    """Write one dashboard frame as a PNG, to look at the layout without a video pass."""
    draw_frame(art, frame)
    art["fig"].canvas.draw()
    buf = np.asarray(art["fig"].canvas.buffer_rgba())[:, :, :3]
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    imageio.imwrite(out_png, np.ascontiguousarray(buf))
    return out_png


def run_dashboard_render(
    dataset_path: str = DATASET_DASHBOARD,
    output_mp4: str = figure_file("dashboard_run.mp4"),
    output_gif: str = figure_file("dashboard_run.gif"),
    stride: int = 2,
    fps: float = 30.0,
    gif_fps: float = 12.0,
    gif_width: int = 900,
    gif_colors: int = 96,
    context: Dict[str, Any] | None = None,
    output_still: str = figure_file("dashboard_run_still.png"),
    output_metrics: str | None = results_file("dashboard_video_metrics.json"),
) -> Dict[str, Any]:
    """Render the video and the GIF, then write the artifact the README caption is gated against.

    `output_metrics=None` keeps the payload in memory only, which is what the multi-stage study does:
    it embeds each render's numbers in its own artifact rather than publishing three files whose
    writer is an f-string no gate can find.
    """
    data = load_recording(dataset_path)
    ctx = dict(context or {})
    ctx.setdefault("title", "Reinforcement learning in Super Mario World")
    ctx.setdefault("level_label", "Yoshi's Island 1")
    ctx.setdefault("level_index", 1)
    ctx.setdefault("level_count", 1)
    ctx.setdefault("clear_line", "")
    n = int(data["states"].shape[0])
    indices = list(range(0, n, max(1, stride)))
    art = build_figure(data, ctx)
    still_frame = indices[len(indices) // 2]
    still_preview(art, still_frame, output_still)
    video = render_video(art, indices, output_mp4, fps)
    gif = render_gif(
        os.path.join(REPO_ROOT, video["mp4"]), output_gif, gif_fps, gif_width, gif_colors
    )

    states = np.asarray(data["states"], dtype=np.float64)
    progress = states[:, 0] - float(states[0, 0])
    probs = np.asarray(data["button_probs"], dtype=np.float64)
    residue = np.asarray(data["residue_subpixels"], dtype=np.float64)
    operator = np.asarray(data.get("operator", np.zeros(n)), dtype=np.float64)
    values = np.asarray(data["values"], dtype=np.float64)
    actions, action_source = action_probabilities(data)
    payload: Dict[str, Any] = {
        "study": (
            "The README hero dashboard, rendered from the recorded episode: the console's own RGB "
            "picture beside the tilemap the level-conditioned models read, with the action "
            "distribution the actor holds, the buttons the console received on a controller graphic, "
            "the terminal-value critic's prediction, and the stage progress bar."
        ),
        "protocol": {
            "dataset": _rel(dataset_path),
            "frames_in_episode": n,
            "frames_rendered": len(indices),
            "stride": stride,
            "context": ctx,
            "goal_px": float(art["goal_px"]),
            "subscreen_px": SUBSCREEN_PX,
            "action_panel_source": action_source,
            "state_names": list(STATE_NAMES),
            "state_names_on_screen": list(OBS_SHORT),
        },
        "video": video,
        "gif": gif,
        "still": {
            "png": _rel(output_still),
            "bytes": os.path.getsize(output_still),
            "frame": int(still_frame),
        },
        "caption": {
            "episode_frames": n,
            "tile_void_cell_share": float(np.mean(art["views"] == VOID_CLASS)),
            "progress_px": float(progress[-1]),
            "camera_travel_px": float(np.ptp(np.asarray(data["camera"], dtype=np.float64))),
            "mean_top_action_probability": float(np.mean(actions.max(axis=1))),
            "mean_max_button_probability": float(np.mean(probs.max(axis=1))),
            "residue_exact_share": float(np.mean(np.abs(residue) < 1e-6)),
            "operator_A_frames": int(operator.sum()),
            "value_first_px": float(values[0]),
            "value_last_px": float(values[-1]),
            "subscreens_reached": int(progress[-1] // SUBSCREEN_PX) + 1,
        },
    }
    if output_metrics is not None:
        write_metrics(output_metrics, payload, command="make dashboard-video")
    logger.info(f"dashboard video: {video}")
    logger.info(f"dashboard gif: {gif}")
    return payload


if __name__ == "__main__":
    run_dashboard_render(context=context_from_run())
