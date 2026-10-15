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
#: five display classes, matching burn-mario-rl's grid: the measured behaviour table collapsed
#: into the classes a reader can tell apart at GIF resolution. 0 empty (dark), 1 ground and
#: wall (blue), 2 coin (yellow), 3 other block (brown), 4 enemy (red).
TILE_COLOURS = (
    "#222638",
    "#6078be",
    "#ffcd3c",
    "#aa7846",
    "#f04646",
)
TILE_LEGEND = (
    "ground and wall",
    "other block",
    "coin",
    "enemy",
)
EMPTY_CLASS = 0
#: the reference grid's view window: 20 columns x 14 rows, Mario at column 5 / row 8
GRID_W, GRID_H = 20, 14
COLS_BEHIND, ROWS_ABOVE = 5, 8
#: the sprite classes, drawn as the cell the sprite occupies - burn-mario-rl's hostile and
#: friendly planes, one cell per live sprite, in the panel the terrain classes share
SPRITE_CELL_CLASSES = {
    "hostile": 4,
    "stompable": 4,
    "interact": 3,
    "platform": 3,
    "collectible": 3,
    "neutral": 3,
    "unprobed": 3,
}


def _rgb(colour: str) -> Tuple[int, int, int]:
    value = colour.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


#: the drawn colours a gate can look for in the rendered still, as raw RGB triples
GROUND_RGB = _rgb(TILE_COLOURS[1])
COIN_RGB = _rgb(TILE_COLOURS[2])
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

#: the layout, in figure fractions of a 1280x720 canvas. The tile panel is sized for a 20x14
#: grid, and the legend sits inside a narrow strip to its right.
RECT_TITLE = [0.010, 0.945, 0.978, 0.040]
RECT_RGB = [0.010, 0.200, 0.463, 0.720]
RECT_PROGRESS = [0.010, 0.150, 0.463, 0.035]
RECT_TIME = [0.010, 0.078, 0.463, 0.064]
RECT_TILE = [0.490, 0.565, 0.280, 0.330]
RECT_LEGEND = [0.790, 0.565, 0.198, 0.330]
RECT_THINK = [0.490, 0.235, 0.330, 0.280]
RECT_PAD = [0.835, 0.235, 0.153, 0.280]
RECT_REWARD = [0.490, 0.090, 0.498, 0.095]
RECT_CLEAR = [0.490, 0.030, 0.498, 0.045]


#: the translevel byte at $7E:13BF and the name each stage is drawn under. The rows carry their
#: evidence in LEVEL_NAME_SOURCES, because they do not share one kind.
#:
#: * `0x2A` and `0x28` are anchored by this repository's own measurements: 10.36.3 walked the map
#:   from the node a fresh file starts on and read `0x2A` there, and 10.66/10.68 measured `0x28`
#:   as the session with no timer, one subscreen of terrain and a door the engine walks the player
#:   out through - Yoshi's House is the only stage in the world with all three.
#: * the rest come from the community's translevel table (SMWCentral thread #23605, the Lunar
#:   Magic vanilla level list, the snesrev/smw decompilation), and that table is *not* consistent
#:   with the anchor: it names `0x29` Yoshi's Island 1 and `0x2A` Yoshi's Island 2, while the map
#:   position and the game's own progression make `0x2A` the first stage. The rows below keep the
#:   table's ordering relative to the anchor, which is an inference, not a measurement.
#:
#: The measurement that would settle it - the level's own title card - was run and is not
#: available on this console route. Entering `0x2A`, whose name is already known, draws no card
#: under any of four attempts: from the restored map state and from a genuine power-on boot walked
#: through the menus, each with a pulsed A and with a single deliberate press. Capturing every
#: frame of the load sequence shows why the null is blind rather than informative - modes `0x10`
#: and `0x12` are the black fade and `0x13` already holds the drawn level with its HUD and its
#: timer at 400 and counting, so the instrument never sees the case whose answer is known and
#: cannot clear the cases whose answers are not. The inferred rows stay labelled as inferred.
LEVEL_NAMES = {
    "0x25": "#1 Iggy's Castle",
    "0x26": "Yoshi's Island 4",
    "0x27": "Yoshi's Island 3",
    "0x28": "Yoshi's House",
    "0x29": "Yoshi's Island 2",
    "0x2a": "Yoshi's Island 1",
    "yoshi_island_1": "Yoshi's Island 1",
}
#: how each key above was identified; anything not "measured here" is a table reading
LEVEL_NAME_SOURCES = {
    "0x25": "translevel table",
    "0x26": "translevel table",
    "0x27": "translevel table, shifted to the anchor",
    "0x28": "measured here",
    "0x29": "translevel table, shifted to the anchor",
    "0x2a": "measured here",
    "yoshi_island_1": "measured here",
}
DEFAULT_TITLE = "Reinforcement learning in Super Mario World"


def level_label(key: str) -> str:
    """The name of a stage when the repository has evidence for one, its identifier otherwise."""
    return LEVEL_NAMES.get(str(key).lower(), str(key))


def level_name_source(key: str) -> str:
    """Whether a stage's drawn name was measured by this repository or read off a table."""
    return LEVEL_NAME_SOURCES.get(str(key).lower(), "unnamed")


def load_recording(path: str = DATASET_DASHBOARD) -> Dict[str, np.ndarray]:
    """Read the dashboard recording, keeping the keys this renderer needs."""
    with np.load(path) as data:
        missing = {
            "states",
            "button_probs",
            "executed",
            "values",
            "camera",
            "patches",
            "rgb",
            "tile_ids",
            "sprites",
        }
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


def window_view(tile_ids: np.ndarray, sprites: np.ndarray, states: np.ndarray) -> np.ndarray:
    """The measured tile buffer and live sprites in the reference grid's exact geometry.

    The recording stores a 33x33 raw-ID window centred on Mario's WRAM position. The displayed
    20x14 window uses the hitbox anchors from burn-mario-rl: Mario's cell is
    ``((x + 8) >> 4, (y + 16) >> 4)`` and a sprite's cell is
    ``((x + 8) >> 4, (y + 8) >> 4)``. This preserves tile-boundary transitions that a rounded
    relative displacement cannot recover.
    """
    from src.environment import sprite_behavior, tile_behavior

    ids = np.asarray(tile_ids, dtype=np.int64)
    player = np.asarray(states, dtype=np.float64)
    classified = np.full(ids.shape, EMPTY_CLASS, dtype=np.int8)
    inside = ids >= 0
    collapse = {
        "decor": 0,
        "coin": 2,
        "platform": 1,
        "solid": 1,
        "slope": 1,
        "harmful": 3,
        "interact": 3,
    }
    for i in np.ndindex(*ids.shape):
        if inside[i]:
            name = tile_behavior.behavior_of(int(ids[i]))
            classified[i] = collapse.get(name, 0)

    rows = np.asarray(sprites, dtype=np.float64)
    active = rows[:, :, 3] >= sprite_behavior.SPRITE_ACTIVE
    centre = ids.shape[1] // 2
    view = np.full((ids.shape[0], GRID_H, GRID_W), EMPTY_CLASS, dtype=np.int8)
    for frame in range(view.shape[0]):
        mario_x, mario_y = float(player[frame, 0]), float(player[frame, 1])
        raw_tx, raw_ty = int(mario_x) // 16, int(mario_y) // 16
        mario_tx, mario_ty = int(mario_x + 8.0) // 16, int(mario_y + 16.0) // 16
        c0 = centre + mario_tx - raw_tx - COLS_BEHIND
        r0 = centre + mario_ty - raw_ty - ROWS_ABOVE
        view[frame] = classified[frame, r0 : r0 + GRID_H, c0 : c0 + GRID_W]

        for slot in range(rows.shape[1]):
            if not active[frame, slot]:
                continue
            sprite_x = int(np.rint(mario_x + rows[frame, slot, 0]))
            sprite_y = int(np.rint(mario_y + rows[frame, slot, 1]))
            col = (sprite_x + 8) // 16 - mario_tx + COLS_BEHIND
            row = (sprite_y + 8) // 16 - mario_ty + ROWS_ABOVE
            if 0 <= row < GRID_H and 0 <= col < GRID_W:
                kind = sprite_behavior.behavior_of(int(rows[frame, slot, 2]))
                sprite_class = SPRITE_CELL_CLASSES.get(kind, 3)
                view[frame, row, col] = max(int(view[frame, row, col]), sprite_class)
    return view


def sprite_classes(sprites: np.ndarray) -> np.ndarray:
    """The measured class name of every recorded sprite slot, per frame.

    Only slots the engine was actually running (status >= 8) carry a class: the recordings
    faithfully keep spawn-list rows too, but a dormant slot is not in the world yet, and the
    panel that drew one would mark a shell that has not spawned.
    """
    from src.environment import sprite_behavior

    ids = np.asarray(sprites, dtype=np.int64)
    out = np.empty(ids.shape[:2], dtype=object)
    for frame in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            out[frame, slot] = (
                sprite_behavior.behavior_of(int(ids[frame, slot, 2]))
                if ids[frame, slot, 3] >= sprite_behavior.SPRITE_ACTIVE
                else ""
            )
    return out


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
    views = window_view(np.asarray(data["tile_ids"]), np.asarray(data["sprites"]), states)
    sprites = np.asarray(data["sprites"], dtype=np.float64)
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

    tile = _panel(fig, RECT_TILE, "the tiles around Mario, as measured")
    tile_img = tile.imshow(
        views[0],
        cmap=ListedColormap(TILE_COLOURS),
        vmin=-0.5,
        vmax=len(TILE_COLOURS) - 0.5,
        aspect="auto",
        interpolation="nearest",
        extent=(0, 1, 0, 1),
        origin="upper",
    )
    mario_x = (COLS_BEHIND + 0.5) / GRID_W
    mario_y = 1.0 - (ROWS_ABOVE + 0.5) / GRID_H
    tile.plot(
        [mario_x],
        [mario_y],
        marker="s",
        color=INK,
        markersize=9,
        fillstyle="none",
        markeredgewidth=2,
    )
    tile.text(mario_x, mario_y - 0.06, "Mario", color=INK, fontsize=7.5, ha="center", va="top")
    for c in range(1, GRID_W):
        tile.axvline(c / GRID_W, color="#1a1e2e", linewidth=0.4, zorder=0.5)
    for r in range(1, GRID_H):
        tile.axhline(r / GRID_H, color="#1a1e2e", linewidth=0.4, zorder=0.5)

    legend = _panel(fig, RECT_LEGEND, card=False)
    legend_colours = [TILE_COLOURS[1], TILE_COLOURS[3], TILE_COLOURS[2], TILE_COLOURS[4]]
    row = 0.92
    step = 0.12
    for i, (colour, label) in enumerate(zip(legend_colours, TILE_LEGEND)):
        y = row - step * i
        legend.add_patch(Rectangle((0.0, y - 0.030), 0.10, 0.060, color=colour))
        legend.text(0.16, y, label, color=INK, fontsize=8.5, va="center")
    y = row - step * len(TILE_LEGEND)
    legend.plot(
        [0.05], [y], marker="s", color=INK, markersize=9, fillstyle="none", markeredgewidth=2
    )
    legend.text(0.16, y, "Mario", color=INK, fontsize=8.5, va="center")

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
        "sprites": sprites,
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
    # the sprites are already part of the view: the window carries their cells, drawn the way
    # the reference grid draws its hostile plane, and nothing needs moving here


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


def render_indices(states_x: np.ndarray, stride: int) -> List[int]:
    """The frames the reel draws: the moving episode at `stride`, a stall at 12x that.

    The clearance draw clears Yoshi's Island 1 with two long pauses at the pipes - 3357 frames
    in the featured episode, more than half the flight - and a reel that draws them at the
    moving stride is a reel that watches Mario stand still for most of its length. A dashboard
    is a picture of the recording, not a clock: the moving stretch keeps every `stride`-th frame
    and a stretch where the player has not advanced keeps one frame in twelve of those, so the
    stall is still on screen - the frame counter jumps, visibly, and the viewer sees the wait
    happen - without being most of the video. Every kept frame is a recorded frame; nothing is
    drawn that the emulator did not produce.
    """
    stride = max(1, stride)
    stall_stride = 12 * stride
    indices: List[int] = []
    last = -1
    n = int(states_x.shape[0])
    i = 0
    while i < n:
        if last < 0 or abs(float(states_x[i]) - float(states_x[last])) > 2.0:
            indices.append(i)
            last = i
            i += stride
        else:
            # no advance since the last kept frame: skip ahead in stall-sized steps, keeping
            # one frame per step so the pause reads as a pause and not as a dropped scene
            i = min(last + stall_stride, n)
            if i < n:
                indices.append(i)
                last = i
                i += stride
    return indices


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
    metrics_command: str = "make dashboard-video",
) -> Dict[str, Any]:
    """Render the video and the GIF, then write the artifact the README caption is gated against.

    `output_metrics=None` keeps the payload in memory only, which is what the multi-stage study does:
    it embeds each render's numbers in its own artifact rather than publishing three files whose
    writer is an f-string no gate can find. `metrics_command` names the target that actually wrote
    the artifact, which is not always this module's own.
    """
    data = load_recording(dataset_path)
    ctx = dict(context or {})
    ctx.setdefault("title", DEFAULT_TITLE)
    ctx.setdefault("level_label", "Yoshi's Island 1")
    ctx.setdefault("level_index", 1)
    ctx.setdefault("level_count", 1)
    ctx.setdefault("clear_line", "")
    n = int(data["states"].shape[0])
    states_x = np.asarray(data["states"], dtype=np.float64)[:, 0]
    indices = render_indices(states_x, stride)
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
            "stall_stride_multiplier": 12,
            "index_rule": (
                "every stride-th frame while the player advances, one frame in twelve of those "
                "while a stall holds - the frame counter on the panel shows the jump"
            ),
            "context": ctx,
            "goal_px": float(art["goal_px"]),
            "subscreen_px": SUBSCREEN_PX,
            "action_panel_source": action_source,
            "state_names": list(STATE_NAMES),
            "state_names_on_screen": list(OBS_SHORT),
            "tile_grid": {
                "columns": GRID_W,
                "rows": GRID_H,
                "mario_column": COLS_BEHIND,
                "mario_row": ROWS_ABOVE,
                "mario_cell": "((x + 8) >> 4, (y + 16) >> 4)",
                "sprite_cell": "((sprite_x + 8) >> 4, (sprite_y + 8) >> 4)",
                "active_sprite_rule": "status >= 8",
            },
            # the drawn stage name is a claim, so the artifact records what kind of evidence it
            # rests on: this repository measured some of these bytes and read the rest off a table
            "level_name_source": level_name_source(str(ctx.get("translevel", ""))),
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
            "tile_empty_cell_share": float(np.mean(art["views"] == EMPTY_CLASS)),
            "frames_with_enemy_cells": int(np.any(art["views"] == 4, axis=(1, 2)).sum()),
            "rendered_frames_with_enemy_cells": int(
                np.any(art["views"][indices] == 4, axis=(1, 2)).sum()
            ),
            "max_enemy_cells_in_frame": int(np.sum(art["views"] == 4, axis=(1, 2)).max()),
            "active_sprite_rows": int(np.sum(art["sprites"][:, :, 3] >= 8)),
            "max_active_sprite_slots_in_frame": int(
                np.sum(art["sprites"][:, :, 3] >= 8, axis=1).max()
            ),
            "progress_px": float(progress[-1]),
            "camera_travel_px": float(np.ptp(np.asarray(data["camera"], dtype=np.float64))),
            "mean_top_action_probability": float(np.mean(actions.max(axis=1))),
            "mean_max_button_probability": float(np.mean(probs.max(axis=1))),
            "residue_exact_share": float(
                np.mean(np.abs(residue[np.asarray(data["live"], dtype=bool)]) < 1e-6)
            ),
            "operator_A_frames": int(operator.sum()),
            "value_first_px": float(values[0]),
            "value_last_px": float(values[-1]),
            "subscreens_reached": int(progress[-1] // SUBSCREEN_PX) + 1,
        },
    }
    if output_metrics is not None:
        write_metrics(output_metrics, payload, command=metrics_command)
    logger.info(f"dashboard video: {video}")
    logger.info(f"dashboard gif: {gif}")
    return payload


def hero_context() -> Dict[str, Any]:
    """The README hero's drawing context, read from the study that decides what the hero says.

    Both `make dashboard-video` and `make level-dashboard-study` render this one picture. Rebuilt on
    its own the standalone one wrote a 1-of-1 title bar and an empty clearance line over the figure
    the study had published, so the target named by the hero's own artifact did not produce it.
    """
    from src.evaluation.level_dashboard_study import gather as gather_study
    from src.evaluation.level_dashboard_study import hero_context as study_hero_context

    return study_hero_context(gather_study())


if __name__ == "__main__":
    run_dashboard_render(context=hero_context())
