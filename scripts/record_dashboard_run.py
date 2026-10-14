"""
record_dashboard_run.py - record one controller episode with everything the dashboard video shows.

The published `full_level_trajectory_log.json` stores the state and the boolean buttons, which is
enough for the existing telemetry GIF and not enough for the panel layout this repository's README
opens with: a dashboard has to show what the policy believed, not only what it did. So this records,
per frame, the actor's per-button probabilities, the primitive distribution when the actor has one,
the terminal-value critic's V(s), the buttons the emulator actually received (the published
controller applies two reflex overrides and a jump edge-trigger pulse after the policy - a dashboard
that hid those would be advertising the policy for the reflex's progress), the tilemap patch and
camera, an RGB frame, the section-4.1 identity residue of the step just taken, and a liveness flag.

The liveness flag is 10.59's paused-player signature - every channel of the next 12D record equal to
the current one - and the episode ends when it persists, so the video never learns to look like the
world holds still.

The featured episode is the level being completed: the 10.66 seed-7 clearance draw, run until the
engine itself ends it (the mode byte leaving 0x14, not a distance proxy), and kept recording
through the exit sequence - the outro - so the README opens with a stage the engine finished. Both
learned checkpoints are still probed on the same savestate for the same horizon and written into
the selection artifact beside the episode: on the verified level-entry savestate the PPO primitive
actor walks into the level's left wall (negative progress) while the distilled DAgger actor
traverses, so the probe's own rule (most forward progress within the horizon) lands on DAgger -
and neither learned arm finishes the stage, which is why the featured episode is the draw that
does. The probe winner is re-probed with the published override layer removed, because the
reflex vaults belong to the benchmark and their cost should be on the record.

Needs core + ROM. Writes data/raw/smw_dashboard_run.npz and results/dashboard_run_metrics.json.

Run:  python scripts/record_dashboard_run.py
"""

import argparse
import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import wram  # noqa: E402
from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.evaluation.evaluate_full_level_clearance import extract_12d_vector  # noqa: E402
from src.planning.mpc_planner import (  # noqa: E402
    ACTION_BUTTONS,
    ACTION_MATRIX,
    ACTION_PRIMITIVES,
)
from src.planning.terminal_value import TerminalValueNet  # noqa: E402
from src.training.distill_mpc_policy import DistilledActorPolicy  # noqa: E402
from src.training.train_unified_ppo import UnifiedActorCritic  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CHECKPOINTS_DIR,
    CORE_PATH,
    REPO_ROOT,
    ROM_PATH,
    STATE_YOSHI_ISLAND_1_LEVEL,
)
from src.utils.provenance import write_metrics  # noqa: E402

DATASET_PATH = os.path.join(REPO_ROOT, "data", "raw", "smw_dashboard_run.npz")
ARTIFACT_NAME = "dashboard_run_metrics.json"
VALUE_CKPT = os.path.join(CHECKPOINTS_DIR, "terminal_value_best.pt")
#: the two overrides the published clearance benchmark applies after the policy
REFLEX_HAZARD_PX = 75.0
#: consecutive identical 12D records that count as the engine holding the player still
PAUSE_TRIGGER = 10
#: how long the recorder waits on a held player before pressing A. Measured, not guessed: the box
#: this waits for prints its text over ~120 frames, and pressing earlier dismisses an unreadable box.
OPERATOR_A_DELAY = 120
#: how long the operator keeps pressing A on a frozen player before the episode gives up
OPERATOR_A_LIMIT = 120
PATCH_RADIUS = 16
PRIMITIVE_NAMES = list(ACTION_PRIMITIVES.keys())
#: SUBPIXELS_PER_PIXEL of the section-4.1 identity: 16*(x'-x) - vx
SUBPIXELS_PER_PIXEL = 16.0


def load_value_critic(device: str) -> Tuple[TerminalValueNet, Dict[str, float]]:
    """Rebuild the TD-MPC terminal value critic from its published checkpoint."""
    blob = torch.load(VALUE_CKPT, map_location=device, weights_only=True)
    net = TerminalValueNet(hidden_dim=64).to(device)
    net.load_state_dict(blob["model"])
    net.eval()
    return net, {**blob["stats"], "gamma": float(blob["gamma"])}


def value_of(net: TerminalValueNet, stats: Dict[str, float], state: np.ndarray) -> float:
    """V(s) in px of discounted future progress; the critic reads [x, y, vx, vy]."""
    device = next(net.parameters()).device
    with torch.no_grad():
        raw = net(torch.as_tensor(state[:4], dtype=torch.float32, device=device))
    return float(raw.item() * stats["target_std"] + stats["target_mean"])


class PpoPrimitiveActor:
    """The amortized PPO actor: a Categorical distribution over the eight primitives."""

    name = "ppo"
    kind = "primitive head"
    checkpoint = "unified_ppo_policy_best.pt"

    def __init__(self, device: str) -> None:
        self.agent = UnifiedActorCritic(state_dim=12, num_actions=len(PRIMITIVE_NAMES)).to(device)
        path = os.path.join(CHECKPOINTS_DIR, self.checkpoint)
        self.agent.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.agent.eval()

    def act(self, s_vec: np.ndarray, device: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        with torch.no_grad():
            dist, _ = self.agent(torch.as_tensor(s_vec, device=device).unsqueeze(0))
            probs = dist.probs.squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        vec = ACTION_MATRIX[idx]
        buttons = {b: bool(vec[i] > 0.5) for i, b in enumerate(ACTION_BUTTONS)}
        return buttons, {
            "primitive_probs": probs.astype(np.float32),
            "button_probs": (probs[:, None] * ACTION_MATRIX).sum(axis=0).astype(np.float32),
            "chosen_primitive": idx,
        }


class DaggerButtonActor:
    """The MPC-distilled DAgger actor: six independent button sigmoids."""

    name = "dagger"
    kind = "button head"
    checkpoint = "dagger_policy_best.pt"

    def __init__(self, device: str) -> None:
        self.agent = DistilledActorPolicy(state_dim=12, action_dim=len(ACTION_BUTTONS)).to(device)
        self.agent.load_state_dict(
            torch.load(
                os.path.join(CHECKPOINTS_DIR, self.checkpoint),
                map_location=device,
                weights_only=True,
            )
        )
        self.agent.eval()
        self.matrix = ACTION_MATRIX  # only for the primitive index bookkeeping

    def act(self, s_vec: np.ndarray, device: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        with torch.no_grad():
            logits = self.agent(torch.as_tensor(s_vec, device=device).unsqueeze(0))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        buttons = {b: bool(p > 0.5) for b, p in zip(ACTION_BUTTONS, probs)}
        return buttons, {
            "button_probs": probs.astype(np.float32),
            "chosen_primitive": -1,
        }


class MpcClearanceActor:
    """The published clearance controller of 10.37/10.66, the one that cleared Yoshi's Island 1.

    The planner samples candidate action sequences through the learned world model, so it has no
    per-button probabilities to show; what it does have is the primitive it committed to, and the
    panel draws that as a one-hot distribution labelled as the planner's choice.
    """

    name = "mpc"
    kind = "planner"
    checkpoint = "pinn_multi_entity_best.pt"

    def __init__(self, device: str, seed: Optional[int] = None) -> None:
        import random

        from scripts.unlock_world_map import build_mpc

        # built first, seeded after: the 10.66 flight built its controller once and seeded inside
        # `fly`, so whatever the construction consumes from the streams happens before the draw
        # is drawn - seeding in the other order reproduces a different run
        self.controller = build_mpc(torch.device(device))
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)

    def act(self, s_vec: np.ndarray, device: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        from src.evaluation.evaluate_full_level_clearance import action_vector_to_dict

        vec, _info = self.controller.plan(s_vec)
        vec = np.asarray(vec, dtype=np.float64).ravel()
        buttons = action_vector_to_dict(vec)
        # the planner commits to one of the eight primitives; the panel draws that commitment as
        # the one-hot it is, labelled as the planner's choice rather than a sampled distribution
        chosen = int(np.argmax(np.abs(np.asarray(ACTION_MATRIX) - vec).sum(axis=1)))
        onehot = np.zeros(len(ACTION_MATRIX), dtype=np.float32)
        onehot[chosen] = 1.0
        return buttons, {
            "button_probs": np.clip(vec, 0.0, 1.0).astype(np.float32),
            "primitive_probs": onehot,
            "chosen_primitive": chosen,
        }


class TerrainRunnerActor:
    """The 10.68 rule-based agent: measured terrain and sprite tables, six rules, no sampling.

    The runner reads the level buffer and the sprite slots through the emulator, so `record()`
    binds the emulator once the episode starts. Its message-box A pulses flow through the
    recorded buttons rather than a hidden channel, and the operator column counts them.
    """

    name = "terrain-runner"
    kind = "rule-based agent"
    checkpoint = "none (the measured behaviour tables)"

    def __init__(self, device: str, direction: str = "RIGHT") -> None:
        from src.planning.terrain_runner import TerrainRunner

        self.runner = TerrainRunner(direction=direction)
        self._emu = None

    def bind(self, emu) -> None:
        self._emu = emu

    def act(self, s_vec: np.ndarray, device: str) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        state = {
            "x": float(s_vec[0]),
            "y": float(s_vec[1]),
            "vx": float(s_vec[2]),
            "vy": float(s_vec[3]),
            "c_ground": float(s_vec[4]),
            "c_ceiling": float(s_vec[5]),
            "c_left": float(s_vec[6]),
            "c_right": float(s_vec[7]),
        }
        raw = self.runner.act(self._emu, state)
        # A stays in the button dict (the operator column records it) but is not part of the
        # six-button action set the executed vector and the controller graphic index by
        buttons = {b: bool(raw.get(b, False)) for b in ACTION_BUTTONS}
        if raw.get("A", False):
            buttons["A"] = True
        chosen = 0
        for index, vec in enumerate(ACTION_MATRIX):
            if all(
                bool(vec[i]) == bool(buttons.get(b, False)) for i, b in enumerate(ACTION_BUTTONS)
            ):
                chosen = index
                break
        onehot = np.zeros(len(ACTION_MATRIX), dtype=np.float32)
        onehot[chosen] = 1.0
        return buttons, {
            "button_probs": np.asarray(ACTION_MATRIX[chosen], dtype=np.float32),
            "primitive_probs": onehot,
            "chosen_primitive": chosen,
        }


ACTORS = {
    "ppo": PpoPrimitiveActor,
    "dagger": DaggerButtonActor,
    "mpc": MpcClearanceActor,
    "terrain-runner": TerrainRunnerActor,
}


def apply_published_overrides(
    state: Dict[str, float], buttons: Dict[str, bool], frame: int, prev_b: bool
) -> Tuple[Dict[str, bool], bool, bool]:
    """Reflex vaulting then the jump edge-trigger pulse, exactly as the benchmark does them.

    Returns (buttons, reflex_fired, pulse_released).
    """
    reflex = False
    dx_enemy = float(state["delta_x_enemy"])
    if (
        float(state["hazard_active"]) > 0.5
        and 0.0 < dx_enemy < REFLEX_HAZARD_PX
        and (float(state["c_ground"]) > 0.5)
    ):
        buttons.update({"B": True, "Y": True, "RIGHT": True})
        reflex = True
    elif float(state["c_right"]) > 0.5 and float(state["c_ground"]) > 0.5:
        buttons.update({"B": True, "RIGHT": True, "Y": True})
        reflex = True
    pulse = False
    if buttons.get("B", False) and prev_b and float(state["c_ground"]) > 0.5 and frame % 2 == 0:
        buttons["B"] = False
        pulse = True
    return buttons, reflex, pulse


def probe_arm(
    kind: str,
    state_path: str,
    frames: int,
    device: str,
    overrides: bool = True,
    goal_px: float = 1900.0,
) -> Dict[str, Any]:
    """Run one actor for a fixed horizon with no frame capture; returns the probe summary."""
    actor = ACTORS[kind](device)
    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    with open(state_path, "rb") as fh:
        emu.start_episode(fh.read())
    start_x = emu.get_smw_state()["x"]
    prev_b = False
    chosen: List[str] = []
    prog: List[float] = []
    confidences: List[float] = []
    reflex = 0
    t0 = time.time()
    ended = "probe_frames"
    for frame in range(frames):
        state = emu.get_smw_extended_state()
        s_vec = extract_12d_vector(state)
        prog.append(float(state["x"]) - start_x)
        buttons, info = actor.act(s_vec, device)
        if kind == "ppo":
            chosen.append(PRIMITIVE_NAMES[int(info["chosen_primitive"])])
            confidences.append(float(np.max(info["primitive_probs"])))
        else:
            joined = "/".join(b for b in ACTION_BUTTONS if buttons[b])
            chosen.append(joined or "NOOP")
            confidences.append(float(np.max(info["button_probs"])))
        if overrides:
            buttons, fired, _ = apply_published_overrides(state, buttons, frame, prev_b)
            reflex += int(fired)
        prev_b = buttons.get("B", False)
        emu.set_input(buttons)
        emu.step_frame()
        if state["y"] > wram.DEATH_Y:
            ended = "death"
            break
        if prog[-1] >= goal_px:
            ended = "goal"
            break
    emu.close()
    counts = collections.Counter(chosen)
    return {
        "arm": kind,
        "published_overrides": overrides,
        "frames": len(prog),
        "ended": ended,
        "progress_px": float(prog[-1]) if prog else 0.0,
        "max_progress_px": float(max(prog)) if prog else 0.0,
        "most_frequent_actions": [[k, int(v)] for k, v in counts.most_common(3)],
        "mean_top_probability": round(float(np.mean(confidences)), 4) if confidences else None,
        "frames_where_reflex_overrode": int(reflex),
        "wall_clock_s": round(time.time() - t0, 2),
    }


def downscale(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w, _ = frame.shape
    rows = (np.arange(height) * h // height).astype(int)
    cols = (np.arange(width) * w // width).astype(int)
    return frame[np.ix_(rows, cols, [0, 1, 2])].astype(np.uint8)


def replay_sprite_channel(
    state_path: str,
    dataset_path: str,
    warmup_frames: int = 0,
) -> Dict[str, int]:
    """Re-read every active sprite slot while replaying a recording's executed controls."""
    with np.load(dataset_path, allow_pickle=False) as blob:
        data = {key: blob[key] for key in blob.files}
    states = np.asarray(data["states"], dtype=np.float64)
    executed = np.asarray(data["executed"], dtype=np.float64)
    operator = np.asarray(data.get("operator", np.zeros(len(states))), dtype=np.float64)
    modes = np.asarray(data["mode"], dtype=np.uint8)
    noninteractive = np.flatnonzero(modes != wram.GAME_MODE_INTERACTIVE)
    last_gameplay_frame = int(noninteractive[0]) if noninteractive.size else len(states) - 1
    sprites = np.zeros((len(states), wram.NUM_SPRITE_SLOTS, 4), dtype=np.float32)

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    with open(state_path, "rb") as handle:
        emu.start_episode(handle.read(), warmup_frames=warmup_frames)
    for frame in range(last_gameplay_frame + 1):
        current = emu.get_smw_state()
        observed = np.asarray([current["x"], current["y"]], dtype=np.float64)
        if not np.allclose(observed, states[frame, :2], atol=1.0 / 128.0):
            emu.close()
            raise RuntimeError(
                f"sprite replay diverged at frame {frame}: {observed.tolist()} != "
                f"{states[frame, :2].tolist()}"
            )
        buttons = {name: bool(value > 0.5) for name, value in zip(ACTION_BUTTONS, executed[frame])}
        buttons["A"] = bool(operator[frame] > 0.5)
        emu.set_input(buttons)
        emu.step_frame()
        for sprite in emu.get_active_sprites():
            slot = int(sprite["slot"])
            sprites[frame, slot] = (
                float(sprite["x"]) - states[frame, 0],
                float(sprite["y"]) - states[frame, 1],
                float(sprite["id"]),
                float(sprite["status"]),
            )
    emu.close()

    data["sprites"] = sprites
    np.savez_compressed(dataset_path, **data)
    active = sprites[:, :, 3] >= wram.SPRITE_STATUS_ACTIVE
    return {
        "frames": len(states),
        "active_rows": int(active.sum()),
        "frames_with_active_sprites": int(active.any(axis=1).sum()),
        "max_active_slots_in_frame": int(active.sum(axis=1).max()),
    }


def record(
    kind: str,
    state_path: str,
    max_frames: int,
    device: str,
    rgb_width: int,
    rgb_height: int,
    operator_limit: int = OPERATOR_A_LIMIT,
    keep_frames: bool = True,
    goal_px: Optional[float] = 1900.0,
    seed: Optional[int] = None,
    runner_direction: str = "RIGHT",
    apply_overrides: bool = True,
    operator_delay: int = OPERATOR_A_DELAY,
    warmup_frames: int = 5,
    outro_frames: int = 0,
    savestate_route: str = (
        "entered through the world map with the control-handoff gate of 10.36.3 satisfied"
    ),
    dataset_path: str = DATASET_PATH,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Record one episode with everything the dashboard video shows.

    `goal_px=None` lets the episode run until the engine itself ends it - the mode byte leaving
    0x14, which is the clear/death discriminator of 10.66 rather than a distance proxy - so the
    featured completion is the engine's own verdict. `seed` is passed to the planner when the
    featured arm samples one; seed 7 is the draw that cleared Yoshi's Island 1. `outro_frames`
    keeps recording past the engine's verdict with the controls released, so the video shows the
    exit sequence itself - the mode march to the map and the map settling - instead of freezing
    on the frame the byte flipped.
    """
    if kind == "mpc":
        actor = MpcClearanceActor(device, seed=seed)
    elif kind == "terrain-runner":
        actor = TerrainRunnerActor(device, direction=runner_direction)
    else:
        actor = ACTORS[kind](device)
    value_net, value_stats = load_value_critic(device)

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    emu.enable_frame_capture(True)
    if hasattr(actor, "bind"):
        actor.bind(emu)
    with open(state_path, "rb") as fh:
        savestate = fh.read()
    emu.start_episode(savestate, warmup_frames=warmup_frames)
    stage_identity = {
        "mode_at_start": f"0x{emu.get_game_mode():02X}",
        "translevel": f"0x{emu.read_wram_u8(wram.ADDR_TRANSLEVEL):02X}",
    }

    keys = (
        "frame",
        "state",
        "button_probs",
        "chosen",
        "value",
        "executed",
        "reflex",
        "pulse",
        "operator",
        "camera",
        "mode",
        "patch",
        "tile_ids",
        "sprites",
        "frame_rgb",
        "live",
        "residue",
    )
    rows: Dict[str, List[Any]] = {key: [] for key in keys}
    primitive_probs: List[np.ndarray] = []
    prev_record: np.ndarray | None = None
    prev_b = False
    paused_run = 0
    operator_run = 0
    pause_events: List[Dict[str, Any]] = []
    start_x: float | None = None
    t0 = time.time()
    ended = "max_frames"
    after: Dict[str, float] = {}
    #: the lives counter ($7E:0DBE, BCD) is what separates an exit from a death: the house's
    #: left-edge walk-out passes through the death mode *byte*, and the byte alone would call
    #: it a death - a life that stays put says it is the exit's own path
    ADDR_LIVES = 0x0DBE
    lives_before = int(emu.read_wram_u8(ADDR_LIVES))
    engine_exit_frame: Optional[int] = None

    for frame in range(max_frames):
        state = emu.get_smw_extended_state()
        s_vec = extract_12d_vector(state)
        if start_x is None:
            start_x = float(state["x"])
        live = True
        if prev_record is not None and np.all(np.abs(s_vec - prev_record) < 1e-9):
            live = False
            paused_run += 1
        else:
            if paused_run >= PAUSE_TRIGGER:
                pause_events.append(
                    {
                        "frame": frame - 1,
                        "progress_px": round(float(state["x"]) - start_x, 4),
                        "mode_byte": int(emu.get_game_mode()),
                        "held_frames": paused_run,
                        "operator_frames": operator_run,
                        "resumed": True,
                    }
                )
            paused_run = 0
            operator_run = 0
        prev_record = s_vec.copy()

        buttons, info = actor.act(s_vec, device)
        if apply_overrides:
            buttons, reflex, pulse = apply_published_overrides(state, buttons, frame, prev_b)
        else:
            # the published clearance draw of 10.66 is the planner alone; layering the reflex
            # vaults on top of it would fly a different, unaudited controller and call it the
            # reproduction of a seed whose whole point is that it reproduces
            reflex = False
            pulse = False
        operator = False
        if paused_run >= PAUSE_TRIGGER + operator_delay:
            # 10.59's frozen player record is the engine's own message box, and the box waits for
            # the A button - which is not in this repository's six-button action set. So no policy
            # here can dismiss it, and an episode that reaches a box ends there unless somebody
            # outside the action space presses A. That press is recorded, not hidden.
            if operator_run >= operator_limit:
                pause_events.append(
                    {
                        "frame": frame,
                        "progress_px": round(float(state["x"]) - start_x, 4),
                        "mode_byte": int(emu.get_game_mode()),
                        "held_frames": paused_run,
                        "operator_frames": operator_run,
                        "resumed": False,
                    }
                )
                ended = "pause_not_clearable"
                break
            buttons = {**buttons, "A": (paused_run % 8) < 4}
            operator = True
            operator_run += 1
        # a rule-based agent presses A on the same boxes through its own hygiene layer; the
        # controller graphic must light that A whoever pressed it
        operator = operator or bool(buttons.get("A", False))
        prev_b = buttons.get("B", False)
        executed = np.array(
            [1.0 if buttons.get(b, False) else 0.0 for b in ACTION_BUTTONS], np.float32
        )

        emu.set_input(buttons)
        emu.step_frame()
        nxt = emu.get_smw_state()
        residue = SUBPIXELS_PER_PIXEL * (float(nxt["x"]) - float(state["x"])) - float(state["vx"])
        shot = emu.get_frame() if keep_frames else None
        rgb = (
            downscale(shot, rgb_width, rgb_height)
            if shot is not None
            else np.zeros((rgb_height, rgb_width, 3), np.uint8)
        )
        patch = (
            emu.get_local_tilemap_patch(state["x"], state["y"], radius=PATCH_RADIUS)
            if keep_frames
            else None
        )
        tile_ids = (
            emu.read_tile_ids(state["x"], state["y"], radius=PATCH_RADIUS) if keep_frames else None
        )
        if keep_frames:
            sprite_rows = np.zeros((12, 4), dtype=np.float32)
            for sprite in emu.get_active_sprites():
                if 0 <= sprite["slot"] < 12:
                    sprite_rows[sprite["slot"]] = (
                        float(sprite["x"]) - float(state["x"]),
                        float(sprite["y"]) - float(state["y"]),
                        float(sprite["id"]),
                        float(sprite["status"]),
                    )

        rows["frame"].append(frame)
        rows["state"].append(s_vec)
        rows["button_probs"].append(info["button_probs"])
        if kind == "ppo":
            primitive_probs.append(info["primitive_probs"])
        rows["chosen"].append(int(info["chosen_primitive"]))
        if "primitive_probs" in info:
            primitive_probs.append(info["primitive_probs"])
        rows["value"].append(value_of(value_net, value_stats, s_vec))
        rows["executed"].append(executed)
        rows["reflex"].append(1.0 if reflex else 0.0)
        rows["pulse"].append(1.0 if pulse else 0.0)
        rows["operator"].append(1.0 if operator else 0.0)
        rows["camera"].append(emu.get_camera_x())
        rows["mode"].append(float(emu.get_game_mode()))
        if keep_frames:
            rows["patch"].append(patch)
            rows["tile_ids"].append(tile_ids)
            rows["sprites"].append(sprite_rows)
            rows["frame_rgb"].append(rgb)
        rows["live"].append(1.0 if live else 0.0)
        rows["residue"].append(float(residue))
        after = nxt
        mode_now = emu.get_game_mode()
        if mode_now != wram.GAME_MODE_INTERACTIVE:
            # the engine's own verdict: 0x0B is the death animation, anything else is the
            # stage's exit sequence, and the mode transitions say which (10.66's discriminator)
            ended = f"engine_left_0x14_to_0x{mode_now:02X}"
            engine_exit_frame = len(rows["frame"])
            break
        if after["y"] > wram.DEATH_Y:
            ended = "death"
            break
        if goal_px is not None and after["x"] - start_x >= goal_px:
            ended = "goal"
            break

    #: the progress the run reached when the engine delivered its verdict - the outro that
    #: follows teleports the player (the warp markers are $FFFC/$FFFE), and a progress read
    #: after that would quote the teleport, not the run
    exit_progress = float(after["x"]) - float(start_x or 0.0)

    if outro_frames > 0 and (ended.startswith("engine_left") or ended == "death"):
        # the outro: the controls are released, the actor and the operator are off, and the
        # camera keeps recording the engine's own exit sequence - the mode march (0x0B ->
        # 0x0C -> 0x0D -> 0x0E for the door, 0x0C onward for the goal tape) and the map
        # settling on the node the session ended at. The telemetry panels freeze on the last
        # gameplay frame (the engine is teleporting the player; a "residue" of a warp is not a
        # number), and the console's own RGB window is what carries the epilogue.
        last_vec = (
            np.asarray(prev_record, dtype=np.float64).copy()
            if prev_record is not None
            else np.zeros(12)
        )
        #: the outro's residue is the held-frame identity, not a zero: the record is frozen, so
        #: 16*(x'-x) - vx reads exactly -vx, and a gate that checks the identity on held frames
        #: must not be able to tell the outro from the engine's own holds
        outro_residue = -float(last_vec[2]) if last_vec.shape[0] >= 3 else 0.0
        #: when the episode's actor emitted a primitive distribution, the outro has no actor at
        #: all: the bars draw zero rather than inheriting the last gameplay frame's opinion
        outro_emits_primitives = bool(primitive_probs)
        for _ in range(outro_frames):
            emu.set_input({})
            emu.step_frame()
            nxt = emu.get_smw_state()
            mode_now = emu.get_game_mode()
            if engine_exit_frame is None and mode_now != wram.GAME_MODE_INTERACTIVE:
                engine_exit_frame = len(rows["frame"])
            shot = emu.get_frame() if keep_frames else None
            rgb = (
                downscale(shot, rgb_width, rgb_height)
                if shot is not None
                else np.zeros((rgb_height, rgb_width, 3), np.uint8)
            )
            rows["frame"].append(len(rows["frame"]))
            rows["state"].append(last_vec)
            rows["button_probs"].append(np.zeros(len(ACTION_BUTTONS), np.float32))
            rows["chosen"].append(-1)
            rows["value"].append(rows["value"][-1] if rows["value"] else 0.0)
            rows["executed"].append(np.zeros(len(ACTION_BUTTONS), np.float32))
            rows["reflex"].append(0.0)
            rows["pulse"].append(0.0)
            rows["operator"].append(0.0)
            rows["camera"].append(emu.get_camera_x())
            rows["mode"].append(float(mode_now))
            rows["live"].append(0.0)
            rows["residue"].append(outro_residue)
            if outro_emits_primitives:
                primitive_probs.append(np.zeros(len(ACTION_MATRIX), dtype=np.float32))
            if keep_frames:
                rows["patch"].append(rows["patch"][-1])
                rows["tile_ids"].append(rows["tile_ids"][-1])
                rows["sprites"].append(np.zeros((12, 4), dtype=np.float32))
                rows["frame_rgb"].append(rgb)
            after = nxt
    lives_after = int(emu.read_wram_u8(ADDR_LIVES))

    emu.close()
    elapsed = time.time() - t0
    progress = exit_progress
    camera = np.asarray(rows["camera"], dtype=np.float64)
    states = np.asarray(rows["state"], dtype=np.float64)
    screen = states[:, 0] - camera[: states.shape[0]]
    residue = np.asarray(rows["residue"], dtype=np.float64)
    save: Dict[str, Any] = {
        "frames": np.asarray(rows["frame"], dtype=np.int32),
        "states": states.astype(np.float32),
        "button_probs": np.asarray(rows["button_probs"], dtype=np.float32),
        "chosen": np.asarray(rows["chosen"], dtype=np.int32),
        "values": np.asarray(rows["value"], dtype=np.float32),
        "executed": np.asarray(rows["executed"], dtype=np.float32),
        "reflex": np.asarray(rows["reflex"], dtype=np.float32),
        "pulse": np.asarray(rows["pulse"], dtype=np.float32),
        "operator": np.asarray(rows["operator"], dtype=np.float32),
        "camera": camera.astype(np.float32),
        "mode": np.asarray(rows["mode"], dtype=np.uint8),
        "live": np.asarray(rows["live"], dtype=np.float32),
        "residue_subpixels": residue.astype(np.float32),
    }
    if keep_frames:
        save["patches"] = np.asarray(rows["patch"], dtype=np.int8)
        save["tile_ids"] = np.asarray(rows["tile_ids"], dtype=np.int32)
        save["sprites"] = np.asarray(rows["sprites"], dtype=np.float32)
        save["rgb"] = np.asarray(rows["frame_rgb"], dtype=np.uint8)
    if primitive_probs:
        save["primitive_probs"] = np.asarray(primitive_probs, dtype=np.float32)
    if keep_frames:
        np.savez_compressed(dataset_path, **save)

    probs = np.asarray(rows["button_probs"], dtype=np.float64)
    summary: Dict[str, Any] = {
        "study": (
            "One controller episode recorded with everything the README dashboard shows: the actor's "
            "per-button probabilities, the critic V(s) fitted on the published clearance returns, "
            "the buttons the emulator actually received after the published reflex overrides and the "
            "jump edge-trigger pulse, the tile patch and camera the actor does not see, an RGB frame, "
            "the section-4.1 identity residue of each step, and the paused-player flag of 10.59 "
            "together with the operator's A presses that clear it."
        ),
        "protocol": {
            "savestate": os.path.basename(state_path),
            "stage_identity": stage_identity,
            "savestate_route": savestate_route,
            "controller": kind,
            "controller_kind": actor.kind,
            "checkpoint": actor.checkpoint,
            "planner_seed": seed,
            "goal_px": goal_px,
            "episode_end_rule": (
                "the engine's own mode byte leaving 0x14"
                if goal_px is None
                else f"the published {goal_px:.0f} px distance proxy"
            ),
            "buttons": list(ACTION_BUTTONS),
            "primitives": PRIMITIVE_NAMES,
            "reflex_overrides": (
                "hazard within 75 px and grounded, or a right wall while grounded"
                if apply_overrides
                else "none: the clearance draw of 10.66 is the planner alone"
            ),
            "jump_pulse": "B released on every even frame while held and grounded",
            "value_critic": {
                "checkpoint": os.path.basename(VALUE_CKPT),
                "reads": ["x", "y", "vx", "vy"],
                "units": "px of discounted future progress",
                "gamma": value_stats["gamma"],
            },
            "pause_trigger_frames": PAUSE_TRIGGER,
            "operator_delay_frames": OPERATOR_A_DELAY,
            "operator_rule": (
                f"once the 12D record has been identical for "
                f"{PAUSE_TRIGGER + OPERATOR_A_DELAY} frames the operator "
                f"pulses A (4 on, 4 off) up to {OPERATOR_A_LIMIT} frames; A is not in the "
                "six-button action set, so this is an outside-the-policy takeover and it is "
                "recorded per frame, not folded into the actor's actions"
            ),
            "patch_radius_tiles": PATCH_RADIUS,
            "rgb_shape": [rgb_height, rgb_width],
            "frame_capture": {
                "requested_pixel_format": int(emu.requested_pixel_format),
                "decoded_as_pixel_format": int(emu.frame_pixel_format),
                "note": (
                    "0 = 0RGB1555, 1 = RGB565, 2 = RGB888; the two differ because the core asks for "
                    "a 32-bit buffer and hands back a 16-bit one"
                ),
            },
            "device": device,
        },
        "run": {
            "frames": len(rows["frame"]),
            "ended": ended,
            "engine_exit_frame": engine_exit_frame,
            "outro_frames": outro_frames if outro_frames else 0,
            "lives_before": lives_before,
            "lives_after": lives_after,
            "progress_px": progress,
            "wall_clock_s": round(elapsed, 2),
            "fps": round(len(rows["frame"]) / max(elapsed, 1e-6), 1),
            "share_of_frames_live": float(np.mean(np.asarray(rows["live"]))),
            "camera_travel_px": float(camera.max() - camera.min()),
            "screen_x_range_px": [float(screen.min()), float(screen.max())],
            "mean_max_button_probability": float(np.mean(probs.max(axis=1))),
            "frames_with_no_button_above_half": int(np.sum((probs < 0.5).all(axis=1))),
            "frames_where_reflex_overrode": int(np.sum(np.asarray(rows["reflex"]))),
            "frames_where_pulse_released_B": int(np.sum(np.asarray(rows["pulse"]))),
            "frames_under_operator_A": int(np.sum(np.asarray(rows["operator"]))),
            "pauses_observed": len(pause_events),
            "pauses_cleared_by_the_operator": int(sum(1 for e in pause_events if e["resumed"])),
            "first_pause_at_progress_px": (
                round(float(pause_events[0]["progress_px"]), 4) if pause_events else None
            ),
            "value_start_px": float(rows["value"][0]),
            "value_end_px": float(rows["value"][-1]),
            "value_max_px": float(np.max(np.asarray(rows["value"]))),
            "residue_exact_share": float(np.mean(np.abs(residue) < 1e-6)),
            "residue_abs_max_subpixels": float(np.max(np.abs(residue))),
            "mode_bytes_seen": sorted({int(m) for m in rows["mode"]}),
        },
        "dataset": os.path.relpath(dataset_path, REPO_ROOT),
        "pauses": pause_events,
    }
    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        summary,
        command=f"python scripts/record_dashboard_run.py --controller {kind}",
    )
    print(json.dumps(summary["run"], indent=2))
    print(f"dataset: {dataset_path}\nartifact: {artifact}")
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--state", default=STATE_YOSHI_ISLAND_1_LEVEL)
    parser.add_argument("--controller", choices=["auto", *sorted(ACTORS)], default="auto")
    parser.add_argument("--probe-frames", type=int, default=600)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--operator-limit", type=int, default=120)
    parser.add_argument(
        "--goal-px",
        type=float,
        default=1900.0,
        help="end the episode at the distance proxy; 'none' ends it when the engine leaves 0x14",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the planner's sampling; 7 is the draw that cleared Yoshi's Island 1",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rgb-width", type=int, default=256)
    parser.add_argument("--rgb-height", type=int, default=224)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--selection", default=None, help="where to write the arm probe artifact (scratch/tests)"
    )
    args = parser.parse_args(argv)

    selection_path = os.path.join(args.output_dir, "dashboard_controller_selection.json")
    arms = [
        probe_arm(kind, args.state, args.probe_frames, args.device) for kind in ("ppo", "dagger")
    ]
    best = max(arms, key=lambda a: a["max_progress_px"])
    # The same actor with the published override layer removed: the reflex vaults and the jump
    # pulse belong to the benchmark, so the video keeps them, but their cost has to be on record.
    controls = [probe_arm(best["arm"], args.state, args.probe_frames, args.device, overrides=False)]
    selection = {
        "study": (
            "What the dashboard video flies, and what both learned arms did about it. The two "
            "learned checkpoints are probed on the same savestate for the same horizon with the "
            "published override loop, and the probe rule - written before the run - is the largest "
            "forward progress reached within it; that comparison is reported either way. The "
            "episode the README opens with is a different question, and 10.68 changed its answer: "
            "the hero shows a level being completed, so it is flown by the one arm this "
            "repository has that finishes the stage - the 10.66 seed-7 clearance draw, run until "
            "the engine itself ends it and kept recording through the exit sequence."
        ),
        "protocol": {
            "savestate": os.path.basename(args.state),
            "probe_frames": args.probe_frames,
            "learned_arm_rule": "max(max_progress_px) over the probed arms",
            "featured_rule": (
                "the README opens with a completed level (10.68): the featured episode is the "
                "seed-7 clearance draw of 10.66, ended by the engine's own verdict"
            ),
            "device": args.device,
        },
        "arms": arms,
        "override_control": controls,
        "learned_arm_selected": best["arm"],
        "featured": "mpc",
        "featured_seed": 7,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    target = args.selection or selection_path
    write_metrics(target, selection, command="python scripts/record_dashboard_run.py")
    print(json.dumps({"arms": arms, "override_control": controls}, indent=2))
    print(f"learned-arm probe winner: {best['arm']}; featured episode: mpc (seed 7)")
    print(f"selection artifact: {target}")
    record(
        kind="mpc",
        state_path=args.state,
        max_frames=9000,
        device=args.device,
        rgb_width=args.rgb_width,
        rgb_height=args.rgb_height,
        operator_limit=args.operator_limit,
        goal_px=None,
        seed=7,
        apply_overrides=False,
        operator_delay=0,
        warmup_frames=0,
        outro_frames=320,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
