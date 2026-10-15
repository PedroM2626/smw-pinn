r"""Gates for README 10.60: the hero block, the generated section, and the recording behind both.

The dashboard is the first artifact in this repository whose purpose is to be looked at, so the
gates refuse the easy failure modes: the hero caption and the section are compared line by line
against their generators (a typed number in a caption is exactly the defect this repository has
been fixing), the video files must exist and stay inside what a README can carry, and the claims the
prose makes about the episode - that the held frames are the residue exceptions, that the operator
pressed a button no policy has, that the selected arm is the one that traversed - are re-derived
from the recording rather than believed.

CPU-only, numpy-only; skipped when the dashboard pipeline has not been run.

Run:  pytest tests/test_dashboard_pipeline.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.environment import wram
from src.evaluation.dashboard_study import render_caption, render_section
from src.evaluation.dashboard_video import (
    COLS_BEHIND,
    EMPTY_CLASS,
    GRID_H,
    GRID_W,
    GROUND_RGB,
    INK,
    LEVEL_NAME_SOURCES,
    LEVEL_NAMES,
    RECT_TILE,
    ROWS_ABOVE,
    action_probabilities,
    render_indices,
    window_view,
)
from src.planning.mpc_planner import ACTION_BUTTONS, ACTION_MATRIX

REPO = Path(__file__).resolve().parents[1]
STUDY_ARTIFACT = REPO / "results" / "dashboard_study_metrics.json"
RUN_ARTIFACT = REPO / "results" / "dashboard_run_metrics.json"
VIDEO_ARTIFACT = REPO / "results" / "dashboard_video_metrics.json"
SELECTION_ARTIFACT = REPO / "results" / "dashboard_controller_selection.json"
RECORDING = REPO / "data" / "raw" / "smw_dashboard_run.npz"
STILL = REPO / "results" / "figures" / "dashboard_run_still.png"


def _payload():
    if not STUDY_ARTIFACT.is_file():
        pytest.skip("the dashboard study has not been run")
    return json.loads(STUDY_ARTIFACT.read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO / "README.md").read_text(encoding="utf-8")


def _recording():
    if not RECORDING.is_file():
        pytest.skip("the dashboard episode has not been recorded")
    with np.load(RECORDING, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def test_readme_10_60_is_the_generated_block() -> None:
    """Not one number in the section is typed."""
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.60 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.60 "), "the generator must start at the section heading"


def test_readme_hero_is_the_generated_caption_at_the_top() -> None:
    """The caption a reader meets first is the one a gate can check, and it is above the abstract."""
    lines = render_caption(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "the README hero disagrees with the generator:\n" + "\n".join(missing[:4])
    hero = text.index(lines[0])
    assert hero < text.index("## 🏆 Direct Answer"), "the hero belongs at the top of the file"


def test_the_hero_media_exists_and_is_a_readme_can_carry() -> None:
    """A GIF that is too big for the page is the same failure as a missing one."""
    payload = _payload()
    for path in (payload["video"]["mp4"], payload["gif"]["gif"]):
        file = Path(path)
        if not file.is_absolute():
            file = REPO / path
        assert file.is_file(), f"{file} is referenced by the README and not committed"
        assert file.stat().st_size < 10_000_000, f"{file} is over GitHub's per-file comfort limit"
    rel = payload["gif"]["gif"].replace("\\", "/")
    assert f"]({rel})" in _readme(), "the hero must embed the GIF, not merely mention it"


def test_the_recording_carries_every_channel_the_dashboard_draws() -> None:
    data = _recording()
    for key in (
        "states",
        "button_probs",
        "executed",
        "values",
        "camera",
        "patches",
        "rgb",
        "live",
        "residue_subpixels",
        "operator",
        "mode",
        "tile_ids",
        "sprites",
    ):
        assert key in data, f"the panel set of 10.68 needs a {key} channel"
    n = data["states"].shape[0]
    assert data["states"].shape == (n, 12)
    assert data["button_probs"].shape == (n, len(ACTION_BUTTONS))
    assert data["rgb"].shape[1:] == (224, 256, 3), "the frame must be full width, not half"
    assert data["patches"].shape[1:] == (33, 33)
    assert data["tile_ids"].shape[1:] == (33, 33), "the recorded window is 33x33 full ids"
    assert data["sprites"].shape[1:] == (12, 4), (
        "every sprite slot is recorded, not just the nearest"
    )
    assert int(data["tile_ids"].min()) == -1, "out-of-buffer cells are recorded as -1"
    # the panel classifies the ids through the measured table, so every recorded id must be one
    # the table knows - an id the probe never measured would draw as unmeasured, and the caption
    # says it does not
    inside = data["tile_ids"][data["tile_ids"] >= 0]
    assert int(inside.max(initial=0)) < 0x200, "per-level ids at 0x200+ would need their own probe"
    # A planner-headed recording commits to whole primitives: its button probabilities are the
    # chosen row, and an all-zero row is the honest idle - a frame with no button at all. What
    # the gate holds is the recording against its own summary, so a frame the summary counts as
    # having no likely button must be on the recording.
    probs = np.asarray(data["button_probs"], dtype=np.float64)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    no_likely = int(np.sum((probs < 0.5).all(axis=1)))
    assert no_likely == run["run"]["frames_with_no_button_above_half"], (
        "the summary's count of frames with no likely button disagrees with the recording"
    )
    if "primitive_probs" in data:
        # the live rows carry the actor's one-hot commitment; the outro rows are the actor-less
        # epilogue (all zeros, honestly) and the recording stays rectangular through them
        live = np.asarray(data["live"], dtype=np.float64) > 0.5
        np.testing.assert_allclose(
            np.asarray(data["primitive_probs"], dtype=np.float64)[live].sum(axis=1),
            1.0,
            atol=1e-4,
        )
    assert set(np.unique(data["executed"]).tolist()) <= {0.0, 1.0}


def test_every_visible_enemy_slot_reaches_the_grid_and_the_reel() -> None:
    """The WRAM tables, not visual sampling, account for every enemy the panel can show."""
    from src.environment import sprite_behavior

    data = _recording()
    states = np.asarray(data["states"], dtype=np.float64)
    sprites = np.asarray(data["sprites"], dtype=np.float64)
    view = window_view(data["tile_ids"], sprites, states)
    visible = 0
    for frame, slot in zip(*np.nonzero(sprites[:, :, 3] >= sprite_behavior.SPRITE_ACTIVE)):
        kind = sprite_behavior.behavior_of(int(sprites[frame, slot, 2]))
        if kind not in sprite_behavior.HURTING:
            continue
        mario_x, mario_y = states[frame, :2]
        mario_tx = int(mario_x + 8.0) // 16
        mario_ty = int(mario_y + 16.0) // 16
        sprite_x = int(np.rint(mario_x + sprites[frame, slot, 0]))
        sprite_y = int(np.rint(mario_y + sprites[frame, slot, 1]))
        col = (sprite_x + 8) // 16 - mario_tx + COLS_BEHIND
        row = (sprite_y + 8) // 16 - mario_ty + ROWS_ABOVE
        if 0 <= row < GRID_H and 0 <= col < GRID_W:
            visible += 1
            assert view[frame, row, col] == 4, (
                f"visible enemy slot {slot} at frame {frame} was overwritten in the grid"
            )

    assert visible > 100, "the completed-stage recording did not carry its visible enemy population"
    red_cells = np.sum(view == 4, axis=(1, 2))
    assert int(red_cells.max()) >= 2, "the grid never shows multiple simultaneous enemies"
    indices = render_indices(states[:, 0], max(2, -(-len(states) // 700)))
    assert int(np.any(view[indices] == 4, axis=(1, 2)).sum()) >= 20, (
        "the reel's sampling removed almost every frame that contained an enemy"
    )
    assert int(np.sum(view[indices] == 4, axis=(1, 2)).max()) >= 2, (
        "no rendered GIF frame carries multiple simultaneous enemies"
    )


def test_the_featured_episode_and_the_learned_arm_probe() -> None:
    """The hero is the completed level; the probe comparison is reported either way.

    10.68 changed what the README opens with: a level being completed, flown by the one
    controller that finishes it. The two learned arms are still probed under the rule that was
    written before the run, and the gate holds both facts: the featured episode is the seeded
    clearance draw, and the probe's own winner is the distilled actor.
    """
    selection = json.loads(SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    assert selection["featured"] == "mpc", "the hero shows a completed level"
    assert selection["featured_seed"] == 7, "the 10.66 clearance draw itself"
    arms = selection["arms"]
    best = max(arms, key=lambda a: a["max_progress_px"])
    assert selection["learned_arm_selected"] == best["arm"] == "dagger"
    ppo = next(a for a in arms if a["arm"] == "ppo")
    assert ppo["progress_px"] < 0.0, "the section's claim is that the PPO arm walks backwards"
    assert ppo["mean_top_probability"] < 0.5, "and that its distribution has not decided anything"
    control = selection["override_control"][0]
    assert control["published_overrides"] is False
    assert control["progress_px"] > best["max_progress_px"], (
        "10.60 says the published override layer costs the actor progress; if a re-run flips that "
        "sign the sentence, not the number, has to change"
    )


def test_the_hero_names_the_instrument_it_was_recorded_on() -> None:
    """The caption's provenance is read, not asserted.

    The hero sentence used to state "the verified level entry of 10.36.3" as a literal, so
    re-recording the episode from another savestate left the claim standing and false, and every
    numeric gate still passed: the numbers were right about the new file and only the origin was
    wrong. The caption now derives the filename from the recording, and this gate holds the two
    facts the prose depends on - the featured episode and the arm probes ran on one instrument,
    and that instrument is the level-entry state every picture claim of 10.36.3 rests on.
    """
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_ARTIFACT.read_text(encoding="utf-8"))
    hero_state = run["protocol"]["savestate"]
    assert hero_state == selection["protocol"]["savestate"], (
        "10.60 probes the learned arms and flies the hero on one savestate; the artifacts disagree"
    )
    assert hero_state == "smw_yoshi_island_1_level.state", (
        f"the hero is recorded from {hero_state}, not the 10.36.3 level-entry state that the "
        "repository's picture claims are anchored to"
    )
    caption = "\n".join(render_caption(_payload()))
    assert f"`{hero_state}`" in caption, (
        "the hero caption must name the file the episode was recorded from"
    )
    assert "10.36.3" not in caption, (
        "the caption carries a section number where it should carry the instrument's name"
    )


def test_every_drawn_level_name_declares_its_evidence() -> None:
    """A stage name is a claim, and the table says which rows it measured and which it copied.

    The community's translevel table names `0x29` Yoshi's Island 1 and `0x2A` Yoshi's Island 2,
    while this repository's own anchor - the node a fresh file starts on, walked to by 10.36.3 -
    reads `0x2A` there. The rows are therefore not all the same kind of statement, and the panel
    must not let them read as if they were. The measurement that would settle it is unavailable:
    entering the known `0x2A` draws no title card either, so the null cannot clear the guess.
    """
    assert set(LEVEL_NAME_SOURCES) == set(LEVEL_NAMES), (
        "every drawn name needs an evidence label, and no label may point at a name that is gone"
    )
    assert LEVEL_NAME_SOURCES["0x2a"] == "measured here"
    assert LEVEL_NAME_SOURCES["0x28"] == "measured here", (
        "the house is identified by its own measurements: no timer, one subscreen, door exit"
    )
    for inferred in ("0x27", "0x29"):
        assert LEVEL_NAME_SOURCES[inferred] != "measured here", (
            f"{inferred} is a table reading shifted to the anchor, not a measurement"
        )
        assert "table" in LEVEL_NAME_SOURCES[inferred]
    video = json.loads(VIDEO_ARTIFACT.read_text(encoding="utf-8"))
    assert video["protocol"]["level_name_source"] == "measured here", (
        "the hero is drawn from the stage this repository identified itself"
    )


def test_held_frames_are_the_residue_exceptions() -> None:
    """The section's central reading, re-derived per row: residue = 16(x_next - x) - vx.

    The live flag describes the step *before* a row (its record against its predecessor) and
    the residue the step *after* it, so a row can carry a held flag and a moving identity at
    once - which is exactly what the engine's one-frame holds and the goal tape's half-rate
    walk produce. The identity itself has to hold on every row with a successor except the one
    whose step crossed the exit (its post-read is the warp), the rows whose step did not move
    have to read exactly minus the velocity, and the whole-pixel re-anchoring 10.59 called a
    reposition stays among the moving rows, where the per-row identity holds by construction.
    """
    data = _recording()
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    states = np.asarray(data["states"], dtype=np.float64)
    residue = np.asarray(data["residue_subpixels"], dtype=np.float64)
    vx = states[:, 2]
    steps = states[1:, 0] - states[:-1, 0]
    identity = 16.0 * steps - vx[:-1]
    exit_frame = int(run["run"]["engine_exit_frame"])
    bad = np.nonzero(np.abs(residue[:-1] - identity) > 1e-6)[0]
    assert all(int(i) in (exit_frame - 1, exit_frame) for i in bad), (
        f"{len(bad)} rows break the per-row identity outside the exit step: {bad[:4].tolist()}"
    )
    frozen = np.abs(steps) < 1e-9
    nonzero = np.abs(residue[:-1]) > 1e-6
    assert np.allclose(residue[:-1][nonzero & frozen], -vx[:-1][nonzero & frozen], atol=1e-6), (
        "a row whose step did not move must read exactly minus the velocity - the holds and the "
        "goal tape's half-rate walk are the exceptions the section names"
    )


def test_the_operator_pressed_a_button_no_policy_has() -> None:
    """A is outside the action set, which is the whole point of the operator column."""
    assert "A" not in ACTION_BUTTONS
    data = _recording()
    operator = np.asarray(data["operator"], dtype=np.float64)
    held = data["live"] < 0.5
    assert operator.sum() > 0.0, "the published episode has a takeover in it"
    assert np.all(held[operator > 0.5]), "an operator frame must be a held frame"
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    assert run["run"]["frames_under_operator_A"] == int(operator.sum())
    for event in run["pauses"]:
        assert event["resumed"], (
            "10.60 describes every pause as cleared; an uncleared one rewrites it"
        )


def test_the_video_numbers_are_the_recording_numbers() -> None:
    """The caption quotes the render, the render quotes the recording; both must agree."""
    payload = _payload()
    data = _recording()
    states = np.asarray(data["states"], dtype=np.float64)
    progress = states[:, 0] - states[0, 0]
    live = np.asarray(data["live"], dtype=np.float64) > 0.5
    caption = payload["caption"]
    assert caption["episode_frames"] == states.shape[0]
    assert caption["progress_px"] == pytest.approx(float(progress[-1]), abs=1e-3)
    # the residue share is quoted over the frames the player was live: the outro that follows
    # the engine's verdict is a held epilogue, and folding it in would quote the exit sequence
    # as if the identity were failing on it
    assert caption["residue_exact_share"] == pytest.approx(
        float(np.mean(np.abs(data["residue_subpixels"][live]) < 1e-6)), abs=1e-9
    )
    assert caption["operator_A_frames"] == int(np.asarray(data["operator"]).sum())
    assert caption["subscreens_reached"] == int(progress[-1] // 256) + 1
    views = window_view(data["tile_ids"], data["sprites"], states)
    indices = render_indices(states[:, 0], max(2, -(-len(states) // 700)))
    assert caption["frames_with_enemy_cells"] == int(np.any(views == 4, axis=(1, 2)).sum())
    assert caption["rendered_frames_with_enemy_cells"] == int(
        np.any(views[indices] == 4, axis=(1, 2)).sum()
    )
    assert caption["max_enemy_cells_in_frame"] == int(np.sum(views == 4, axis=(1, 2)).max())
    active = np.asarray(data["sprites"])[:, :, 3] >= 8
    assert caption["active_sprite_rows"] == int(active.sum())
    assert caption["max_active_sprite_slots_in_frame"] == int(active.sum(axis=1).max())


def test_the_frame_buffer_defect_is_recorded_as_a_fact() -> None:
    """10.60's second instrument: the core's request and the buffer's format must be on record."""
    run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    capture = run["protocol"]["frame_capture"]
    assert capture["requested_pixel_format"] == 2
    assert capture["decoded_as_pixel_format"] == 1
    assert capture["requested_pixel_format"] != capture["decoded_as_pixel_format"], (
        "if the core ever starts honouring its own request, the section's sentence about it is stale"
    )


def test_window_view_classifies_ids_and_marks_missing_cells() -> None:
    """The panel's window is the measured table's answer for each recorded 16-bit id.

    The 33x33 recorded grid is classified and then cropped to 20x14 with Mario at
    (ROWS_ABOVE, COLS_BEHIND). A cell the buffer does not have stays empty (class 0).
    """

    radius = 16
    ids = np.full((1, 2 * radius + 1, 2 * radius + 1), -1, dtype=np.int64)
    centre_x, centre_y = 39, 22
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if wram.tilemap_addr(centre_x + dx, centre_y + dy) is not None:
                ids[0, dy + radius, dx + radius] = 0x0025
    ids[0, radius + 2, radius] = 0x0100
    ids[0, radius - 2, radius + 1] = 0x0124
    ids[0, radius - 1, radius + 2] = 0x013F
    states = np.zeros((1, 12), dtype=np.float64)
    states[0, :2] = (centre_x * 16, centre_y * 16)
    view = window_view(ids, np.zeros((1, 12, 4)), states)
    assert view.shape == (1, GRID_H, GRID_W)
    assert view[0, ROWS_ABOVE + 1, COLS_BEHIND] == 1, "the grass top is ground/wall"
    assert view[0, ROWS_ABOVE - 3, COLS_BEHIND + 1] == 1, "the stair corner is ground/wall"
    assert view[0, ROWS_ABOVE - 2, COLS_BEHIND + 2] == 3, "a harmful tile is not an enemy"
    assert view[0, ROWS_ABOVE, COLS_BEHIND] == 0, "the blank filler is empty"
    assert int((view[0] == EMPTY_CLASS).sum()) > 0, (
        "a window centred on the floor must have empty cells"
    )
    assert int((view[0] == EMPTY_CLASS).sum()) < view[0].size, "and must not be all empty"


def test_window_view_draws_live_sprites_as_cells() -> None:
    """The reference grid's presentation: the sprite occupies a cell in the terrain window.

    The view is now cropped to 20x14, so sprites must be within the visible region to appear.
    """
    from src.evaluation.dashboard_video import SPRITE_CELL_CLASSES

    radius = 16
    ids = np.full((1, 2 * radius + 1, 2 * radius + 1), 0x0025, dtype=np.int64)
    sprites = np.zeros((1, 12, 4))
    sprites[0, 0] = (-11.0, 44.0, 0x05, 9.0)  # a live Rex, one cell left, three below
    sprites[0, 1] = (120.0, 0.0, 0x91, 8.0)  # a message block far right
    sprites[0, 2] = (16.0, 0.0, 0x01, 8.0)  # a shell one cell right of Mario
    sprites[0, 3] = (0.0, 0.0, 0x05, 1.0)  # dormant: spawn-list status, no cell
    states = np.zeros((1, 12), dtype=np.float64)
    states[0, :2] = (64.0, 352.0)
    view = window_view(ids, sprites, states)
    assert view.shape == (1, GRID_H, GRID_W)
    assert view[0, ROWS_ABOVE + 2, COLS_BEHIND - 1] == SPRITE_CELL_CLASSES["stompable"], (
        "the Rex's centre marks its cell as an enemy cell"
    )
    msg_col = COLS_BEHIND + 8
    sprite_row = ROWS_ABOVE - 1
    if msg_col < GRID_W:
        assert view[0, sprite_row, msg_col] == SPRITE_CELL_CLASSES["interact"], (
            "the message block marks its exact hitbox-centre cell eight columns right"
        )
    assert view[0, sprite_row, COLS_BEHIND + 1] == SPRITE_CELL_CLASSES["neutral"], (
        "the parked shell marks its exact hitbox-centre cell one column right"
    )
    assert view[0, sprite_row, COLS_BEHIND] == 0, (
        "the dormant slot at Mario's horizontal cell must not paint an enemy"
    )


def test_sprite_cells_use_absolute_tile_boundaries_not_rounded_displacements() -> None:
    """Eight pixels can cross a tile boundary even though round(8 / 16) is zero."""
    radius = 16
    ids = np.full((1, 2 * radius + 1, 2 * radius + 1), 0x0025, dtype=np.int64)
    sprites = np.zeros((1, 12, 4), dtype=np.float64)
    states = np.zeros((1, 12), dtype=np.float64)
    states[0, :2] = (7.0, 352.0)
    sprites[0, 0] = (8.0, 0.0, 0x05, 8.0)

    view = window_view(ids, sprites, states)
    sprite_row = ROWS_ABOVE - 1
    assert view[0, sprite_row, COLS_BEHIND + 1] == 4
    assert view[0, sprite_row, COLS_BEHIND] == 0


def test_the_tile_panel_draws_the_ground_below_mario() -> None:
    """The orientation, checked on the rendered pixels rather than on the code that renders them.

    The floor band is detected by the ground/wall colour: ground below Mario, empty cells (from
    outside the tile buffer) further below. The ground band's centroid must sit below the panel's
    vertical centre.
    """
    if not STILL.is_file():
        pytest.skip("the dashboard render has not been run")
    import imageio.v2 as imageio

    canvas = imageio.imread(STILL)[:, :, :3].astype(int)
    height, width, _ = canvas.shape
    x0, y1, w, h = RECT_TILE
    left, right = int(x0 * width), int((x0 + w) * width)
    top, bottom = int((1.0 - (y1 + h)) * height), int((1.0 - y1) * height)
    middle = (top + bottom) // 2
    panel = canvas[top:bottom, left:right]

    def pixels(block: np.ndarray, colour: tuple) -> int:
        distance = np.abs(block.reshape(-1, 3) - np.asarray(colour)).sum(axis=1)
        return int((distance < 12).sum())

    lower = canvas[middle:bottom, left:right]
    assert pixels(lower, GROUND_RGB) > 30, "the panel has no ground colour under Mario at all"
    mask = np.abs(panel - np.asarray(GROUND_RGB)).sum(axis=2) < 12
    counts = mask.sum(axis=1)
    if counts.sum() > 0:
        centroid = float((np.arange(panel.shape[0]) * counts).sum() / counts.sum())
        assert centroid > panel.shape[0] / 2.0, "the ground band's centre of mass is above Mario"


def test_the_mario_marker_occupies_the_declared_grid_row() -> None:
    """An upper-origin image needs its row index converted back to axes coordinates."""
    if not STILL.is_file():
        pytest.skip("the dashboard render has not been run")
    import imageio.v2 as imageio

    canvas = imageio.imread(STILL)[:, :, :3].astype(int)
    height, width, _ = canvas.shape
    x0, y1, w, h = RECT_TILE
    left, right = int(x0 * width), int((x0 + w) * width)
    top, bottom = int((1.0 - (y1 + h)) * height), int((1.0 - y1) * height)
    cell_x = left + int((COLS_BEHIND + 0.5) * (right - left) / GRID_W)
    cell_y = top + int((ROWS_ABOVE + 0.5) * (bottom - top) / GRID_H)
    mirrored_y = top + int((GRID_H - ROWS_ABOVE - 0.5) * (bottom - top) / GRID_H)
    ink = np.asarray(tuple(int(INK[i : i + 2], 16) for i in (1, 3, 5)))

    def ink_pixels(cx: int, cy: int) -> int:
        block = canvas[cy - 8 : cy + 9, cx - 8 : cx + 9]
        return int((np.abs(block - ink).sum(axis=2) < 18).sum())

    assert ink_pixels(cell_x, cell_y) > 8, "Mario is not outlined at row 8 of the upper-origin grid"
    assert ink_pixels(cell_x, mirrored_y) < ink_pixels(cell_x, cell_y), (
        "the marker is still mirrored vertically above its declared grid row"
    )


@pytest.mark.parametrize("head", ["primitive", "marginal"])
def test_the_action_panel_shows_a_distribution_that_exists(head: str) -> None:
    """A button-headed actor does not emit primitive probabilities; the joint that stands in for
    them is derived from its own marginals and labelled, and either way the bars must sum to one."""
    rng = np.random.default_rng(0)
    marginals = rng.uniform(0.0, 1.0, size=(4, len(ACTION_BUTTONS)))
    data = {"button_probs": marginals}
    if head == "primitive":
        raw = rng.uniform(0.0, 1.0, size=(4, len(ACTION_MATRIX)))
        data["primitive_probs"] = raw / raw.sum(axis=1, keepdims=True)
    actions, source = action_probabilities(data)
    np.testing.assert_allclose(actions.sum(axis=1), 1.0, atol=1e-9)
    assert (actions >= 0.0).all()
    assert ("actor's own" if head == "primitive" else "marginals") in source
