"""
level_dashboard_study.py - the README's multi-stage dashboard: what the map gives, and what the
hero shows when it is pointed at each of those stages.

10.60 built the hero on one entered level and 10.66 asked for more. Three questions have to be
answered before a second picture means anything, and this module only reports answers that were
measured:

* **What is out there?** `scripts/survey_world_map.py` walks the map cursor and presses A on every
  position it reaches. A position is a stage when the engine leaves the map, loads a timer and gives
  control to a player that walks.
* **Does anything clear?** `scripts/unlock_world_map.py` flies the published clearance controller at
  the entered stage and reads the mode byte. The engine returning to the map at the *same* node is a
  game over, so a clear needs the cursor to move.
* **What does the actor do per stage?** `scripts/record_level_dashboards.py` records one episode each
  and measures the distance a held-right walk reaches in that stage, which is the only denominator
  this repository has evidence for.

Run:  python -m src.evaluation.level_dashboard_study
      make level-dashboard-study
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.evaluation.dashboard_video import level_label, run_dashboard_render
from src.utils.logging import get_logger
from src.utils.paths import REPO_ROOT, dataset_path, figure_file, results_file

logger = get_logger(__name__)

SURVEY_ARTIFACT = "results/world_map_survey.json"
PROGRESSION_ARTIFACT = "results/world_map_clearance.json"
STUDY_ARTIFACT = "results/level_dashboards_cleared.json"
RUN_ARTIFACT = "results/dashboard_run_metrics.json"
#: the 3000-frame campaign, whose budget - not the level - is what stopped its best run
PRIOR_CLEARANCE_ARTIFACT = "results/world_map_progression.json"


def _read(path: str) -> Dict[str, Any]:
    with open(
        path if os.path.isabs(path) else os.path.join(REPO_ROOT, path), encoding="utf-8"
    ) as f:
        return json.load(f)


def _rel(path: str) -> str:
    resolved = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    return os.path.relpath(resolved, REPO_ROOT).replace("\\", "/")


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def stage_name(identity: Dict[str, Any], node: Any) -> str:
    """What to call a stage when only the engine's own bytes identify it."""
    translevel = str(identity.get("translevel", ""))
    named = level_label(translevel)
    if named != translevel.lower():
        return named
    return f"level {translevel}" if node is None else f"level {translevel} at node {node}"


def stage_width_px(stage: Dict[str, Any]) -> float:
    """How far the stage's own tile buffer holds terrain, in pixels.

    This is the denominator the progress bar and the bottom line want: not a goal distance nobody has
    seen the engine confirm, and not how far some controller happened to get, but the width of the
    level as the console stores it. A run that stops short of it stopped early; a run that reaches it
    has walked the whole stage.
    """
    return float(stage["terrain"]["terrain_px"])


def bottom_line(stage: Dict[str, Any], progress: Optional[float] = None) -> str:
    """The corner the mock puts a clearance rate in, with the verdict the engine itself gave.

    Two numbers share the line because one of them is about the stage and the other is about the
    policy on screen: whether the engine ended the stage without the death mode - the 10.66
    discriminator, read off the episode's own mode channel - and how far the actor the dashboard
    is actually showing got against the width the stage's buffer holds.
    """
    value = float(stage["episode"]["progress_px"]) if progress is None else float(progress)
    width = stage_width_px(stage)
    evidence = stage.get("clear_evidence") or {}
    if evidence.get("cleared"):
        rate = "cleared: the engine ended the stage without the death mode"
    elif int(stage["mpc"]["tries"]) > 0:
        clears = int(stage["mpc"]["clears"])
        tries = int(stage["mpc"]["tries"])
        rate = (
            f"cleared in {clears} of {tries} attempts"
            if clears
            else f"0 of {tries} attempts cleared it"
        )
    else:
        rate = "no run has ever finished this session"
    return f"{rate} · this arm {value:.0f} / {width:.0f} px"


def render_contexts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One dashboard context per stage: the title bar, and the line the mock puts a rate in."""
    stages = payload["stages"]
    count = len(stages)
    contexts: List[Dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        # The bar is scaled to the width the stage's own buffer holds, not to the 1900 px goal proxy
        # published in 10.37 - which 10.66 shows the engine never corroborates - and not to how far
        # some controller happened to get, which says more about the budget than about the stage.
        span = max(stage_width_px(stage), 256.0)
        contexts.append(
            {
                "title": "Reinforcement learning in Super Mario World",
                "level_label": stage_name(stage["identity"], stage["map_node"]),
                "level_index": index,
                "level_count": count,
                "clear_line": bottom_line(stage),
                "goal_px": span,
                "dataset": stage["episode"]["dataset"],
                "output_stem": f"dashboard_level_{index}",
                "progress_px": float(stage["episode"]["progress_px"]),
            }
        )
    return contexts


#: keys the render context carries for the study's own bookkeeping, which the renderer never draws
#  and which must therefore never reach the render artifact: a `dataset` in the picture's protocol
#: naming a file other than the one the picture was drawn from is a provenance claim that is false
#: in the artifact that exists to check provenance against.
BOOKKEEPING_KEYS = ("dataset", "output_stem", "progress_px")


def drawing_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """The context as the renderer may see it: the drawing keys, and nothing else."""
    return {k: v for k, v in context.items() if k not in BOOKKEEPING_KEYS}


def hero_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The README hero's context: the published episode's stage, drawn as one of the set.

    The hero stays pointed at the recording 10.60 published - the arm probe, the pause table and the
    residue claim are all chained to that file - but the picture now says which stage it is and how
    far the arm on screen got against the width the stage itself has. The progress on that line is
    the hero recording's own, not the stage's: the two episodes of Yoshi's Island 1 start from
    different captures of the same entry, and a line that quoted the wrong one would be the mistake
    this file keeps having to unmake.
    """
    contexts = render_contexts(payload)
    hero_progress = float(_read(RUN_ARTIFACT)["run"]["progress_px"])
    yoshi = next(
        (c for c in contexts if c["level_label"] == level_label("0x2A")),
        contexts[0] if contexts else {},
    )
    stage = next(
        (
            s
            for s in payload["stages"]
            if stage_name(s["identity"], s["map_node"]) == yoshi.get("level_label")
        ),
        None,
    )
    out = drawing_context(yoshi)
    out["level_index"] = 1
    out["level_count"] = max(1, len(contexts))
    if stage is not None:
        out["goal_px"] = max(stage_width_px(stage), 256.0)
        out["clear_line"] = bottom_line(stage, hero_progress)
    return out


def render_level_videos(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Render one MP4, GIF and still per stage, and keep the numbers each one drew."""
    rendered: List[Dict[str, Any]] = []
    strides = {s["episode"]["dataset"]: int(s.get("render_stride", 2)) for s in payload["stages"]}
    for context in render_contexts(payload):
        dataset = dataset_path(os.path.basename(context["dataset"]))
        stem = context["output_stem"]
        art = run_dashboard_render(
            dataset_path=dataset,
            output_mp4=figure_file(f"{stem}.mp4"),
            output_gif=figure_file(f"{stem}.gif"),
            output_still=figure_file(f"{stem}_still.png"),
            output_metrics=None,
            stride=strides.get(context["dataset"], 2),
            context=drawing_context(context),
        )
        rendered.append(
            {
                "level_index": context["level_index"],
                "label": context["level_label"],
                "dataset": _rel(dataset),
                "goal_px": float(art["protocol"]["goal_px"]),
                "clear_line": context["clear_line"],
                "video": art["video"],
                "gif": art["gif"],
                "still": art["still"],
                "caption": art["caption"],
            }
        )
        logger.info(f"level dashboard rendered: {context['level_label']} -> {art['video']['mp4']}")
    return rendered


def render_survey_table(survey: Dict[str, Any]) -> List[str]:
    lines = [
        "| map cursor | reached by | translevel | timer | right-press dx | the engine gave a stage |",
        "|---|---|---|---|---|---|",
    ]
    for node in survey["nodes"]:
        path = "+".join(node["path"]) if node["path"] else "(start)"
        dx = node.get("right_response_px")
        lines.append(
            f"| {tuple(node['node'])} | `{path}` | `{node['translevel']}` | "
            f"{node['timer_hundreds']} | {dx if dx is not None else '-'} | "
            f"{'yes' if node.get('walkable') else 'no'} |"
        )
    return lines


def attempt_cleared(attempt: Dict[str, Any]) -> bool:
    """The one definition of a cleared flight, shared by the table, the prose and the gates.

    The campaign records the map wait under `after_exit` and the recorder under `reached_the_map`;
    two spellings of the same fact is how a table ends up denying what the paragraph above it claims,
    so both are read here and nowhere else.
    """
    reached = bool((attempt.get("after_exit") or {}).get("on_map")) or bool(
        attempt.get("reached_the_map")
    )
    return reached and not bool(attempt.get("died_on_the_way"))


def render_clearance_table(clearance: Dict[str, Any]) -> List[str]:
    """One row per flight: what the controller reached, and what the engine said about how it ended.

    The two right-hand columns are the whole point. A run that leaves the level through `0x0B` died;
    one that reaches the map without it finished the stage. The cursor is read only after the map has
    been given `MAP_SETTLE` frames to stop reporting the node that was left.
    """
    lines = [
        "| round | stage | attempt | px | frames | ended in | died | the map at | cleared |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for round_record in clearance["rounds"]:
        stage = f"`{round_record.get('translevel')}` at {tuple(round_record.get('node') or [])}"
        for attempt in round_record.get("attempts") or []:
            lines.append(
                f"| {round_record['round']} | {stage} | {attempt['attempt']} | "
                f"{attempt['max_progress_px']:.4f} | {attempt['frames']} | "
                f"`{attempt['final_mode']}` | "
                f"{'yes' if attempt.get('died_on_the_way') else 'no'} | "
                f"{attempt.get('map_cursor') or '-'} | "
                f"{'**yes**' if attempt_cleared(attempt) else 'no'} |"
            )
    return lines


def render_stage_table(payload: Dict[str, Any]) -> List[str]:
    lines = [
        "| stage | translevel | terrain the buffer holds | rightward walk reaches | "
        "flown by | the episode | share | the engine's verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for stage in payload["stages"]:
        frontier = stage["walk_right_frontier"]
        episode = stage["episode"]
        evidence = stage["clear_evidence"]
        width = stage_width_px(stage)
        share = episode["progress_px"] / width if width > 0 else float("nan")
        verdict = (
            f"**cleared** (left `0x14` for `{evidence['exit_mode']}`)"
            if evidence["cleared"]
            else f"no ({evidence['reason']})"
        )
        lines.append(
            f"| `{stage_name(stage['identity'], stage['map_node'])}` | "
            f"`{stage['identity']['translevel']}` | {width:.0f} px "
            f"({stage['terrain']['terrain_subscreens']} subscreen"
            f"{'s' if stage['terrain']['terrain_subscreens'] != 1 else ''}) | "
            f"{frontier['frontier_px']:.4f} px ({frontier['ended']}) | "
            f"`{episode['controller']}` | {episode['progress_px']:.4f} px in "
            f"{episode['frames']} frames | {_pct(share)} | {verdict} |"
        )
    return lines


def gather(rendered: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    survey = _read(SURVEY_ARTIFACT)
    progression = _read(PROGRESSION_ARTIFACT)
    study = _read(STUDY_ARTIFACT)
    return {
        "study": (
            "The dashboard on more than one stage: what the world map actually hands over, what the "
            "clearance controller does when it is flown at the entered level with the mode byte "
            "watched, and one recorded episode per stage with the distance a rightward walk reaches."
        ),
        "protocol": {
            "survey": "scripts/survey_world_map.py",
            "progression": "scripts/unlock_world_map.py",
            "recorder": "scripts/record_level_dashboards.py",
            "renderer": "src/evaluation/dashboard_video.py",
            "survey_protocol": survey["protocol"],
            "progression_protocol": progression["protocol"],
            "recorder_protocol": study["protocol"],
        },
        "survey": survey,
        "progression": progression,
        "prior_clearance": (
            _read(PRIOR_CLEARANCE_ARTIFACT) if os.path.isfile(PRIOR_CLEARANCE_ARTIFACT) else None
        ),
        "stages": study["stages"],
        "rendered": rendered or [],
    }


def render_section(payload: Dict[str, Any]) -> List[str]:
    r"""The whole of README section 10.66, generated from the artifacts."""
    survey = payload["survey"]
    progression = payload["progression"]
    positions = survey["discovery"]["positions"]
    walkable = [n for n in survey["nodes"] if n.get("walkable")]
    distinct = sorted({str(n["translevel"]) for n in walkable})
    cleared = progression.get("cleared_rounds", 0)
    new_positions = progression.get("map_after", {}).get("new_positions", [])
    rendered = {int(r["level_index"]): r for r in payload.get("rendered", [])}
    yoshi = next(
        (s for s in payload["stages"] if str(s["identity"]["translevel"]).lower() == "0x2a"),
        payload["stages"][0],
    )
    yoshi_width = stage_width_px(yoshi)
    prior = payload.get("prior_clearance") or {}
    prior_budget = (prior.get("protocol") or {}).get("max_frames_per_attempt", "-")
    best_prior = max(
        (
            max((a["max_progress_px"] for a in r.get("attempts") or []), default=float("nan"))
            for r in prior.get("rounds") or [{"attempts": []}]
        ),
        default=float("nan"),
    )
    cleared_rounds = [r for r in progression["rounds"] if r.get("cleared")]
    if cleared_rounds:
        win = cleared_rounds[0]
        clear = win["cleared"]
        replay = clear.get("replay") or {}
        next_round = next(
            (r for r in progression["rounds"] if r["round"] > win["round"]),
            None,
        )
        opened = (
            f"and the stage behind the new node is `{next_round['translevel']}`, at node "
            f"{tuple(next_round['node'])}, {next_round['terrain']['terrain_px']} px of terrain"
            if next_round and next_round.get("terrain")
            else "and the map was not entered again after it"
        )
        attempt_record = next(
            (a for a in win.get("attempts", []) if a["attempt"] == clear["attempt"]),
            {},
        )
        clearance_sentence = (
            f"The stage `{win['translevel']}` at node {tuple(win['node'])} - the one 10.36.3 built "
            f"the savestate for - was finished by attempt {clear['attempt']} of round "
            f"{win['round']}: {attempt_record.get('max_progress_px', float('nan')):.4f} px in "
            f"{attempt_record.get('frames', 0)} frames, the mode going `0x14` to "
            f"`{clear['exit_modes'][0]}` with no `0x0B` anywhere, and the map settling at node "
            f"{tuple(clear['cursor_after'])} - a node the fresh file cannot reach. Re-flown from the "
            f"same savestate with the same seed, the run reached "
            f"{replay.get('max_progress_px', float('nan')):.4f} px, a difference of "
            f"{replay.get('progress_difference_px', float('nan')):.6f} px: the clear is "
            f"reproducible, not one lucky draw. The map then reached "
            f"{len(new_positions)} positions it did not reach before "
            f"({', '.join(str(tuple(p)) for p in new_positions)}), {opened}."
        )
    else:
        clearance_sentence = (
            "No round finished a stage: every run that ended at all left `0x14` for `0x0B`, the death "
            "animation, and came back to the node it started from."
        )

    lines: List[str] = [
        "### 10.66 The Same Dashboard on Every Stage the Map Will Actually Give, and What Clears Mean",
        "",
        f"The hero is one stage, and one stage is what the repository had: 10.36.3 built a savestate "
        f"by walking the map to the first node past Yoshi's House, and every picture-based claim since "
        f"has been resting on it. Asked for more levels, the honest answer starts by asking the "
        f"engine. `scripts/survey_world_map.py` walks the map cursor from that node in all four "
        f"directions, {len(positions)} distinct positions deep, and presses A on each.",
        "",
        *render_survey_table(survey),
        "",
        f"{len(walkable)} of those positions hand over a session the player can walk in, and they "
        f"cover {len(distinct)} level numbers: {', '.join(f'`{t}`' for t in distinct)}. Two cursor "
        f"positions can be the same stage - the survey found `{survey['nodes'][0]['translevel']}` at "
        f"both ends of one step - which is why the stages below are keyed by the translevel byte and "
        f"not by the node.",
        "",
        "**Writing the cursor is not a teleport.** The first version of the survey parked the cursor "
        "by writing `$7E:1F17`/`$7E:1F19` and then pressed A, exactly as 10.36.3's `warp_on_map` does. "
        "Three different written positions entered the same level, `0x2A`, every time: the engine "
        "keeps its own node index and those two bytes are what it reports, not what it decides. "
        "10.36.3 never noticed because the position it wrote is the position its own walk had "
        "already parked the cursor on. The survey now replays a recorded direction sequence from the "
        "saved map state before every probe, and reads the cursor back to prove the probe was "
        "standing where it meant to.",
        "",
        "**How wide a stage is, read off the buffer.** Whether a run that stops has finished is a "
        "question about the level, and the level answers it: the buffer at `$7E:C800` stores one "
        "subscreen after another, so the last one holding anything but air is where the terrain ends. "
        f"Yoshi's Island 1 holds {yoshi_width / 256:.0f} subscreens - {yoshi_width:.0f} px - which "
        f"is {yoshi_width / 1900.0:.2f} times the distance 10.37 calls the goal. That number is what "
        "turned the clearance question around: the first campaign gave the controller "
        f"{prior_budget} frames per attempt and its furthest run covered {best_prior:.0f} px of a "
        f"{yoshi_width:.0f} px level, so the wall it hit was the frame budget, not the end of the "
        "stage - and the same controller, flown with room enough to finish, does finish it below. The "
        f"stages are {', '.join(f'{stage_width_px(s):.0f} px' for s in payload['stages'])} wide in the "
        "order they are listed, and the progress bar on every dashboard is now scaled to its own "
        "stage's terrain rather than to a goal distance nobody has seen the engine confirm.",
        "",
        "**What a clear is - three definitions in one day.** A stage that ends is a stage the engine "
        "leaves, so `scripts/unlock_world_map.py` flies the published clearance controller at the "
        "entered level and samples `$7E:0100` every frame instead of trusting a distance. The first "
        "definition this file used was 10.37's: `x >= 1900 px`. The second was this script's own "
        "first draft - reaching the map mode `0x0E` counts - and it reported every round as cleared, "
        "because a game over also ends on the map. The third is the byte: a death leaves `0x14` for "
        "`0x0B`, the death animation, and a run that reaches the map without passing through it has "
        "finished the stage. That is a discriminator the console supplies, not an interpretation the "
        "harness brings.",
        "",
        *render_clearance_table(progression),
        "",
        f"{cleared} of {len(progression['rounds'])} rounds cleared. " + clearance_sentence,
        "",
        "**One episode per stage, flown by the agent that can finish it.** "
        "`scripts/record_level_dashboards.py` records the same channels 10.60 records at each "
        "stage the map hands over - surveyed or unlocked - plus the two channels 10.68 added: "
        "the full 16-bit tile ids of the window and every live sprite slot. Yoshi's Island 1 is "
        "flown by the published clearance controller on its seed-7 draw until the engine itself "
        "ends the stage; the blank house session is walked back out through its door by the "
        "rule-based runner of 10.68; the other two sessions are documented by the distilled actor "
        "for a fixed horizon, because 10.68 measured no agent can finish them at all.",
        "",
        *render_stage_table(payload),
        "",
        "The verdict column is the engine's own: the mode byte leaving `0x14` for anything other "
        "than the death animation is a stage the engine ended without losing the player, and "
        "`0x0B` is a death. No distance proxy appears in that column, on purpose.",
        "",
    ]
    for index in sorted(rendered):
        record = rendered[index]
        caption = record["caption"]
        gif = _rel(record["gif"]["gif"])
        lines += [
            f"![The dashboard on {record['label']}: the console's own picture, the tile window that "
            f"stage's models read, and what the actor believed on every frame]({gif})",
            "",
            f"**Level {record['level_index']}: {record['label']}.** "
            f"{caption['episode_frames']} frames, "
            f"{caption['progress_px']:.4f} px of progress, the tile window's void class on "
            f"{_pct(caption['tile_void_cell_share'])} of its cells, "
            f"{record['gif']['bytes'] / 1e6:.2f} MB GIF / "
            f"{record['video']['bytes'] / 1e6:.2f} MB MP4: "
            f"`{_rel(record['video']['mp4'])}`.",
            "",
        ]
    return lines


def main() -> Dict[str, Any]:
    from src.utils.provenance import write_metrics

    payload = gather()
    hero = run_dashboard_render(
        context=hero_context(payload), metrics_command="make level-dashboard-study"
    )
    payload["hero"] = hero
    payload["rendered"] = render_level_videos(payload)
    write = results_file("level_dashboard_study_metrics.json")
    write_metrics(write, payload, command="make level-dashboard-study")
    logger.info(f"level dashboard study: {write}")
    return payload


if __name__ == "__main__":
    main()
