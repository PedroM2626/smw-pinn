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
import math
import os
from typing import Any, Dict, List

from src.evaluation.dashboard_video import (
    GOAL_PX,
    level_label,
    run_dashboard_render,
)
from src.utils.logging import get_logger
from src.utils.paths import REPO_ROOT, dataset_path, figure_file, results_file

logger = get_logger(__name__)

SURVEY_ARTIFACT = "results/world_map_survey.json"
PROGRESSION_ARTIFACT = "results/world_map_progression.json"
STUDY_ARTIFACT = "results/level_dashboards.json"
RUN_ARTIFACT = "results/dashboard_run_metrics.json"


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


def furthest_measured_px(stage: Dict[str, Any]) -> float:
    """The largest forward progress any arm reached in this stage, within this study.

    Not the level's length - nothing here has seen the end of a level. It is the only denominator
    that exists, and the picture names it as what it is.
    """
    return max(
        float(stage["walk_right_frontier"]["frontier_px"]),
        float(stage["mpc"]["best_progress_px"] or 0.0),
    )


def bottom_line(stage: Dict[str, Any]) -> str:
    progress = float(stage["episode"]["progress_px"])
    best = furthest_measured_px(stage)
    return f"this arm {progress:.0f} px   furthest measured here {best:.0f} px"


def render_contexts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One dashboard context per stage: the title bar, and the line the mock puts a rate in."""
    stages = payload["stages"]
    count = len(stages)
    contexts: List[Dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        # A bar scaled to 1900 px - the goal distance published for Yoshi's Island 1 - would show a
        # stage that has never been measured past 413 px as a stub. The denominator is the stage's
        # own, rounded to whole subscreens so the tick marks still say something.
        span = math.ceil(furthest_measured_px(stage) / 256.0) * 256.0
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


def hero_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The README hero's context: the published episode's stage, drawn as one of the set.

    The hero stays pointed at the recording 10.60 published - the arm probe, the pause table and the
    residue claim are all chained to that file - but the picture now says which stage it is and how
    far the arm on screen got next to the furthest anything measured has got there. The progress on
    that line is the hero recording's own, not the stage's: the two episodes of Yoshi's Island 1 are
    started from different captures of the same entry, and a line that quoted the wrong one would be
    the mistake this file keeps having to unmake.
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
    out = dict(yoshi)
    out["level_index"] = 1
    out["level_count"] = max(1, len(contexts))
    out["goal_px"] = GOAL_PX
    if stage is not None:
        best = furthest_measured_px(stage)
        out["clear_line"] = (
            f"this arm {hero_progress:.0f} px   furthest measured here {best:.0f} px"
        )
    return out


def render_level_videos(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Render one MP4, GIF and still per stage, and keep the numbers each one drew."""
    rendered: List[Dict[str, Any]] = []
    for context in render_contexts(payload):
        dataset = dataset_path(os.path.basename(context["dataset"]))
        stem = context["output_stem"]
        art = run_dashboard_render(
            dataset_path=dataset,
            output_mp4=figure_file(f"{stem}.mp4"),
            output_gif=figure_file(f"{stem}.gif"),
            output_still=figure_file(f"{stem}_still.png"),
            output_metrics=None,
            context={
                k: v
                for k, v in context.items()
                if k not in ("output_stem", "dataset", "progress_px")
            },
        )
        rendered.append(
            {
                "level_index": context["level_index"],
                "label": context["level_label"],
                "dataset": _rel(dataset),
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


def render_progression_table(progression: Dict[str, Any]) -> List[str]:
    lines = [
        "| round | node | best attempt | px | mode the attempt ended on | the map, if it reached it |",
        "|---|---|---|---|---|---|",
    ]
    for round_record in progression["rounds"]:
        attempts = round_record.get("attempts") or []
        best = max(attempts, key=lambda a: a["max_progress_px"]) if attempts else {}
        cursor = best.get("map_cursor")
        lines.append(
            f"| {round_record['round']} | {tuple(round_record.get('node') or [])} | "
            f"{best.get('attempt', '-')} | {best.get('max_progress_px', float('nan')):.4f} | "
            f"`{best.get('final_mode', '-')}` | "
            f"{('`' + str(tuple(cursor)) + '`') if cursor else 'not reached'} |"
        )
    return lines


def render_stage_table(payload: Dict[str, Any]) -> List[str]:
    lines = [
        "| stage | translevel | rightward walk reaches | the actor's episode | MPC best | "
        "furthest measured | episode repeats |",
        "|---|---|---|---|---|---|---|",
    ]
    for stage in payload["stages"]:
        frontier = stage["walk_right_frontier"]
        episode = stage["episode"]
        mpc = float(stage["mpc"]["best_progress_px"] or 0.0)
        lines.append(
            f"| `{stage_name(stage['identity'], stage['map_node'])}` | "
            f"`{stage['identity']['translevel']}` | {frontier['frontier_px']:.4f} px "
            f"({frontier['ended']}) | {episode['progress_px']:.4f} px ({episode['ended']}) | "
            f"{mpc:.4f} px | {furthest_measured_px(stage):.4f} px | "
            f"{episode['repeats_the_measured_attempt_px']:.6f} px |"
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
    best_round = max(
        (
            max((a["max_progress_px"] for a in r.get("attempts") or []), default=float("nan"))
            for r in progression["rounds"]
        ),
        default=float("nan"),
    )
    new_positions = progression.get("map_after", {}).get("new_positions", [])
    rendered = {int(r["level_index"]): r for r in payload.get("rendered", [])}

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
        "**Nothing clears.** A stage that ends is a stage the engine leaves, so "
        "`scripts/unlock_world_map.py` flies the published clearance controller at the entered level "
        "and samples `$7E:0100` every frame. ",
        "",
        *render_progression_table(progression),
        "",
        f"{cleared} rounds cleared, and the map reaches {len(new_positions)} positions it did not "
        f"reach before. Every run that ended at all ended the same way: the mode leaves `0x14` for "
        f"`0x0B`, "
        f"walks `0x0C` and `0x0D`, and reaches the map at **the node it started from**. That is a game "
        f"over, not a clear - and the definition matters, because the first version of this script "
        f"called any return to the map a clear and scored this same footage as a stage completed. The "
        f"cursor byte is what says otherwise. The "
        f"furthest attempt reached `{best_round:.4f}` px of forward progress with the engine still in "
        f"interactive mode when the frame budget ran out, which is `{best_round / 1900.0:.2f}` times "
        f"the distance the published clearance benchmark calls the goal. `goal_reached` in 10.37 is a "
        f"distance test, and the engine does not corroborate it. Two runs of this controller do not "
        f"agree on how far it gets - its cross-entropy draws are not seeded - so the stage table "
        f"below reports the best of its own three attempts, and the number above is the best of the "
        f"four this script took.",
        "",
        "**One episode per stage.** `scripts/record_level_dashboards.py` records the same channels "
        "10.60 records at each stage the survey captured, and measures the distance a held RIGHT+Y "
        "walk reaches there - the only length this repository has evidence for, since no arm has "
        "seen the end of a level.",
        "",
        *render_stage_table(payload),
        "",
        "The last column is the recorder's own control: the episode that gets its frames kept is a "
        "re-run of the attempt that was selected for its progress without keeping anything, and the "
        "two must agree to the subpixel because the actor and the console are both deterministic. "
        "Where they would not, one of those claims would be false.",
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
    hero = run_dashboard_render(context=hero_context(payload))
    payload["hero"] = hero
    payload["rendered"] = render_level_videos(payload)
    write = results_file("level_dashboard_study_metrics.json")
    write_metrics(write, payload, command="make level-dashboard-study")
    logger.info(f"level dashboard study: {write}")
    return payload


if __name__ == "__main__":
    main()
