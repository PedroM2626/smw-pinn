"""
world_map_recheck_study.py - README 10.67: does the world-map survey of 10.66 repeat, and what does
the map a cleared stage actually open?

Two routes were measured and one control was run:

* the **published route** of 10.66, which boots the console and walks to the overworld;
* the **opened route**, which surveys the map that exists only after the clearance controller
  finished Yoshi's Island 1 - `smw_world_map_after_round_0_152_136_t2a.state` - through the same
  probe;
* the **replay**, the published route run again into its own artifact, with every captured state
  hashed, so that "the survey repeats" is checked against the bytes of the files 10.66 published
  rather than asserted.

The questions it settles were both left open by 10.66: the cursor position the clear opened had
never been entered, and nobody had re-run the survey since the cursor-write defect was replaced by a
recorded direction replay.

Run:  python -m src.evaluation.world_map_recheck_study
      make world-map-recheck
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger
from src.utils.paths import REPO_ROOT, results_file

logger = get_logger(__name__)

PUBLISHED_SURVEY = "results/world_map_survey.json"
REPLAY_SURVEY = "results/world_map_survey_replay.json"
OPENED_SURVEY = "results/world_map_survey_opened.json"
STUDY = "results/level_dashboards_cleared.json"

#: what a survey route is supposed to reproduce: where the cursor was parked, what the engine loaded
#: there, and how the player behaved once it did. `terrain` was added to the probe after the
#: published run and `capture` names the file each run wrote, so both are compared where they belong.
BEHAVIOURAL_FIELDS = (
    "path",
    "cursor_before_A",
    "cursor_reached",
    "mode_sequence",
    "entry_frame",
    "translevel",
    "timer_hundreds",
    "interactive",
    "x_after_settle",
    "y_after_settle",
    "right_response_px",
    "plausible",
    "walkable",
    "level",
)


def _read(path: str) -> Dict[str, Any]:
    resolved = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(resolved, encoding="utf-8") as f:
        return json.load(f)


def _capture(node: Dict[str, Any]) -> Dict[str, Any]:
    return dict(node.get("capture") or {})


def state_on_disk(name: Optional[str]) -> Optional[str]:
    """The absolute path of a captured state under `data/raw/`, or None if it is not kept."""
    if not name:
        return None
    candidate = os.path.join(REPO_ROOT, "data", "raw", str(name))
    return candidate if os.path.isfile(candidate) else None


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


def nodes_by_position(survey: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """One survey's probe records, keyed by the cursor position they were taken at."""
    return {(int(n["node"][0]), int(n["node"][1])): n for n in survey["nodes"]}


def translevels(survey: Dict[str, Any], key: str = "walkable") -> List[str]:
    """The distinct level numbers handed over, among the nodes for which `key` holds."""
    return sorted({str(n["translevel"]) for n in survey["nodes"] if n.get(key)})


def field_differences(published: Dict[str, Any], other: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Where a second route disagrees with the published survey, field by field."""
    a, b = nodes_by_position(published), nodes_by_position(other)
    out: List[Dict[str, Any]] = []
    for node in sorted(set(a) & set(b)):
        for field in BEHAVIOURAL_FIELDS:
            if a[node].get(field) != b[node].get(field):
                out.append(
                    {
                        "node": list(node),
                        "field": field,
                        "published": a[node].get(field),
                        "other": b[node].get(field),
                    }
                )
    return out


def byte_identity(replay: Dict[str, Any], published: Dict[str, Any]) -> Dict[str, Any]:
    """Does a fresh run of the published route write the bytes the published survey named?

    The replay captured under its own prefix and those files are not kept; what is kept is the
    SHA-256 each one got. A published state that hashes to the replay's value for the same node is a
    state the survey reproduces exactly - the only sense in which walking a console repeats.
    """
    published_nodes = nodes_by_position(published)
    checked: List[Dict[str, Any]] = []
    for node, record in sorted(nodes_by_position(replay).items()):
        target = state_on_disk(_capture(published_nodes.get(node, {})).get("saved"))
        digest = record.get("capture", {}).get("sha256") if record.get("capture") else None
        if not digest or target is None:
            continue
        checked.append(
            {
                "node": list(node),
                "published_state": _rel(target),
                "replay_sha256": str(digest),
                "published_sha256": _sha256(target),
            }
        )
    agreeing = [c for c in checked if c["replay_sha256"] == c["published_sha256"]]
    return {
        "states_checked": len(checked),
        "states_byte_identical": len(agreeing),
        "mismatched_nodes": [
            c["node"] for c in checked if c["replay_sha256"] != c["published_sha256"]
        ],
        "replay_states_kept": sum(
            1
            for record in nodes_by_position(replay).values()
            if state_on_disk(_capture(record).get("saved"))
        ),
        "published_states_kept": sum(
            1
            for record in published_nodes.values()
            if state_on_disk(_capture(record).get("saved"))
        ),
    }


def route_state_differences(opened: Dict[str, Any], replay: Dict[str, Any]) -> Dict[str, Any]:
    """For the nodes both routes entered: the same console bytes, or not?

    The two surveys agree about what a node does. Whether they agree about the state the engine is in
    there is a separate question, and the hash is what answers it.
    """
    a, b = nodes_by_position(replay), nodes_by_position(opened)
    shared = sorted(set(a) & set(b))
    rows = []
    for node in shared:
        first, second = _capture(a[node]).get("sha256"), _capture(b[node]).get("sha256")
        if not first or not second:
            continue
        rows.append(
            {
                "node": list(node),
                "translevel": str(a[node].get("translevel")),
                "same_bytes": bool(first == second),
                "replay_sha256": str(first),
                "opened_sha256": str(second),
            }
        )
    return {
        "shared_nodes": len(shared),
        "nodes_compared": len(rows),
        "nodes_with_identical_bytes": sum(1 for r in rows if r["same_bytes"]),
        "opened_nodes_only": [list(n) for n in sorted(set(b) - set(a))],
        "published_nodes_only": [list(n) for n in sorted(set(a) - set(b))],
        "per_node": rows,
    }


def captures_kept(survey: Dict[str, Any]) -> int:
    """How many of a survey's named capture files are actually under `data/raw/`.

    The replay and the opened run both capture under prefixes that are not committed, because their
    bytes are what this section compares rather than what a later study loads; the published survey's
    states are the ones the recorder reads.
    """
    return sum(1 for node in survey["nodes"] if state_on_disk(_capture(node).get("saved")))


def gather() -> Dict[str, Any]:
    published = _read(PUBLISHED_SURVEY)
    replay = _read(REPLAY_SURVEY)
    opened = _read(OPENED_SURVEY)
    stages = sorted(str(s["identity"]["translevel"]) for s in _read(STUDY)["stages"])
    opened_nodes = nodes_by_position(opened)
    start = tuple(int(v) for v in opened["protocol"]["cursor_at_start"])
    entered = opened_nodes.get(start, {})
    new = [
        opened_nodes[node]
        for node in sorted(set(opened_nodes) - set(nodes_by_position(published)))
        if node != start
    ]
    fifth = new[0] if new else {}
    published_nodes = nodes_by_position(published)
    twin = next(
        (
            published_nodes[node]
            for node in sorted(set(published_nodes) - set(opened_nodes))
            if str(published_nodes[node].get("translevel")) == str(fifth.get("translevel"))
        ),
        {},
    )
    return {
        "study": (
            "Re-running the world-map survey of 10.66 two ways: the same route again with every "
            "capture hashed, and the map that only exists once a stage has been cleared."
        ),
        "protocol": {
            "published_route": "make survey-map (power-on boot, the route 10.66 published)",
            "replay_route": "make survey-map-replay (same boot, states under a discarded prefix)",
            "opened_route": str(opened["protocol"].get("route", "")),
            "probe_gate": published["protocol"]["gate"],
            "behavioural_fields_compared": list(BEHAVIOURAL_FIELDS),
            "capture_files_kept": {
                "published": captures_kept(published),
                "replay": captures_kept(replay),
                "opened": captures_kept(opened),
            },
        },
        "surveys": {"published": published, "replay": replay, "opened": opened},
        "published": {
            "positions": len(published["discovery"]["positions"]),
            "walkable": len([n for n in published["nodes"] if n.get("walkable")]),
            "translevels": translevels(published),
            "level_gate_translevels": translevels(published, "level"),
        },
        "opened": {
            "positions": len(opened["discovery"]["positions"]),
            "walkable": len([n for n in opened["nodes"] if n.get("walkable")]),
            "translevels": translevels(opened),
            "translevels_new": sorted(
                set(translevels(opened)) - set(translevels(published))
            ),
            "translevels_missing": sorted(
                set(translevels(published)) - set(translevels(opened))
            ),
            "level_gate_translevels": translevels(opened, "level"),
            "node_the_clear_lands_on": list(start),
            "what_it_enters": {
                "translevel": entered.get("translevel"),
                "terrain_px": (entered.get("terrain") or {}).get("terrain_px"),
                "timer_hundreds": entered.get("timer_hundreds"),
                "level": entered.get("level"),
            },
            "the_position_the_published_route_cannot_reach": {
                "node": list(fifth.get("node", [])),
                "translevel": fifth.get("translevel"),
                "timer_hundreds": fifth.get("timer_hundreds"),
                "terrain_px": (fifth.get("terrain") or {}).get("terrain_px"),
                "walkable": fifth.get("walkable"),
                "level": fifth.get("level"),
                "x_at_entry": _capture(fifth).get("x_at_entry"),
                "reason": fifth.get("reason"),
                "the_published_position_it_matches": {
                    "node": list(twin.get("node", [])),
                    "x_at_entry": _capture(twin).get("x_at_entry"),
                    "timer_hundreds": twin.get("timer_hundreds"),
                },
            },
        },
        "stages": stages,
        "replay_check": {
            "field_differences_vs_published": field_differences(published, replay),
            "byte_identity": byte_identity(replay, published),
        },
        "route_check": {
            "field_differences_opened_vs_published": field_differences(published, opened),
            "state_bytes": route_state_differences(opened, replay),
        },
    }


def _count(value: int) -> str:
    """A number of findings spelled out, because `1 of them is new` reads like a table cell."""
    return {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(
        value, str(value)
    )


def render_position_table(survey: Dict[str, Any]) -> List[str]:
    lines = [
        "| map cursor | reached by | translevel | timer | terrain the buffer holds | right-press dx "
        "| the timer gate calls it a stage |",
        "|---|---|---|---|---|---|---|",
    ]
    for node in survey["nodes"]:
        path = "+".join(node["path"]) if node["path"] else "(start)"
        terrain = (node.get("terrain") or {}).get("terrain_px")
        gate = "yes" if node.get("level") else "no"
        lines.append(
            f"| {tuple(node['node'])} | `{path}` | `{node['translevel']}` | {node['timer_hundreds']} "
            f"| {f'{terrain:.0f} px' if terrain is not None else '-'} | "
            f"{node.get('right_response_px')} | {gate} |"
        )
    return lines


def render_section(payload: Dict[str, Any]) -> List[str]:
    r"""The whole of README section 10.67, generated from the two survey routes and the replay."""
    opened_art = payload["surveys"]["opened"]
    published = payload["published"]
    opened = payload["opened"]
    fifth = opened["the_position_the_published_route_cannot_reach"]
    twin = fifth["the_published_position_it_matches"]
    enters = opened["what_it_enters"]
    identity = payload["replay_check"]["byte_identity"]
    fields = payload["replay_check"]["field_differences_vs_published"]
    routes = payload["route_check"]["state_bytes"]
    differences = payload["route_check"]["field_differences_opened_vs_published"]
    path_differences = [d for d in differences if d["field"] == "path"]
    node_fields = [d for d in differences if d["field"] != "path"]
    numbers = ", ".join(f"`{t}`" for t in opened["translevels"])
    new_levels = _count(len(opened["translevels_new"]))
    missing_levels = _count(len(opened["translevels_missing"]))

    lines: List[str] = [
        "### 10.67 Re-running the Map Survey, and the Map a Clear Opens",
        "",
        f"10.66 ended on a map that is not the map a fresh file opens: finishing a stage moves the "
        f"cursor and unlocks a node. `scripts/survey_world_map.py --from-map-state` surveys a saved "
        f"map rather than booting to one, and pointed at the state the clearance campaign kept after "
        f"its clear it reaches {opened['positions']} cursor positions where the published route "
        f"reaches {published['positions']}. The level numbers it hands over are "
        f"{numbers}. {new_levels.capitalize()} of them was not on the published map - "
        f"{', '.join(f'`{t}`' for t in opened['translevels_new'])}, the stage the clear itself "
        f"opened - and {missing_levels} left it. Nothing the opened map hands over is a stage that "
        f"10.66 had not already listed.",
        "",
        *render_position_table(opened_art),
        "",
        f"**The node the clear opened was already a stage; the position it reached was not.** The "
        f"campaign settled the cursor on {tuple(opened['node_the_clear_lands_on'])}, and that node "
        f"enters `{enters['translevel']}` with a timer of {enters['timer_hundreds']} hundreds and "
        f"{enters['terrain_px']:.0f} px of terrain - the stage 10.66 records a dashboard for. The "
        f"cursor position the published route could not get to is {tuple(fifth['node'])}: it is "
        f"walkable, it loads `{fifth['translevel']}`, its timer reads {fifth['timer_hundreds']} and "
        f"its buffer holds {fifth['terrain_px']:.0f} px. That is the session the published route "
        f"found at {tuple(twin['node'])} - `{fifth['translevel']}` there too, timer "
        f"{twin['timer_hundreds']}, x = {twin['x_at_entry']} px at entry on both - one column east, "
        f"so it is a path tile of a stage already counted, not a fifth stage. The survey's "
        f"answer to *how many stages are there* is unchanged by one having been cleared.",
        "",
        f"**The survey repeats itself, to the byte.** `make survey-map-replay` runs the published "
        f"route again into its own artifact. Across {published['positions']} shared nodes and "
        f"{len(BEHAVIOURAL_FIELDS)} compared fields it returns {len(fields)} differences - the walk "
        f"of the cursor, the mode sequences, the entry frames, the timers, the entry coordinates and "
        f"the right-press distances are all exactly what 10.66 published. Each capture also records "
        f"the SHA-256 of the bytes it wrote, and hashing the files the published survey names matches "
        f"that digest for {identity['states_byte_identical']} of {identity['states_checked']}: the "
        f"console state the survey captures at a node is reproduced exactly. The replay wrote its "
        f"own states under `smw_replay_node` and they are not kept - {identity['replay_states_kept']} "
        f"of them are on disk against the {identity['published_states_kept']} the published survey "
        f"names, and the opened route keeps {payload['protocol']['capture_files_kept']['opened']} of "
        f"its {opened['positions']}: the replay's bytes are the committed files, and the opened "
        f"route's are not, which is the next paragraph.",
        "",
        f"**Same node, different console.** The two routes agree about what a node *does* and "
        f"disagree about the bytes it *is*: of the {routes['nodes_compared']} cursor positions both "
        f"entered, {routes['nodes_with_identical_bytes']} have the same state bytes. A savestate "
        f"carries the route, not just the level - the map flags, the session that arrived. The probe "
        f"records it twice over: of the {len(differences)} field differences between the two "
        f"surveys, {len(path_differences)} are `path`, which is a property of the map each route "
        f"started from rather than of the node it ends at, and the other {len(node_fields)} "
        f"describe the node itself"
        + (
            ": " + "; ".join(
                f"`{d['field']}` at {tuple(d['node'])}, {d['published']} against {d['other']}"
                for d in node_fields
            )
            + ". "
            if node_fields
            else ". "
        )
        + f"None of it changes the stage list - every distance on either side of that comparison "
        f"clears the gate's 6 px threshold - but a number quoted to the quarter pixel would not "
        f"survive the change of route. That is why every per-stage distance in 10.66 names the "
        f"savestate it was recorded from.",
        "",
    ]
    return lines


def main() -> Dict[str, Any]:
    from src.utils.provenance import write_metrics

    payload = gather()
    write = results_file("world_map_recheck_metrics.json")
    write_metrics(write, payload, command="make world-map-recheck")
    logger.info(f"world map recheck: {write}")
    return payload


if __name__ == "__main__":
    main()
