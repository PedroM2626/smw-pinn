r"""Gates for README 10.58: the generated section, and the claims its prose makes.

The section is emitted by `render_section` from `results/residue_process_metrics.json`, so the
tables and the numbers inside the sentences are the same objects - a re-run that moves the lattice
moves the prose. What a whole-table check cannot see is a quantifier, and this section's whole point
is a quantifier: *every* exception frame in *every* recording is accounted for by one of two
deterministic mechanisms. That claim is re-evaluated here over the artifact, per recording, and the
model-class conclusions (a diffusion mis-states the spread by a factor that grows with the horizon;
only a kernel that models run duration covers) are recomputed rather than remembered.

CPU-only, reads the artifact and the recordings' derived blocks; skipped when the study has not run.

Run:  pytest tests/test_residue_process_study.py -q
"""

import json
from pathlib import Path

import pytest

from src.evaluation.residue_process_study import (
    BAND,
    KERNELS,
    render_section,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "residue_process_metrics.json"


def _payload():
    if not ARTIFACT.is_file():
        pytest.skip("the 10.58 residue study has not been run")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _readme():
    return (REPO / "README.md").read_text(encoding="utf-8")


def test_readme_10_58_is_the_generated_block() -> None:
    """Not one number in the section is typed."""
    lines = render_section(_payload())
    text = _readme()
    missing = [line for line in lines if line and line not in text]
    assert not missing, "README 10.58 disagrees with the generator:\n" + "\n".join(missing[:4])
    assert lines[0].startswith("### 10.58 "), "the generator must start at the section heading"


def test_the_residue_is_a_lattice_not_a_density() -> None:
    """Finding 1: the residual never leaves the sub-pixel grid, in any recording."""
    lattices = [block["lattice"] for block in _payload()["per_recording"].values()]
    assert lattices
    assert max(b["max_off_integer_deviation_subpixels"] for b in lattices) == 0.0
    assert max(b["distinct_residue_values"] for b in lattices) <= 13
    assert min(b["rate_exactly_zero"] for b in lattices) > 0.90
    for block in lattices:
        assert block["largest_gap_between_atoms_subpixels"] >= 1.0, (
            "a lattice with no gap between atoms is a continuum, which is the claim being denied"
        )


def test_the_clamp_accounting_is_exhaustive_per_recording() -> None:
    """Finding 2: clamp plus one-pixel reposition leaves no exception frame anywhere."""
    clamps = _payload()["per_recording"]
    for name, block in clamps.items():
        clamp = block["clamp"]
        pinned_share = clamp["share_of_exceptions_pinned"]
        leftover = clamp["n_exceptions_left_after_the_clamp"]
        assert pinned_share is not None and pinned_share > 0.70, f"{name}: {pinned_share}"
        assert clamp["rate_residue_exactly_minus_velocity_on_pinned"] == pytest.approx(1.0), name
        assert clamp["rate_exception_when_pinned"] == pytest.approx(1.0), name
        assert clamp["rate_exception_when_not_pinned"] < 0.01, f"{name} off-clamp rate"
        assert (
            clamp["n_exceptions_pinned_in_episode"] + leftover == clamp["n_exceptions_in_episode"]
        ), f"{name}: the two mechanisms do not cover the in-episode exceptions"
        if leftover:
            assert clamp["leftover_rate_exactly_one_pixel_off_velocity"] > 0.999, (
                f"{name}: the leftovers are not a single pixel: {clamp['leftover_atoms']}"
            )
    verdict = _payload()["verdict"]
    assert verdict["the_clamp_explains_the_exceptions"] is True
    assert verdict["leftover_are_one_pixel_off_velocity"] is True


def test_an_iid_fit_is_only_good_for_one_frame() -> None:
    """Finding 3: the over-dispersion of the cumulative residue grows with the horizon."""
    per_recording = _payload()["per_recording"]
    for name, block in per_recording.items():
        horizons = block["calibration"].get("horizons", {})
        ordered = sorted(horizons, key=lambda key: int(key[1:]))
        if len(ordered) < 3:
            continue
        ratios = [horizons[key]["over_dispersion_vs_iid"] for key in ordered]
        assert ratios[0] < 1.5, f"{name}: one frame should be about iid by construction"
        assert ratios[-1] > ratios[len(ratios) // 2], f"{name}: the spread does not grow super-iid"
        assert ratios[-1] > 4.0, f"{name}: longest-horizon over-dispersion is {ratios[-1]:.2f}x"
    assert min(_payload()["summary"]["over_dispersion_h60_range"]) > 4.0


def test_only_a_kernel_with_run_duration_covers_at_length() -> None:
    """Finding 4: at the longest horizon the renewal kernel covers and the other three do not."""
    per_recording = _payload()["per_recording"]
    longest = {}
    for name, block in per_recording.items():
        horizons = block["calibration"].get("horizons", {})
        if not horizons:
            continue
        key = max(horizons, key=lambda item: int(item[1:]))
        longest[name] = horizons[key]["kernels"]
    assert len(longest) >= 5, "the calibration must cover most recordings for this claim"
    assert all(kernels["run_length_renewal"]["coverage_90"] > 0.85 for kernels in longest.values())
    for name, kernels in longest.items():
        for weaker in ("gaussian_iid", "jump_iid", "markov_geometric"):
            assert kernels[weaker]["coverage_90"] < kernels["run_length_renewal"]["coverage_90"], (
                f"{name}: {weaker} is not worse than the renewal kernel at {name}'s longest horizon"
            )
    means = _payload()["summary"]["coverage_mean_90"]
    assert set(means) == set(KERNELS)
    winners = _payload()["summary"]["proper_score_winner_by_recording"]
    assert sum(1 for value in winners.values() if value == "run_length_renewal") >= 4, winners
    assert BAND == 0.90


def test_the_state_has_no_channel_for_the_boundary_it_clamps_against() -> None:
    """The section's actionable claim is that the missing variable is not in the state vector."""
    from src.environment.wram import ADDR_PLAYER_X, ADDR_PLAYER_X_SUB

    assert ADDR_PLAYER_X != ADDR_PLAYER_X_SUB, (
        "the position and its sub-pixel accumulator are separate addresses; the study's claim that "
        "the ±1 px leftovers cannot be accumulator rounding depends on that"
    )
    source = (REPO / "src" / "environment" / "wram.py").read_text(encoding="utf-8")
    for channel in ("ADDR_PLAYER_X", "ADDR_VX", "ADDR_COLLISION"):
        assert channel in source, f"the WRAM map lost {channel}"
    assert "camera" not in source.lower(), (
        "the section says no channel of the eight exposes the camera or the level bound; if the map "
        "grows one, the prediction it makes about the residue has to be re-measured"
    )


def test_section_12_quotes_the_counts_it_reinterprets() -> None:
    """The Section 12 bullet that withdraws "irreducible" carries four artifact numbers."""
    payload = _payload()
    clamps = [block["clamp"] for block in payload["per_recording"].values()]
    shares = [c["share_of_exceptions_pinned"] for c in clamps]
    text = _readme()
    assert f"{100 * min(shares):.1f}-{100 * max(shares):.1f}%" in text, (
        f"section 12 no longer quotes the clamp share {100 * min(shares):.1f}-"
        f"{100 * max(shares):.1f}%"
    )
    leftover = sum(c["n_exceptions_left_after_the_clamp"] for c in clamps)
    assert str(leftover) in text
    assert f"{payload['summary']['transitions']:,}" in text
    assert "0.1056" in text and "irreducible" in text, (
        "the reinterpretation must name the figure and the word it withdraws"
    )


def test_the_section_states_what_it_did_not_measure() -> None:
    """A section that reinterprets a published constant has to say how far the reading reaches."""
    section = _readme()
    start = section.index("### 10.58 ")
    body = section[start : section.index("## 11. ", start)]
    for phrase in (
        "inferred from the recorded position not moving",
        "The vertical identity is not scored here",
        "linear ridge fits",
        "recorded trajectories under the actions the player actually took",
    ):
        assert phrase in body, f"10.58's limitations dropped {phrase!r}"
    assert "0.1056" in body, "the section must name the figure it reinterprets"
