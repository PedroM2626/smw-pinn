r"""
test_cli_parity.py
The `smw-pinn` runner must reach every study the Makefile can run.

`src/cli.py` advertises itself as mirroring the Makefile, and for a while it did; then the
studies after 10.44 were added as Makefile targets and never as subcommands, so ten of them -
including the ones the README's own reproduction list tells you to run - were reachable only
through the generic passthrough while the sentence about mirroring stayed in print. This is the
check that would have caught it: parse the Makefile, take every target whose recipe is a single
`-m module`, and require a subcommand with the same name.

CPU-only, no artifacts, no emulator.

Run:  pytest tests/test_cli_parity.py -q
"""

import re
from pathlib import Path

from src.cli import CHECK_COMMANDS, MODULE_COMMANDS

REPO = Path(__file__).resolve().parents[1]
# Infrastructure targets, not studies: no module, or a multi-leg recipe whose legs are covered by
# the single-module names below them (projection-cell/effective-velocity/neural-ode pair a CPU
# study with the shared console runner, which is registered as `physics-injection-mpc`'s module).
NOT_A_STUDY = {
    "help",
    "install",
    "install-cuda",
    "run",
    "clean",
    "test",
    "test-cov",
    "lint",
    "format",
    "format-check",
    "typecheck",
    "check-all",
    "install-info",
    "smoke",
    "smoke-all",
    "record-jump",
    "record-sprint",
    "record-gameplay",
    "record-pixel",
    "record-multi-entity",
    "record-set-multi-entity",
    "record-tilemap",
}

TARGET = re.compile(r"^([a-z][a-z0-9_-]*):$", re.M)
MODULE = re.compile(r"-m ([a-zA-Z0-9_.]+)")


def _single_module_targets() -> dict[str, str]:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    blocks = TARGET.split(text)
    out: dict[str, str] = {}
    for name, body in zip(blocks[1::2], blocks[2::2]):
        modules = sorted({m for m in MODULE.findall(body.split("\n\n")[0])})
        if len(modules) == 1:
            out[name] = modules[0]
    return out


def test_every_single_module_make_target_has_a_subcommand() -> None:
    targets = _single_module_targets()
    assert len(targets) > 25, f"the Makefile parse found only {len(targets)} study targets"
    reachable = set(MODULE_COMMANDS) | set(CHECK_COMMANDS)
    missing = sorted(name for name in targets if name not in reachable and name not in NOT_A_STUDY)
    assert not missing, (
        "Makefile targets with no `smw-pinn` subcommand, while src/cli.py claims to mirror "
        f"them: {missing}"
    )


def test_a_subcommand_never_points_at_a_module_that_does_not_exist() -> None:
    """The reverse direction: a renamed study must not leave a dead command behind."""
    dead = []
    for name, module in MODULE_COMMANDS.items():
        relative = module.replace(".", "/")
        if (
            not (REPO / f"{relative}.py").exists()
            and not (REPO / relative / "__init__.py").exists()
        ):
            dead.append(f"{name} -> {module}")
    assert not dead, "smw-pinn subcommands pointing at modules that are gone: " + ", ".join(dead)


def test_the_three_two_leg_studies_map_to_their_forward_module() -> None:
    """A target that trains and then flies resolves to the leg that produces the artifact.

    `make projection-cell`, `make effective-velocity` and `make neural-ode` run the CPU study and
    then the shared console runner; a single subcommand can only name one of them, and naming the
    forward leg is the choice that keeps `smw-pinn <name>` equivalent to the emulator-free half.
    """
    for name, forward in (
        ("projection-cell", "src.evaluation.projection_cell_benchmark"),
        ("effective-velocity", "src.evaluation.effective_velocity_benchmark"),
        ("neural-ode", "src.evaluation.neural_ode_integrator_benchmark"),
        ("corrected-physics", "src.evaluation.corrected_physics_ablation"),
    ):
        assert MODULE_COMMANDS[name] == forward, f"{name} no longer names its forward leg"
