# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research artifact, "changed" includes *self-corrections of published
numbers* - those are reported here and in README Section 12, never applied silently.

## [Unreleased]

### Added

- **PIML-MFRL** (README Section 10.39): model-free PPO on the authentic SNES console with
  three independently switchable physics couplings - a Control-Lyapunov/HJB critic
  penalty (Approach A), a differentiable CBF-QP actor safety layer with a discrete
  categorical filter (Approach B), and a physics-violation penalty on the PPO surrogate
  (Approach C). New modules: `src/losses/physics_rl_losses.py`,
  `src/models/cbf_projection.py`, `src/training/piml_mfrl.py`; config
  `configs/piml_mfrl.yaml`; a **per-mechanism ablation study**
  (`src/evaluation/piml_mfrl_study.py`, README 10.39.1) that runs the model-free baseline
  plus each coupling in isolation (A / B / C) and combined (A+B+C) over 3 seeds at a
  10,000-frame budget, and emulator-free coverage in `tests/test_piml_mfrl.py`. The
  ablation reports an honest, localised null on Yoshi's Island 1: every condition sits
  within +/-2.2% of the baseline (far inside the ~+/-280 seed std) and the executed-action
  violation is ~0 everywhere, so B/C are correctly inert on open ground - Approach C
  reproduces the baseline return exactly on all three seeds because its only gradient term
  is lambda * violation; the one clear effect is the monotonic ~1.5x training-time cost of
  the couplings. Recorded in `results/piml_mfrl_metrics.json` with figure
  `results/figures/piml_mfrl_comparison.png`.
- CI: a native `windows-latest` job (path/CWD/subprocess parity) and a Linux
  Python `3.10 / 3.11 / 3.12` test matrix (`.github/workflows/ci.yml`).
- A `print()`-in-`src/` convention guard (`tests/test_no_print_in_src.py`) with a
  documented allowlist for the two legitimate console-stdout modules.
- Community files: `SECURITY.md`, `CODE_OF_CONDUCT.md`, this `CHANGELOG.md`,
  `.env.example`, an issue-template `config.yml` and a feature-request template, and a
  third-party notice for the bundled Libretro core (`src/environment/bin/README.md`).
- `requirements.lock`: a pinned snapshot of the validated reference environment.

### Changed

- mypy typed core expanded from 19 to 40 modules: the whole reusable library is now
  checked (`src/models`, `src/losses`, `src/planning`, `src/perception`, `src/utils`, plus
  the environment data/vectorised-sim layer, the trainer and the per-variable/rollout
  evaluators). Passed as directories in `src/cli.py`; the ctypes emulator wrapper and the
  result-producing benchmark/evaluation CLI scripts remain outside the strict set by design.
- CI: the `windows-latest` job now runs the same substantive gates as Linux
  (`lint`, `typecheck`, `test-cov` with the coverage floor), via the console script since
  `make` is absent on Windows; only `format-check` stays Linux-only because the Windows
  runner checks text files out with CRLF (`.github/workflows/ci.yml`).
- Zero-shot cross-level control benchmark re-recorded as a clean **5-seed** protocol
  (`src/evaluation/evaluate_cross_level_control.py`): every controller is now mean +/- std
  with `_meta` provenance, replacing the single unseeded draw that the `STALE` marker had
  flagged (README 10.28). This cleared the last `STALE` row in `results/MANIFEST.md`.

### Fixed

- The zero-shot cross-level control artifact (`cross_level_control_metrics.json`) was a
  single irreproducible draw: the CEM planners and the random baseline drew from an
  unseeded RNG. It is now reseeded per seed and aggregated over 5 seeds; the deterministic
  DAgger policy reproduces exactly (+/-0.00), and the MPC rows report their seed variance.
- `src/models/cbf_projection.py`: `DiscreteCBFCategoricalFilter.action_table` is set via
  `nn.Module.register_buffer`, so mypy resolved reads of it through `__getattr__` and typed
  it `Tensor | Module`, failing `physics_action_violation_table` once the module entered the
  expanded typed core (CI typecheck, exit 1/2). Added the class-level `action_table:
  torch.Tensor` annotation (the project's documented buffer-typing convention); verified
  with a fresh, cacheless `mypy --no-incremental` over all 40 core modules.
- CI run #14 (`Typecheck (mypy via Makefile)`, exit code 2): the `dev`/`all` extras
  declared `mypy>=1.0.0` with no upper bound, so CI resolved a newer interpreter that
  crashed the typecheck gate while the locally validated version passed. Pinned mypy to
  the validated band `>=2.3.1,<2.4` in `pyproject.toml` and `requirements.txt` (parity
  preserved), mirroring the existing ruff pin so CI, pre-commit and local use a checker
  version that makes `make typecheck` green.

## [0.1.0] - 2026-09-22

Paper-reproduction release: the four-architecture MLP/LSTM/Soft-PINN/Hard-PINN benchmark,
the sample-efficiency and multi-seed studies, and the closed-loop MBRL line
(MPC, Dyna-PPO, model-free PPO, online/safe MBPO, deep ensembles, multi-entity and
tilemap world models, DAgger, distillation, cross-stage generalization).

### Fixed (audited self-corrections, README Section 12)

- The Section 10.6 MPC numbers were re-recorded after the episode-preamble probe showed
  the committed savestate restores into engine mode `0x08`, not interactive `0x14`
  (the preamble alone was worth a 3.4x progress difference). The gameplay-mode poke was
  unified into `SnesLibretroEmulator.start_episode()`.
- The Dyna-PPO row of the Section 10.27 master table was corrected: it had been pasted
  from the MPC row (164.75 px) instead of the recorded 115.00 px.
- Jump-impulse non-identifiability from single transitions is reported as a negative
  result (Section 10.37.1) rather than fitted away; Yoshi's Island 2 capture is reported
  as blocked with its diagnostics artifact (Section 10.36), not a fabricated state.

### Added

- `results/MANIFEST.md` artifact index with CI-enforced provenance (`_meta`), writer,
  command, README-section mapping and checkpoint-freshness gates.

[Unreleased]: https://github.com/PedroM2626/smw-pinn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PedroM2626/smw-pinn/releases/tag/v0.1.0
