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
  `configs/piml_mfrl.yaml`; a multi-seed hardware comparison
  (`src/evaluation/piml_mfrl_study.py`, README 10.39.1) and emulator-free coverage in
  `tests/test_piml_mfrl.py`. The study reports an honest null result on Yoshi's Island 1
  (PIML-MFRL statistically indistinguishable from model-free PPO within the +/-300 seed
  standard deviation, with an executed-action violation rate ~0 for both) recorded in
  `results/piml_mfrl_metrics.json`.
- CI: a native `windows-latest` job (path/CWD/subprocess parity) and a Linux
  Python `3.10 / 3.11 / 3.12` test matrix (`.github/workflows/ci.yml`).
- A `print()`-in-`src/` convention guard (`tests/test_no_print_in_src.py`) with a
  documented allowlist for the two legitimate console-stdout modules.
- Community files: `SECURITY.md`, `CODE_OF_CONDUCT.md`, this `CHANGELOG.md`,
  `.env.example`, an issue-template `config.yml` and a feature-request template, and a
  third-party notice for the bundled Libretro core (`src/environment/bin/README.md`).
- `requirements.lock`: a pinned snapshot of the validated reference environment.

### Changed

- mypy typed-core list extended to the new physics RL modules (`src/cli.py`).

### Fixed

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
