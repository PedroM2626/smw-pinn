# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research artifact, "changed" includes *self-corrections of published
numbers* - those are reported here and in README Section 12, never applied silently.

## [Unreleased]

### Added

- **DeepONet neural-operator baseline** (README Section 10.41): the repository's first
  neural-operator contribution, adding the operator-learning family (Lu et al., 2021) to
  the statistical-vs-physics-informed taxonomy. `src/models/deeponet.py`
  (`DeepONetDynamics`: branch MLP encodes the 14-sensor state-action reading into a
  p=64 basis; trunk MLP evaluates the basis at output-channel query coordinates in
  [-1, 1]; per-channel constant term; arbitrary-coordinate querying supported) trained
  under the exact unified protocol by `src/evaluation/deeponet_benchmark.py`
  (emulator-free, CI-safe; published comparison rows are read from the committed
  `benchmark_metrics.json`), writing `results/deeponet_benchmark_metrics.json` with
  `_meta` provenance and indexed in `results/MANIFEST.md`. Empirical outcome on the
  canonical split (seed 42): test MSE 7.8800 - 2.1x lower than the statistical MLP and
  the best physics-free architecture in the study - yet a kinematic residual of 273.15
  with a 99.58% multi-start violation rate (0% velocity violations), confirming at
  operator level that structure-free learning does not recover the discrete integration
  identity; unit tests in `tests/test_deeponet.py`.
- **DeepONet neural-operator baseline** (README Section 10.41): the repository's first
  neural-operator contribution, adding the operator-learning family (Lu et al., 2021) to
  the statistical-vs-physics-informed taxonomy. `src/models/deeponet.py`
  (`DeepONetDynamics`: the branch MLP encodes the 14-sensor state-action reading into a
  p=64 basis, the trunk MLP evaluates the basis at output-channel query coordinates in
  [-1, 1], plus a per-channel constant term; arbitrary-coordinate querying supported)
  trained under the exact unified protocol by `src/evaluation/deeponet_benchmark.py`
  (emulator-free, CI-safe; published comparison rows are read from the committed
  `benchmark_metrics.json`), writing `results/deeponet_benchmark_metrics.json` with
  `_meta` provenance and indexed in `results/MANIFEST.md`. Empirical outcome on the
  canonical split (seed 42): test MSE 7.8800 - 2.1x lower than the statistical MLP and
  the best physics-free architecture in the study - yet a kinematic residual of 273.15
  with a 99.58% multi-start violation rate (0% velocity violations), confirming at the
  operator level that structure-free learning does not recover the discrete integration
  identity. Wired as the `deeponet` CLI/Make target with smoke config
  `configs/smoke_deeponet.yaml`; unit tests in `tests/test_deeponet.py`.
- **Neural-operator family study: Physics-Constrained DeepONet + FNO** (README Section
  10.42): the two follow-up branches announced in 10.41.2, now measured.
  `PhysicsConstrainedDeepONetDynamics` (in `src/models/deeponet.py`) confines the
  branch/trunk operator to force/contact residuals and integrates them through the
  Section 4 hard kinematic shell, so its kinematic residual is identically zero by
  construction (52,742 params). `FNODynamics` (`src/models/fno.py`) is a Fourier Neural
  Operator (Li et al., 2021): 14-sensor lattice collocation, 2 spectral convolution
  blocks (6 Fourier modes, width 32), interpolated field decoding at output-channel
  queries (14,537 params). `src/evaluation/operator_benchmark.py` trains all three
  operators under the unified protocol with per-model reseeding (each row reproduces
  standalone; the DeepONet reference row re-produced 10.41 exactly) and writes
  `results/operator_benchmark_metrics.json` with `_meta`, indexed in
  `results/MANIFEST.md`. Genuine outcomes (seed 42): PC-DeepONet 0.5766 test MSE with
  0 violations on every rollout frame (matches the Hard PINN's 0.5783 / analytical
  zero); FNO 0.4025 test MSE - the repository's best single-step, 30% below the Hard
  PINN - but 88.4% multi-start kinematic-violation and 28.8% velocity-violation rates,
  the sharpest instance yet of the accuracy-vs-guarantees trade-off. Wired as the
  `operators` CLI/Make target with smoke config `configs/smoke_operators.yaml`; tests
  in `tests/test_deeponet.py` (exact-kinematics guarantee) and `tests/test_fno.py`.
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
- **Physics parameter identification - the inverse problem** (README Section 10.40): the
  repository's first inverse-problem contribution, recovering the seven engine constants
  $\theta$ (traction budget, subpixel ratio, asymmetric gravity pair, plus coast friction and a
  full four-channel rigid collision response on the terrain-contact byte) from observed
  trajectories via a differentiable analytic integrator and generalised
  (channel-variance-weighted) least squares. A Laplace / Gauss-Newton posterior
  (`posterior_laplace`) turns the point estimate into per-constant standard errors, a correlation
  matrix and a Fisher-eigenvalue identifiability diagnostic; full-covariance (Cholesky) sampling
  and a random-walk Metropolis sampler on the exact likelihood (`mcmc_random_walk`) propagate that
  uncertainty to a credible interval on the transfer result (MCMC agrees with Laplace to 0.25% on
  the strongly-excited synthetic case); a bootstrap cross-check and a data-range warm-start
  handle the inactive-constraint
  (velocity-ceiling) gradient pathology. New modules `src/inverse/parameter_identification.py`,
  `src/evaluation/inverse_transfer_benchmark.py` (emulator-free, CI-safe), wired as the
  `inverse-transfer` CLI/Make target; results in `results/inverse_identification_metrics.json`
  (E1 recovers a hidden seven-constant world - six to <0.7% and held-jump gravity to 2.5%
  relative error - with every constant flagged identified; E2 quantifies the residual
  misspecification ceiling on real gameplay; E3 shows the identified model transfers held-out
  control predictions like the oracle while the hard-coded prior is systematically optimistic);
  tests in `tests/test_inverse_identification.py`. `src/inverse` is covered by the mypy typed core.
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

- Typecheck gate on CI (mypy, exit 1/2 on the operator-learning commits): the reference
  hardware runs torch 2.5.1, whose stubs resolve `nn.Module` buffer attributes directly,
  while CI installs the newest torch inside the `>=2.5.1,<2.7` envelope, whose stubs type
  the same `register_buffer` attribute read as the union `Tensor | Module`. Surfaced as
  `"Tensor" not callable [operator]` (`fno.py`, sensor-grid line) and `Tensor | Module`
  assignment errors on the canonical query-grid buffers of `src/models/fno.py` and
  `src/models/deeponet.py` (both absent locally under torch 2.5.1 stubs). Fixed by
  adopting the project's documented buffer-typing convention (class-level
  `attr: torch.Tensor` annotations, cf. `src/models/cbf_projection.py`) for
  `sensor_grid`, `canonical_query_coords` and `residual_query_coords`; behavior is
  unchanged (annotation-only), and the gate is verified locally with a cacheless
  `mypy --no-incremental` over the 44 core modules.
- `scripts/navigate_to_level.py` (Yoshi's Island 2 capture, README Section 10.36.2): the
  level-2 branch pulsed Y to "dismiss a message box" even though a Y edge on that slide is
  documented to fire a $0x14\to0x_{C}$ map return - the pulses themselves collapsed the read
  into a wrapped $Y=65502$ transition state. Removed the Y-exit (idle-settle instead, which
  reaches a plausible deep in-level state) and replaced the static 60-consecutive-plausible-frames
  gate, which a frozen-but-stable frame can fool, with a control-response probe (Mario must move
  under held RIGHT, with START toggles for a possible entry pause). The capture is still honestly
  blocked - the player's physics do not step at this node (byte-identical $X,v_y$ under sustained
  input) - and `results/yi2_capture_attempt.json` is refreshed to record `no control handoff`, so
  the harness now fails loudly rather than saving a frozen frame.
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
