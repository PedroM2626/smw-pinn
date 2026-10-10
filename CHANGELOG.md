# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research artifact, "changed" includes *self-corrections of published
numbers* - those are reported here and in README Section 12, never applied silently.

## [Unreleased]

### Added

- **README 10.58, `src/evaluation/residue_process_study.py`** (`make residue-process`,
  `smw-pinn residue-process`, emulator-free, ~3 min over the six committed recordings): the residue of
  section 4.1 measured as a *process* rather than as an error bar - the lattice it lives on, its
  memory, the per-frame rules that fail to reproduce it, whether the richer recordings rank it, and a
  four-way calibration of drift-plus-Gaussian diffusion, iid jumps, a two-state geometric chain and a
  run-length renewal process against the recorded trajectories at 1/5/15/30/60 frames. The answer to
  "is it noise": no. The residual takes 5-13 values per recording and never leaves the sub-pixel grid
  (off-integer deviation exactly 0.0); an exception is followed by an exception at 104x-626x the rate
  after a clean frame; 75.4-99.3% of exception frames are positions that do not move while the
  velocity byte holds, where the residue is exactly minus that velocity; and the 197 frames the clamp
  leaves over are all exactly one whole pixel off. An iid fit is reasonable only at one frame - at 60
  it under-predicts the spread by 5.1-5.9x - and only the kernel that models run *duration* covers,
  which is what a constraint looks like to a model that cannot see it. The section is generated from
  the artifact including its prose, and `tests/test_residue_process_study.py` re-derives the
  quantifiers: exhaustive clamp accounting per recording, the growth of the over-dispersion, and the
  coverage ordering at each recording's longest horizon.
- **The corrected forms of README section 4 are constructible, and priced.**
  `src/utils/kinematics.py` names the two position-integration conventions;
  `ResidualDynamics`, `ProjectedDynamics`, `HardResidualPINNDynamics` and
  `AnalyticalKinematicsDynamics` take `position_velocity="next"|"carried"`,
  `GroundContactConsistencyLoss`/`CompositePINNLoss` take `contact_rule` (`zero_velocity` as
  published, `zero_increment` as §4.3.5 now states), and `RolloutEvaluator` takes
  `max_vx`/`terminal_vy`/`tolerance_px` so a rollout can be scored against the speed class the
  telemetry reaches. Every default is unchanged, deliberately: the parity control in the new study
  exists to prove the published path was not moved.
- **README 10.57, `src/evaluation/corrected_physics_ablation.py`** (`make corrected-physics`,
  `smw-pinn corrected-physics`, emulator-free, publishes no checkpoint): 16 arms over five seeds,
  built by the same `build_arm`/`build_loss` as 10.47, one axis changed at a time - the §4.1
  convention, the §4.3.5 ground rule, and the predicate's velocity bound - plus a same-weights
  replay that changes the graph without retraining, and a tolerance ladder (0.2 and 0.002 px)
  inside the study itself. `tests/test_corrected_physics_ablation.py` pins all four tables cell by
  cell *and* re-evaluates the quantifiers the findings use, including "at 72.0 every arm flags
  0.0000" and "the carried position error is one number across four models".
- **README 10.57.1, the same eight arms flown on the console** (`physics_injection_mpc_benchmark.py
  --study corrected`, the second step of `make corrected-physics`, 55 episodes in 16 min): the four
  shells x two conventions, with the published `next` arm beside each `carried` one and the three
  reference rows reproducing the 10.47 and 10.53 closed loops to the pixel. The console disagrees
  with the drift column for two of the three families - the corrected MLP/PINN arm loses 319.90 px
  ($p$ 0.018) and dies in the same hole in 5 seeds out of 5, where its published twin dies in 1,
  while the corrected FNO *gains* 28.61 px and the DeepONet moves 5.92 px at $p$ 0.651. It also
  found the study's own instrument limit: three of the eleven rows are duplicates from different
  weight files with byte-identical action sequences, so eleven episodes-per-seed of evidence is
  eight programs, which the gate now recomputes two independent ways.
- **`tests/test_cli_parity.py`:** the `smw-pinn` runner claims to mirror the Makefile and had not
  kept up - ten study targets from 10.45 to 10.56 had no subcommand. The ten are registered, with
  the new study, and the test refuses a single-module Makefile target without one.

- **A prediction target the repository never had** (README Section 10.53):
  `src/models/effective_velocity_dynamics.py` makes the velocity the engine *integrates
  with* the thing a network predicts, in three modes that differ in one line - `next`
  ($\hat x = x + \hat v_{t+1}/16$, the published shell, rebuilt so the axis can be flipped
  inside one class), `carried` ($\hat x = x + v_t/16$, the convention 10.49 measured on the
  console) and `offset` ($\hat x = x + (v_t + \varepsilon_t)/16$, a free head for the frames
  where the console leaves its own rule). `src/evaluation/effective_velocity_benchmark.py`
  trains 3 families x 3 conventions x 3 mechanisms over five seeds and reports the published
  predicate at a tolerance ladder (0.2 to 0.002 px) under both velocity conventions, with the
  console scored on the same grid. The `next` mode reproduces 10.47's `residual` cells to
  $0.0$ px of drift on all nine cells, so every contrast is the convention alone. The nine
  `carried` cells share one held-out position error, 0.1056 px with a spread of 0.000000 px,
  because no weight vector touches position there - better than the best learned position
  head in 10.47's grid (0.1184 px) - and they hold the published violation predicate at
  0.0000 down to 0.002 px with no clamp and no penalty, where the shells that publish 0.0000
  at the tolerance are flagged on 0.9345-0.9578 of frames once the tolerance is tightened.
  The composite penalty's kinematic term is identically zero on a carried graph (so part of
  10.47's "soft" attribution moves to the bound and contact terms) and is what destroys the
  offset head's gain on the graphs that have one (0.1065 px to 0.2119 px). On the console
  `physics_injection_mpc_benchmark.py --study effective` flies the nine unconstrained arms
  beside their published-convention twins: `fno_next_none` makes no progress
  ($-4.25 \pm 2.98$ px) while `fno_carried_none` covers 468.69 px (+472.94 px, $d_z = +2.24$,
  $p = 0.007$) and `fno_offset_none` covers 603.38 px (+607.63 px, $d_z = +227.20$); all six
  contrasts that move away from `next` are positive.
- **Section 4 audited against the code and the telemetry** (README Section 10.54):
  `src/evaluation/physics_claim_audit.py` records, for each physics claim of Sections 4.1-4.3,
  the README text itself, every file declared to implement it (verified by the source literal
  that makes it that implementation, so a refactor invalidates the audit instead of silently
  passing) and what the recorded transitions say. All nine claims are implemented by every
  declared site; two of them are the same integration identity written with two different
  velocities, and the audit reports that as a contradiction rather than picking a side; four
  are not satisfied by the console's own data - the §4.1 next-velocity reading, the §4.2.1
  jump window (7.39% of velocities below $-80$, min $-112.0$), the §4.2.5 terminal velocity
  (13.37% above $+64$) and §4.3.5's non-penetration condition, which the telemetry satisfies
  on 0.00% of the 2,943 frames it conditions on (median $|v_y| = 6.0$) - the condition
  `GroundContactConsistencyLoss` penalises.
- **The canonical sparse-identification estimator, run on every recording** (README Section
  10.56): `src/inverse/sindy_identification.py` implements SINDy as published (degree-2
  polynomial dictionary, ridge with the constant exempt, sequential thresholding to a sparse
  support), with two decisions that turn out to carry the section: the columns are standardised
  for conditioning only and rescaled back before thresholding, so $\alpha$ is in physical units,
  and the derivative is the forward difference over one frame, which is what the engine's own
  step is. `src/evaluation/sindy_identification_benchmark.py` identifies five recordings - the
  published gameplay set and the four this repository recorded for §10.31, §10.41, §10.45 and
  §10.52 - under least squares and a Huber variant, with a six-point threshold sweep. The Huber
  fit recovers the sub-pixel scale as 16.00 on all five recordings (worst deviation
  $9.4 \times 10^{-5}$) where least squares gives 16.17-19.01 and §10.37's gradient
  identification of the same recording gave 21.41; it recovers the gravity gate at 79-98% of its
  measured size where least squares returns it with the *sign inverted* on all four recordings
  that contain it; and it invents no gate on the fifth, which has none, so the spurious-structure
  list is empty. The threshold is a units statement: at $\alpha = 0.02$ the jump recording's
  displacement law reduces to one term, $0.0625\,v_x$, at relative error $1.8 \times 10^{-6}$,
  while at $\alpha = 0.1$ - above $1/16$ - the velocity term is deleted and the pooled held-out
  $R^2$ is *better* than at $\alpha = 0.5$. What the method does not recover is reported with the
  same table: the horizontal velocity law is a 23-term interaction pile at $R^2$ 0.124, §4.3.4's
  single friction coefficient does not exist in this data, and the pruning step shows why -
  36 of the 81 dictionary columns are constant or exact duplicates on the published recording,
  so no identification on it can exceed rank 45.
- **The numerical method as an axis** (README Section 10.55): `src/models/neural_ode_dynamics.py`
  reads the same networks used in 10.42/10.47/10.53 as a *continuous* acceleration field
  ($\mathrm{d}x/\mathrm{d}t = v_x/16$, $\mathrm{d}v/\mathrm{d}t = a(s,a)$) and takes one frame as
  one integration step of that field, under four named solvers: forward Euler, the velocity-first
  (semi-implicit) split every published shell implements, explicit midpoint and classical RK4.
  `src/evaluation/neural_ode_integrator_benchmark.py` trains 3 families x 4 solvers x 2 bounding
  mechanisms over five seeds, and its measure is a fitted one: the coefficient $c$ in
  $\hat x = x + v_t/16 + c\,\hat a/16$ is regressed out of each trained arm's held-out behaviour
  and the identical estimator is applied to the recorded console. Every arm measures its own
  method (0.0000 / 1.0000 / 0.5025-0.5042 / 0.5009-0.5014), and the console's fitted slope is
  0.4956 horizontally - which this section then explains away rather than publishes: the slope is
  carried by an average of 12.6 of 1,792 held-out frames per split, the exact-frame rate is 93.5%
  over all frames and 99.3% on the frames with no contact flag set, and on that contact-free
  subset the fitted slope falls to 0.0720. The engine's step is a first-order accumulation and the
  exceptions are collisions. Euler is also the best predictor (0.1056 px, the 10.53 `carried`
  floor, with the tight-tolerance violation rate at exactly 0.0000) while the second-order methods
  pay 0.1182-0.1209 px and the split 0.1281-0.1366 px, so the accuracy NeuralODEs are bought for
  is aimed at a truncation error the target does not have. Closed loop
  (`physics_injection_mpc_benchmark.py --study ode`) `deeponet_euler_free` covers
  643.02 $\pm$ 0.08 px, the tightest between-seed spread this repository has recorded for a
  controller that progresses (the only tighter row is `fno_euler_free` at 0.078 px, on -0.69 px
  of travel); for the MLP the ordering inverts against the open loop (midpoint and RK4 reach the
  frame budget in all five seeds at 610.40 and 614.09 px where Euler dies once at 521.66 px).
  The study also carries its own reproducibility check, because its `euler` and `symplectic` arms
  are 10.53's `carried` and `next` cells through a second code path: position error reproduces to
  0.0000 px and MLP closed-loop progress to +0.65 px, drift does not reproduce at all (gaps of
  0.0008 to 88.00 px) and the FNO's closed-loop row moves by -469.38 px - which corrects 10.53's
  finding 7 and is recorded in README Section 12. The comparison is attributable to the code path
  and not to the machine because the command was run twice: `results/checkpoints/ode_published.sha256`
  holds the first run's hashes for the twelve published weights, the second run's files match them
  byte-for-byte, and `tests/test_neural_ode_dynamics.py` recomputes the committed files against that
  listing on every CI run.



- **The excluded cell, filled** (README Section 10.51): `src/models/output_projection.py`
  imposes the engine's velocity bounds as a post-hoc projection of a *state-output*
  network - clip the proposed velocities, re-derive the positions from the clipped
  values - which is the cell 10.47 declared unimplementable and left out, and is
  deliberately not the CBF action filter of 10.16. `src/evaluation/projection_cell_benchmark.py`
  fits one per family under the grid protocol and
  `physics_injection_mpc_benchmark.py --study projection` flies them with the published
  planner. On open-loop prediction the cheap mechanism wins: the projection beats the
  in-graph shell by 41.98 px of drift for the MLP ($d_z = -2.64$, $p = 0.004$) and
  31.67 px for the DeepONet ($p = 0.031$), tying on the FNO (+7.46, $p = 0.51$), at no
  cost in data fit. On the console the ordering inverts for two of three families - the
  projection covers $299.50 \pm 252.34$ px where the PC-DeepONet covers
  $635.61 \pm 14.18$, and $207.50 \pm 143.06$ with three deaths where the hard FNO
  covers $554.74 \pm 28.61$ with none - while for the MLP it is worth +221.34 px over
  the shell and is statistically indistinguishable from the engine rules ($-99.37$ px,
  $d_z = -0.46$, $p = 0.365$). The candidate mechanism is measured rather than asserted:
  the wrapper overwrites its own network's velocity on 15.41% of held-out frames for the
  MLP against 18.05% and 18.54% for the two operators that lost.
- **The gravity gate under targeted excitation** (README Section 10.52):
  `scripts/record_jump_gameplay.py` records WRAM telemetry whose policy alternates long
  jump holds with short early releases (55% of jumps released within three frames), 6,116
  transitions of which 942 are airborne, rising and already released;
  `src/evaluation/gate_excitation_benchmark.py` measures the tiers straight from the
  console and then hands three recordings to the engines of 10.43.9 unchanged. The
  published recording turns out to contain no second vertical tier at all: its
  released-ascent frames (513 of 6,329 training transitions, so the branch is not scarce)
  have median $\Delta v_y = 3.0$, equal to the held branch, equal to falling, and
  unchanged under every realignment of the button latch against the physics from $-2$ to
  $+2$ frames - which makes four published negatives (10.37.1, 10.43.9, 10.46, 10.47)
  accurate measurements rather than estimator failures, and narrows 10.45's coverage
  remedy to "sample the branch where releasing changes the fall". Both MPC recordings do
  exhibit it (6.0 released against 3.0 held, uniform across all four ascent-phase bands),
  and there the template engine discovers the gate with all four criteria agreeing -
  while fitting only $-0.8117$ and $-0.6734$ of a true $-3.0$ step. gplearn recovers
  nothing on any recording; the PySR leg could not run in this environment and is
  recorded with the reason. The velocity clamp moves again under a second targeted policy:
  35.150 here against 36.075 on the sprint recording and 47.775 on the published one.


- **The plateau-coincidence test** (README Section 10.50):
  `src/evaluation/plateau_provenance_benchmark.py` decides the question 10.46 recorded but
  refused to interpret - the plain DeepONet's implied plateau of 35.61 sitting 1.3% from
  the console's sustained 36.075 - by manipulating the data instead of the estimator: each
  family is retrained on training splits truncated at four velocity caps (49, 36, 30, 24)
  and probed over the full recorded range, three seeds per cell. The DeepONet's plateau
  follows its support down ($32.12 \to 17.36$, slope $0.644$, $r = 0.924$) and the published
  35.61 lies inside the seed spread of the untruncated cell ($32.12 \pm 9.08$), so the
  coincidence was the recording plus one draw, not the representation. The FNO tracks too
  ($0.709$, $r = 0.824$); the MLP does not ($0.329$, $r = 0.424$) and is not even monotone -
  capping its support at 30 moves its plateau *up* to 39.14 - which is the same lesson
  10.43 drew from the other direction: a fixed point of the driven map is a statement about
  the extrapolation, not about where the data ends.

- **The rollout violation figure decomposed** (README Section 10.48):
  `src/evaluation/rollout_diagnostics.py` separates the three properties the published
  predicate mixes - integration consistency measured against the velocity a model
  actually advanced by, step-to-step velocity smoothness, and bound exceedance - and
  `src/evaluation/kinematic_metric_decomposition_benchmark.py` re-scores every
  committed model (6 published + 15 grid arms) with the three quantities separated,
  plus the recorded telemetry itself as a reference row. For an exact integrator the
  published figure *is* the jump rate, digit for digit (`grid:FNO/residual/soft`:
  violation 0.1975, jump rate 0.1975, integration residual $9.2\times10^{-6}$ px),
  Mario's own transitions trip the predicate on 0.0462 of frames, and models posting
  0.0000 violations leave the velocity bounds on up to 46.6% of rollout frames. The
  study also re-classifies the probed ceilings under four traction thresholds and
  finds the accepted set is 3, 3, 9 and 9 records at 1.0/1.5/2.5/3.5 px/frame - so
  Section 10.46's "three of six pass" is load-bearing in its chosen constant.
- **Which documented constant the telemetry supports** (README Section 10.49):
  `src/evaluation/velocity_class_benchmark.py` measures the velocity envelope of all
  four recordings (45,389 transitions) against Section 4.3's three horizontal classes
  and against Section 4.2's vertical window, and tests the two integration conventions.
  No frame anywhere exceeds 49.0 sub-pixels/frame and none reaches the P-meter class of
  72.0, so the repository's universal `max_vx = 72.0` is the cap of a speed class the
  data never enters; re-scored against the documented run cap of 48.0, the template
  engine's 47.775 is 0.47% away (it had been published as 33.6% off) and the sustained
  36.075 of 10.45 is 24.8% below the cap, i.e. a policy's sustained speed rather than a
  bound. The vertical window $[-80, +64]$ is exceeded on 16.3% to 29.2% of recorded
  transitions (the data reaches -112.0 and +70.0), so the hard shells rewrite real
  console states on a fifth to a third of frames, and the console integrates position
  with the velocity at frame $t$ (median residual exactly 0.0000 px) while every
  implementation in the repository uses the predicted $t+1$ velocity (median 0.0625 px).
- **Physics-injection grid** (README Section 10.47): `src/models/residual_dynamics.py`
  factors the two things the published shell models always changed together - the
  *target* the network predicts and the *kinematics the graph enforces* - into one
  switchable shell, and `src/evaluation/operator_physics_injection_benchmark.py`
  crosses it with family: 3 architectures x {state, residual} x {none, soft, hard}
  = 15 cells plus the two published shell classes re-fitted as references, five seeds
  each under the 10.42 protocol (~27 min, emulator-free). A hard FNO did not exist
  before this section: `hard` had exactly one instance in the repository. Outcome:
  the kinematic-consistency property credited to the shell in 10.27 and 10.42 belongs
  to the increment parameterisation (violation $0.0000$-$0.0210$ for every residual
  cell trained on its data term or behind the shell, versus $0.8470$-$0.9987$ for every
  state cell); the clamp buys
  boundedness and $-26.7$ to $-34.0$ px of drift; the soft penalty costs the
  horizontal channel a factor of 5-12 at the state target and beats the shell on drift
  at the residual target for two of three families. The shared shell reproduces the
  published classes bit-for-bit (`MLP/residual/hard` = the re-fitted Hard PINN at test
  loss $0.621353$ and drift $137.062939$ px), and `tests/test_residual_dynamics.py`
  pins that equality and the exclusion of the ill-defined `state x hard` cell.
- **The same arms flown on the console** (README Section 10.47):
  `src/evaluation/physics_injection_mpc_benchmark.py` drives nine controllers with the
  published 10.44 planner and reproduces its three reference rows exactly, then
  measures that the sign of the shell's contribution is not a property of physics:
  a DeepONet predicting increments with *no* constraint is statistically
  indistinguishable from the hand reverse-engineered engine rules
  ($623.06 \pm 17.55$ px, $d_z = +0.09$, $p = 0.84$), the same FNO with the shell is
  the third-best controller in the repository ($554.74 \pm 28.61$ px) while without it
  it walks backwards ($-4.25 \pm 2.98$ px, $d_z = -35.66$), and for the MLP the shell
  costs $199.64$ px and two extra deaths. A matched pair also closes the
  accuracy-versus-control gap quantitatively: the soft penalty improves the FNO's
  rollout drift by $58.0$ px ($p = 0.024$) and its closed-loop progress by $0.38$ px.
- **The 10.46 structural probes applied to the grid** (`--registry grid`, a separate
  artifact so the published 10.46 file is never rewritten): five of the nine residual
  cells keep a plausible-traction plateau at $18.89$-$26.02$ px/frame, every
  `residual/soft` cell and the hard FNO have no fixed point at all, no cell of any
  family or mechanism recovers the gravity gate, and identification through a
  surrogate becomes survivable when the target is an increment and the arm is trained
  on its data term ($98.3$-$99.4\%$ worst-case error for every `none`/`hard` residual
  arm against $87.2$-$1646.6\%$ for the state arms, with the `residual/soft` DeepONet and
  FNO back at $688.8\%$ and $539.6\%$).

- **Learned operators driven as controllers on the real console** (README Sections 10.44,
  10.44.1): `src/evaluation/inverse_model_mpc_benchmark.py` now also loads the DeepONet,
  Physics-Constrained DeepONet and FNO checkpoints of 10.41/10.42 and pilots them with the
  same CEM-MPC planner, objective, savestate, frame budget and seed protocol, so the closed
  loop tests the prediction Section 10.42.3 states as a hazard. Outcome: the hybrid is the
  best controller in the repository ($635.61 \pm 14.18$ px, 5/5 seeds, zero deaths,
  $d_z = +0.50$ and statistically indistinguishable from the hand reverse-engineered rules);
  the FNO loses 247 px to those rules with $d_z = -12.14$ on a spread of only $\pm 3.56$ px,
  i.e. it fails reproducibly rather than noisily, which is the accuracy-versus-control
  dissociation 10.42 predicted; and the plain DeepONet fails the *other* way
  ($193.66 \pm 197.54$ px, one death). The Hard PINN and the PC-DeepONet share the same hard
  kinematic shell and differ by 336 px and three deaths, so the shell bounds the failure mode
  rather than determining the behaviour.
- **Structural probes on learned dynamics models** (README Section 10.46):
  `src/evaluation/learned_structure_probe_benchmark.py` puts the committed checkpoints of
  Sections 8, 10.41 and 10.42 under the same interrogation the discovered laws were given -
  the fixed point of the fully-driven map, the median held-jump tier separation - with the
  same acceptance rules, and reports which model actually contains the engine's constraint.
  Emulator-free (~1 min), wired as the `learned-probes` Make/CLI target, artifact indexed in
  `results/MANIFEST.md`, tested in `tests/test_learned_structure_probe.py`.
  Outcome: all six models reach a stable positive fixed point, but three of them only after
  accelerating at 12-21 px/frame against the engine's maximum real increment of 1.8, so the
  crossing is an extrapolation artifact and the study classifies it as one - three models
  keep a plausible ceiling, and none of the three is near the engine's (21.7-25.5 against a
  WRAM reference of 72.0, a support of 49.0 and the 36.075 that 10.45 measured on purpose).
  No model recovers the gravity gate (largest separation 0.86 against a true -2.80). The
  clamp carried by the Hard PINN and PC-DeepONet shells sits at 72.0, above the observed
  support, so it never binds and the probe is not a tautology - recorded per model as
  `shell_clamp_binds_within_support`.
- **Identification through a learned surrogate, measured rather than argued** (README Section
  10.46): the same 900-step, warm-started identification of 10.40-E2 is re-run against
  pseudo-transitions rolled out by each learned model, to settle the question of why nobody
  differentiates the constants through a trained operator. Every surrogate is worse than the
  recording it was trained on - 87.2% worst-case error for the best (FNO) against the direct
  fit's 84.1%, which itself reproduces the published 10.40-E2 figure - and the drift of the
  recovered constants tracks the surrogate's own quality (3.9 units for the two
  kinematically-constrained models, 23.8 for the Soft-PINN, whose identification loss
  diverges to 4.8e7). The README also records the structural half of the argument, verified
  rather than asserted: `simulate_step` is differentiable w.r.t. theta, and
  `src/environment/snes_emulator.py` is a ctypes binding with no tensor path, which is why
  every emulator-in-the-loop method here is gradient-free by necessity.
- **Multi-seed closed-loop comparison** (README Section 10.44.1):
  `src/evaluation/inverse_model_mpc_benchmark.py` now accepts `--seeds` and, over CEM seeds
  42-46 with the world models fitted once, reports mean $\pm$ std per controller, the
  budget-reached and pit-death counts, and a paired test against the established-rules
  controller (`_paired`, with the Wilcoxon floor at $2/2^5$ stated rather than glossed).
  Outcome, recorded as a self-correction in README Section 12: the 10.44 table's ordering of
  the two closed-form models was single-draw noise and reverses - the identified constants
  are 8.10 px *ahead* on the mean ($d_z = +0.54$, $p = 0.29$, indistinguishable), the
  symbolic failure is robust and enormous ($d_z = -8.42$, no overlap in five pairs) but its
  seed-42 pit death is not (1 death in 5), and the published PINN row is bimodal - three
  seeds die near 114 px, two reach ~576 - so $299.20 \pm 253.32$ px is the number to quote,
  matching the shape of the independent 10.38 reproduction ($420.9 \pm 266.2$ px).
- **Structure-specificity control for the template engine** (README Section 10.43.9): the
  authorship objection - the nested-template dictionary was written by someone who had
  already read the 10.37 engine rules - is now tested rather than conceded.
  `fit_ceiling_families` in `src/inverse/structure_selection.py` fits four *mutually
  exclusive* explanations of a velocity plateau to the same driven rows (no ceiling, rigid
  clamp, quadratic drag with an asymptote, exponential relaxation) and the study generates
  data from each of three of them. The selector names the true mechanism in 3 of 3 controls
  with BIC, held-out RMSE and the tail criterion agreeing every time, so the machinery
  discriminates between structures instead of confirming the one the console happens to
  use. Unit-tested per mechanism, including the signature that distinguishes drag from a
  clamp (two asymptotes, one per traction tier).
- **gplearn search-effort sensitivity** (README Section 10.43.9): the published limitations of
  10.43.9 named this measurement as the obvious missing one, so it is now run instead of
  deferred. `run_gplearn_effort_sweep` sweeps population x generations over
  $500\times25 \to 2000\times300$ - a 48x increase in evaluations, starting where 10.43's S1b
  left off and reaching far past it - with the row budget pinned so only effort varies. The
  fixed point is absent at the first three budgets and appears at $2000\times300$, and held-out
  $R^2$ *falls* from 0.926 to 0.907 at exactly that step: under a mean-error fitness the
  constraint is available to gplearn too but is paid for in accuracy, which is the mechanism
  10.43.9 hypothesised and had not measured. Together with the PySR flip at 3x iterations this
  makes the section's claim quantitative rather than qualitative - both tree engines are
  budget-limited, and what distinguishes them is the price of the structure ($3\times$ versus
  $48\times$), not whether one is blocked. Finding 7, the limitations of 10.43.9, 10.43.7's
  limitation (i) and README Section 12 carry the corrected wording; the sentence that created
  the debt ("gplearn was not swept, which is the obvious next measurement") is gone.
- **PySR search-budget sensitivity** (README Section 10.43.9): the published ablation now
  also re-runs the PySR leg at `--pysr-budgets 40,120` iterations on the same replicate and
  reports whether the structural answer moves with the budget. It does, and the entry records
  that as a correction to the section it belongs to: at 40 iterations the bound is missed
  (held-out $R^2$ 0.9983, driven map overshooting the velocity support by 0.0014 px/frame with
  no fixed point, 633 s of serial search), at 120 iterations the same search returns the whole
  law ($R^2$ 1.0000, overshoot 0.0, 1030 s) - the stored expression evaluates to
  $\hat v = \min(v + 1.8, 48.0)$ on the driven branch, with the walk tier at 1.0, the run tier
  at 1.8 and the Coulomb deadband at 0.5999992 against a true 0.6. Discovery is therefore
  budget-limited here, not blocked by the representation or by the fitness, and §10.43.9's
  criterion-only reading is explicitly narrowed to the budget at which it was measured.
- **Excitation-targeted recording and ceiling re-measurement** (README Section 10.45):
  `scripts/record_sprint_gameplay.py` records a second WRAM dataset whose only purpose is to
  saturate the speed bound - the 10.37 established-rules MPC drives Mario with a
  speed-weighted objective, and every transition is read back from WRAM after the frame, so
  the policy chooses which states are visited and never what those states were.
  `src/evaluation/sprint_excitation_benchmark.py` then runs the excitation profile and all
  three engines of 10.43.9 over both recordings under the identical protocol, and its verdict
  reports the *direction* the clamp estimate moved against the WRAM reference rather than the
  direction that would be convenient. Emulator required for the recording, not for the
  analysis; the published datasets are never overwritten.

- **Closed-loop control with inverse-problem world models** (README Section 10.44): the
  inverse answers of 10.40 and 10.43 are driven on the real console for the first time.
  `src/models/inverse_world_models.py` wraps the identified seven-constant map and the
  bagged genetic-program laws as `nn.Module`s on the planner's
  `(state, action) -> next_state` contract (contact bytes propagated from the current
  frame, the same convention as the 10.37 baseline), and
  `src/evaluation/inverse_model_mpc_benchmark.py` runs one CEM-MPC controller - horizon
  15, 256 candidates, 3 refinement iterations, the published 10.6 objective, the
  interactive Yoshi's Island 1 savestate, 300 frames, `set_global_seed` before every row -
  against four dynamics models: the established WRAM engine rules, the 10.40 identified
  constants, the 10.43 discovered laws and the published Hard Residual PINN. Emulator and
  ROM required; without them the entry point exits with a diagnostic and writes nothing.
  Writes `results/inverse_model_mpc_metrics.json` and
  `results/figures/inverse_model_mpc_progress.png`, wired as the `inverse-mpc` Make/CLI
  target and tested in `tests/test_inverse_world_models.py` (wrapper/integrator parity,
  action-agreement metric, figure emission, the no-hardware refusal).
  Outcome: the identified constants match the hand-measured ones (599.75 vs 605.00 px)
  while reproducing only 49.7% of the reference controller's actions, and `max_vx` never
  leaves its 72.0 warm start; the discovered laws reach 60.31 px and die in a pit at frame
  282 because their horizontal program is the bare contact terminal `ceiling` - a
  persistence model, indistinguishable from 10.43.5's null on the one-step metric, which
  cannot plan a run. 605.00 px and the PINN's 573.94 px reproduce the published 10.37.2 and
  10.6 rows, so the harness is the published one.
- **Three-engine discovery ablation** (README Section 10.43.9): the caveat Section 10.43
  attached to its own negative result - that the undiscovered velocity ceiling was a
  statement about *one* representation - is now measured instead of stated.
  `src/inverse/structure_selection.py` adds a nested-template engine: twelve structures
  (`H1_constant` through `H6_contact`, `V1_constant` through `V6_ground_reset`), each fitted
  with active-set clamping on the rows where a constraint binds, scored by BIC on the fit
  rows, by held-out RMSE, and by held-out RMSE restricted to frames at >=90% of the observed
  speed support, with an agreement flag whenever the three criteria disagree.
  `src/evaluation/symbolic_engine_ablation_benchmark.py` runs that engine, the published
  gplearn estimator, and PySR (unbounded numerically-optimised constants, `min`/`max`
  available, serial and deterministic; an optional `symbolic` extra, recorded as unavailable
  rather than skipped) on identical design matrices, row budgets, held-out banks and probes,
  on the hidden world (3 replicates x 3 seeds) and again on genuine WRAM telemetry, and
  assembles its verdict from the measured rates so no clause of the conclusion is asserted
  ahead of the number. Emulator-free, ~2 h CPU with the PySR legs and both budget sweeps; writes
  `results/symbolic_engine_ablation_metrics.json` and
  `results/figures/symbolic_engine_ablation.png`, indexed in `results/MANIFEST.md`, wired as
  the `symbolic-engines` Make/CLI target with smoke config
  `configs/smoke_symbolic_engines.yaml`, tested in `tests/test_structure_selection.py` and
  `tests/test_symbolic_engine_ablation.py` (including tests that the reading refuses to claim
  an accuracy advantage it did not measure).
  Outcome, reported as a self-correction in README Section 12: the ceiling is *identifiable*
  from these windows - as a candidate structure it is recovered at 48.000 against a true 48.0
  and preferred by BIC and both held-out criteria in 3/3 replicates, with constants within
  1.4e-4% of truth - but free tree search misses it identically for both engines at the
  published budget (0/3 bagged gplearn replicates, 0/3 PySR fits) even though PySR fits the
  same law far better ($R^2$ 0.988 vs 0.805), which rules out the terminal range and the fit
  accuracy as the explanation at that budget. The budget itself is not ruled out: the same
  search at 120 iterations discovers the law outright (see the entry above), so this section's
  conclusion is stated as budget-limited rather than as a property of the fitness. The
  gravity gate is the opposite case: gplearn's drawn constants recover it in 1/3 replicates
  (tier separation -1.35 of a true -2.80) and PySR's optimised constants in 3/3 (-2.80),
  which is a representation limit. On telemetry the ordering survives (template 0.989, PySR
  0.430, gplearn -1.7e-4, reproducing 10.43.5 to the digit) and a new trap appears: PySR
  reports a fixed point at 34.193 that the recording itself violates at 49.0, so a
  turnover-finding probe can accept a bound the data contradicts. Section 10.43.7's item 1
  and limitation (i) carry the narrowed wording.

- **Terrain-conditioned residual discovery** (README Section 10.43.8): the counterfactual
  that Section 10.43.5 only inferred is now measured.
  `src/evaluation/symbolic_tilemap_residual_benchmark.py` re-runs the grey-box residual
  control on `smw_tilemap_dataset.npz`, where the recorded 7x7 local block buffer is
  available to the search, under five conditioning conditions (`state`, `state+geom`,
  `state+geom+inter`, `state+full`, and a shuffled-geometry placebo), each scored by
  genetic programming (median over 3 seeds, best and worst reported) and by least squares
  on the identical design matrix. Support requires the geometry condition to predict out of
  sample, to beat the state-only fit, *and* to beat the placebo - "less catastrophic" is
  explicitly not "explained" (`build_verdict`, unit-tested against both failure modes).
  Emulator-free, deterministic, ~6 min CPU; writes
  `results/symbolic_tilemap_residual_metrics.json` with `_meta`, indexed in
  `results/MANIFEST.md`, wired as the `symbolic-tilemap` Make/CLI target with smoke config
  `configs/smoke_symbolic_tilemap.yaml`, tested in
  `tests/test_symbolic_tilemap_residual.py`.
  Outcome: terrain occupancy explains the identified model's *vertical* velocity residual
  (held-out R^2 0.311 -> 0.365, placebo -0.007, best expression a ground-gated downward
  reset `mul(max(mul(patch_fill, vy_x_below), vy), ground)`) and does not explain the
  horizontal one (gain exactly +0.000, no terrain descriptor correlating above 0.068); the
  placebo additionally caught a +0.077 apparent gain in the y channel as overfitting, and
  least squares beats genetic programming on the horizontal residual (+0.081 vs -0.011).
- **Symbolic regression as an inverse-problem method** (README Section 10.43): the
  inverse line of Section 10.40 extended from *fitting constants inside a posited law*
  to *discovering the law*, with genetic programming (`gplearn`, piecewise-affine
  primitive set) on four one-step update channels. `src/inverse/symbolic_regression.py`
  adds the transition-bank plumbing, excitation-normalised fitting (GP's bounded
  terminal constants cannot otherwise express a 48-sub-pixel ceiling and a 1-sub-pixel
  increment at once), the seed-bagged `BaggedLaw`, the `AnalyticLaw` wrapper that puts
  the 10.40 parametric map behind the same `increment()` contract (so both models are
  composed, probed and rolled out by identical code), and the **probe stage** that
  defines each of the seven constants as a response of the discovered map - the velocity
  ceiling as its fixed point, rejected when the map never accelerates, so a degenerate
  zero-drive law cannot masquerade as a discovered bound.
  `src/evaluation/symbolic_inverse_benchmark.py` runs S1 (6 replicates x 3 GP seeds,
  paired Wilcoxon/t/Cohen's d_z against the parametric estimator refit on the same
  windows), S1b (budget control over a 13x search range, reporting accuracy *and*
  structure), S1c (excitation-reweighting control), S2 (120-frame rollouts: drift, cap
  overshoot, integration residual), S3 (real WRAM telemetry plus a grey-box control that
  fits GP to the identified model's residuals over all eleven observable channels) and
  S4 (the 10.40-E3 control battery, with a reproduction check against the published
  rows). Emulator-free and deterministic; writes `results/symbolic_inverse_metrics.json`
  with `_meta`, indexed in `results/MANIFEST.md`, wired as the `symbolic-inverse`
  Make/CLI target with smoke config `configs/smoke_symbolic_inverse.yaml`, tested in
  `tests/test_symbolic_regression.py`.
  Empirical outcome - a mostly *negative* result, reported as such: the discrete
  integration identity is rediscovered exactly (1-node program, `R^2 = 1.0000`, probed
  scale 0.001% off on synthetic and 0.69% off on real telemetry, beating the parametric
  fit of the same constant), but the rigid velocity ceiling is never discovered (0 fixed
  points in 27 draws over a 13x budget range; 68.7% of rollout frames above the true cap,
  and bagging *worsens* it to 83.4% because averaging non-saturating laws does not restore
  a constraint), the held-jump gravity gate is absent from 78% of vertical laws, and
  aggregate accuracy stays 2-3 orders behind the posited structure (weighted one-step MSE
  1.15e-2 vs 8.12e-5, Wilcoxon p = 0.03125, d_z = 3.18). Growing the budget lifts held-out
  `R^2` from 0.13 to 0.62 without moving either structural read-out, while over-weighting
  the under-excited frames cuts `g_hold` error from 81.3% to 11.6% at a measurable cost in
  aggregate accuracy - excitation limits discovery, not just identifiability. On genuine
  telemetry the discovered velocity laws do not beat the fit-mean predictor at all, which
  is why every per-law row is published against a null baseline: the honest reading is
  that "no force law + exact integration + the contact byte" (weighted 0.0144) out-predicts
  the posited-but-misspecified 10.40-E2 analytic map (0.0746), and the residual control
  localises the remaining gap to `v_x`, whose analytic-model residual has negative *train*
  `R^2` - unobserved tile geometry, not a better estimator. `gplearn` added to the runtime
  dependencies (manifest parity + `requirements.lock` audit lines).
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

- The validated `mypy` band moved from `>=2.3.1,<2.4` to `>=2.3.1,<2.5` (`pyproject.toml`,
  `requirements.txt`), accepting Dependabot PR #8 rather than merging it blind: `mypy 2.4.0`
  was pointed at the exact typed-core gate `src/cli.py` passes (the same 52 modules) and
  reported no issues, and the 3.11/3.12/ubuntu and native-Windows legs the pin now allows are
  validated by the CI run the merge triggers. The band stays an upper bound on purpose - the
  reason it exists is CI run #14, where an unbounded `mypy>=1.0.0` resolved an interpreter that
  crashed the typecheck gate with exit code 2.
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

- **Four of the twenty-five formulas GitHub renders with their subscripts eaten.** An underscore
  can satisfy CommonMark's emphasis flanking *inside* a formula, so the parser pairs it with another
  underscore on the same line and the page shows a rendered expression that no longer says what the
  source says - no raw LaTeX, no error box, nothing that looks broken. Counted on the live page:
  **25 emphasis runs still hold LaTeX fragments**, so 25 formulas are silently wrong on the rendered
  README. Four were fixed in this pass, chosen because their shape is the one a local rule can
  state - two `}_{` / `}_\` subscripts sharing a line (the Section 5.4 LSTM formulation, the
  gradient-stiffness bullet, the PPO physical-residual charge, the hybrid actor's four integration
  rules); each formula now has a line of its own and
  `test_no_two_flanking_underscores_share_a_line` reports exactly those four on the previous commit
  and none on this one. The remaining 21 pair an underscore in a formula with one outside it, a
  shape no local grammar of GitHub's parser has been derived for; they are recorded in README
  Section 12 with their count, because a defect you cannot assert is a defect you have to publish.
- **Twenty-four expressions the page showed as source, with no error box to say so.** Two rules of
  GitHub's renderer, both measured on this repository's own page rather than assumed: it forms no
  math inside emphasis (317 `<em>` runs, 0 of them holding a formula), and it does not read a `$`
  that touches a word character as a delimiter. So `*Measured … $\hat v_{x,t+1}$ …*` notes and
  spans like `344.73 $\mu$s`, `$\approx$173`, `$122\times$`, `frame$^2$`, `rank-$p$` and `$x$/$y$`
  printed literally - valid LaTeX, no "Unable to render expression." anywhere, invisible to the
  tokenizer checks and to the render-budget gate. `src/utils/typography.py` now converts the
  typographic ones to the Unicode they always meant and `unemphasise_math` moves a whole-line
  note's italics onto its label; the three that were neither were reworded. The renderer that
  writes Section 4's measured qualifiers emits the new form, so the document and its citation gate
  changed together. `test_no_span_is_written_where_github_cannot_form_it` refuses both shapes and
  was validated against the previous commit's README, where it reports 30 sites.
- **The table of contents was missing five sections, and one entry was in the wrong place.** The
  10.57 ToC line had been pasted onto the prose of Section 12 instead of into the contents, so the
  section had no entry and Section 12 had a stray bullet in the middle of a sentence; Sections
  10.1-10.4 had never been listed at all. `test_every_study_section_has_a_table_of_contents_entry`
  now checks the direction the anchor check never did - every `### 10.x` heading must be linked from
  the contents block - and refuses an entry sitting anywhere else in the document. Both mutations
  (drop an entry, move one into the body) were verified to make it fail.
- **A claim about the repository's own state that the batch itself had made false.** README
  Section 12's eighth entry closed by saying §4.3.5 "declares zero implementing sites because
  nothing in the repository yet implements the corrected rule", which was true when §10.54 first
  ran and stopped being true in the same commit that added `contact_rule="zero_increment"` to the
  penalty and the ground branch to `AnalyticalKinematicsDynamics` - §10.54's own finding 2 records
  the two implementations. The sentence now states what the audit measures, the same entry carries
  10.57.1's closed-loop half of the decision instead of leaving it as an inference from drift, and
  Section 1's range over the study sections - still "10.37-10.56" after 10.57 shipped - is pinned by
  `test_the_readme_ranges_its_own_study_sections_correctly`, which refuses a range that stops short
  of the section it is printed in.
- **Expressions the README's own renderer refuses.** `$ +15.71$` is not math to GitHub: a
  span may not open after a space, so the cell's dollars paired against their neighbours and
  KaTeX was handed `15.71$, $` - "Unable to render expression." The same mispairing came from
  WRAM addresses left out of a code span (`($7E:00E4 / 7E:00D8$)`), from one code span split
  across a line break, from a `next`/`carried` pair whose closing backticks had been eaten,
  and from a `0x_{C}` subscript that was really the world-map mode `0x0C`. Raw `<`/`>` inside
  `$…$` are now `\lt`/`\gt`, which survive the HTML pass. `tests/test_readme_math_rendering.py`
  pins all four properties over the whole file and is validated by corrupting one span of
  each kind; a fifth test checks that every table-of-contents link lands on a heading.
- **The 10.47 control-table gate was the source of its own unrenderable cells.** It built the
  expected row with `" $ {:+.2f}$, $ {:.4f}$"`, so the README matched a form GitHub cannot
  render and the table it generated was wrong in exactly the way the test could not see. The
  format string now emits `$+15.71$, $0.3125$`.
- **Section 10.54's claim column printed LaTeX as text.** `render_audit_table` stripped the
  `$$` from a stored claim without re-delimiting it, so five of its ten rows showed
  `\hat{X}_{t+1} \ne ...` literally. Claims are now quoted as inline math, with
  `\lvert`/`\rvert` for absolute values: a raw `|` splits a table row and the usual `\|`
  escape is KaTeX's *double* bar.
- **Retired weight counts** (README Sections 1, 5, 7, 9, 10.41; Section 12 records it): the
  Hard PINN was printed as 9,992 and 41,862, the MLP as 36,360 and the LSTM as 206,600. The
  profiler, every study artifact and the constructor defaults the canonical benchmark uses
  agree on 36,486, 36,744 and 223,368; 9,992 was nearest the compact `--matched-baseline`
  pair (10,054 / 10,184), a different configuration, so Section 1's "72.5% lighter than the
  MLP" compared a model that was never trained here against one that was.
- **Six other statements the committed artifacts contradict**, each corrected to the value
  the artifact carries: Section 10.4's "304 px" of drift is 78.13 px at the published start
  and 288.73 px over the 10-start variant; Section 7's hypothesis-test heading said 5 seeds
  where `multiseed_benchmark_metrics.json` records 10 (its Wilcoxon floor is 0.002, which 5
  seeds cannot reach); Section 10.29.3's DAgger row is 831.75 px at 2,707.5 FPS, not 833.50
  at 2,860.4; Section 10.10 claimed the ensemble's variance flags an out-of-distribution
  shock while its own table shows the spread *falling* (0.4609 in, 0.4189 out, ratio 0.909),
  which is retracted - the table's verdict cell now states the ratio it measures, and
  `test_readme_10_10_ensemble_uncertainty_is_the_artifact` pins the three figures *and* the
  direction, so a re-run that reversed the ordering cannot pass unnoticed; Section 10.19's
  "prior 164 px barrier" was not the prior best (115.0 px
  blind, 328.9 px hand-coded); Section 10.52's re-scored ceilings were quoted as 25.6% and
  26.8% below the run cap where the fitted clamps give 26.8% and 24.8%; and Section 10.51
  credited three of six soft cells with zero out-of-bounds frames where the grid gives four.
- **Section 1's headline superlative was a scope error, not a wrong number.** It read "the
  lowest reported prediction error and the strongest kinematic consistency across the evaluated
  metrics"; on the same split and seed, 10.42 measures `PhysicsConstrained_DeepONet` at $0.5766$
  and `FNO` at $0.4025$ against the Hard PINN's $0.5783$, 10.37's zero-parameter engine rules carry
  an *identically* zero kinematic residual, and 10.48/10.53 show the quoted `0.0%` violation rate
  is a within-tolerance result (0.9345-0.9578 of rollout frames are flagged at 0.002 px). The
  sentence is now confined to the four architectures of Section 8, says so, and the section names
  the studies that train the repository's other models.
- **GitHub's math budget, measured, is why the second half of the README never rendered.**
  The repository's page returns the generic "Unable to render expression." for every
  expression past the first ~1,368: 1,368 came out as formulas, the next 300 did not, and all
  of them were syntactically valid. The README held 1,759 spans, so Sections 10.52-12 were
  unreadable however correct their LaTeX was, and no syntax check could see it.
  `src/utils.typography.demath_typographic` now converts the 602 spans that were never
  mathematics - a lone `$\pm$`, `$R^2$`, `$\times$`, `$\mu\text{s}$`, and any span whose whole
  content is a bare number - and it is applied by the 16 `render_*` functions that emit README
  rows *and* by the gates that compare them, so presentation has one definition instead of two
  typed copies that can disagree. Symbols and formulas keep their math. Three gates hold it:
  `test_readme_stays_inside_githubs_math_budget` (≤1,300 spans, with headroom under the
  measured cap), `test_display_math_is_never_a_paragraph_continuation` (a `$$…$$` line that
  follows text is parsed as *inline* math, which is what made the 12D state expression error
  with "'_' allowed only in math mode") and `test_no_macro_githubs_katex_build_refuses`
  (`\operatorname{sign}` → `\mathrm{sign}`; GitHub runs KaTeX with a macro allowlist).
- **Two stale pointers and one stale deferral.** Section 10.47's limitation asked for "the
  CBF machinery of 10.16" - which is the cross-stage generalization study - and is now
  pointed at 10.39, where `CBFQPLayer` lives, and at 10.51, which has since filled the
  `state x hard` cell it called unimplemented. Section 10.36 still said the OOD-with-danger
  holdout "stays deferred"; it has been delivered as Section 10.35. Section 10.49's command
  comment said "all four datasets" while `data/raw` holds seven recordings and the study
  covers four of them.
- **The README no longer duplicates an index it does not own.** Section 11.1's tree listed
  every `results/*.json` with a one-line description; `results/MANIFEST.md` is that index, is
  gated, and is owned by the writers, so the 38-line copy is gone and the tree points at it.

- Study scratch directories are per-process (`mworld_*_scratch_<pid>`). A fixed name let a
  seconds-scale smoke run delete the working directory of a multi-seed study that was
  still training in it, and the run died mid-seed on a missing parent directory.
- `pysr_available()` now verifies that the Julia runtime boots, not only that the package
  is importable, and the skipped-leg reason is split accordingly
  (`pysr_skip_reason()`): a machine whose Julia depot will not start used to publish
  "the pysr package is not installed" into the artifact, which is a statement about the
  software that the observation does not support.
- The closed-loop artifacts of `physics_injection_mpc_benchmark.py` paired every controller
  against the hand-written engine rules and against nothing else, which is the wrong
  reference for a study whose rows differ by one line of the graph. They now also publish the
  per-seed outcomes and every within-family paired contrast (`within_study_contrasts_px`).

- README Section 10.43.5 (self-correction, also recorded in Section 12): the identified
  model's horizontal-velocity residual was said to be "tile geometry and slope acceleration
  that never enter the observation". Section 10.43.8 tested that claim against the
  repository's own terrain recording and it failed - the horizontal residual did not move
  when the patch was handed to the search, while the vertical residual did. The sentence now
  states what the experiment establishes (the search separates the part of a model's form
  error that is a closed form of the observation from the part that is not) and names the one
  candidate this recording still cannot test: the slope tile class, which never occurs in
  this stage's patches. No number changed; the interpretation was stronger than the evidence.

- CI legs `Lint + tests (ubuntu, py 3.10)` and `Native tests (windows)`: the four
  fitting tests of `tests/test_symbolic_regression.py` raised
  `AttributeError: 'SymbolicRegressor' object has no attribute '_validate_data'`.
  Root cause is a transitive dependency, not this repository's code: `gplearn` 0.4.2
  calls `BaseEstimator._validate_data`, which the scikit-learn release resolved on the
  Python 3.10 leg had removed, while the 3.11/3.12 legs resolved a working version - so
  the same commit passed two matrix legs and failed two. `scikit-learn` is therefore
  declared explicitly and capped to the validated band `>=1.0.2,<1.6` in
  `pyproject.toml` and `requirements.txt` (parity preserved, `requirements.lock` already
  recorded 1.5.2), pinning every environment to the version the published 10.43 artifact
  was actually generated with. The fitting tests are the guard: an incompatible
  scikit-learn fails them at once instead of surfacing as a broken study.

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

- README Sections 4.1-4.3 and Section 10.54 (documentation correction, also recorded in
  Section 12): the audit that *found* four physics claims the recording does not support has
  been followed by rewriting the claims. §4.1's structural violation is now defined against
  the frame's initial velocity rather than $\hat v_{x,t+1}$; §4.2.1 and §4.2.5 are stated as
  an impulse range and a terminal parameter that the telemetry leaves by 7.39% (to $-112.0$)
  and 13.37% (to $+70.0$) rather than as bounds the engine enforces; §4.2.4's doubling is now
  qualified by the stratum it is actually observable in (released descent, 31.49% of 867
  frames, against released ascent still stepping $+3.0$ in 84.21% of 532); and §4.3.5 is
  rewritten from "vertical velocity is forced to zero" to the rule the data gives - the ground
  flag suppresses the gravity increment, exactly zero on 90.72% of the 2,943 grounded
  un-jumping frames, while the velocity itself is zero on 0.00% of them. The audit gained a
  per-claim satisfaction test, a stratum table for the tiers, a collision-free measure of the
  identity (exact on 96.99% of 2,755 frames), and a `prose_and_code_disagree` list, because
  correcting the documentation does not correct the code: `ResidualDynamics` and
  `ProjectedDynamics` still advance position with the predicted next velocity,
  `RolloutEvaluator` still scores $|v_x| \le 72$ and $v_y \le 64$ as absolute bounds, and
  `GroundContactConsistencyLoss` with `AnalyticalKinematicsDynamics` still zero the vertical
  velocity on a grounded frame - §4.3.5 now declares zero implementing sites, which is the
  honest statement of that state. Every corrected sentence carries its measured qualifier
  *inside* §4, emitted by `render_section_4_qualifiers` and compared against the artifact by
  the citation gate, so a rule cannot outlive its measurement again.
- README Section 10.54 (documentation correction, also recorded in Section 12): Section 4.1
  states the integration identity with $v_{x,t}$ and defines a structural violation with
  $\hat v_{x,t+1}$ two paragraphs later. Both readings are implemented in this repository -
  the penalty and the rollout predicate read the first, every hard shell since 10.27
  integrates with the second - and the telemetry decides in favour of the equation, so the
  audit now publishes the pair as a measured contradiction instead of leaving the reader to
  notice it. Two numbers are retracted along with it. (i) The kinematic-violation zeros that
  Sections 8, 10.27, 10.42 and 10.47 publish as the signature of a hard shell are
  inside-tolerance results: at 0.002 px the same arms are flagged on 0.9345-0.9578 of their
  rollout frames, and a model with neither penalty nor clamp that integrates the way the
  console does stays at exactly 0.0000 at every tolerance tried. (ii) Section 4.3.5's
  non-penetration condition, which `GroundContactConsistencyLoss` enforces, is satisfied on
  0.00% of the 2,943 grounded un-jumping frames in the training split (median $|v_y| = 6.0$
  sub-pixels/frame) under all three ways of conditioning the stratum, so that penalty term
  asks for something this recording does not show.
- README Section 10.53 narrows Section 10.47's attribution of its own `soft` cells: on a
  graph that integrates with the velocity the frame carries, the composite penalty's
  kinematic term evaluates to exactly 0.0000, so whatever those cells measured came from the
  bound and contact terms; on the `next` graphs the same term is not merely inactive but
  contradicted (7.36 px$^2$ for the DeepONet, 7.54 for the FNO), because the penalty demands
  the convention the shell refuses.

- The Section 10.6 MPC numbers were re-recorded after the episode-preamble probe showed
  the committed savestate restores into engine mode `0x08`, not interactive `0x14`
  (the preamble alone was worth a 3.4x progress difference). The gameplay-mode poke was
  unified into `SnesLibretroEmulator.start_episode()`.
- The Dyna-PPO row of the Section 10.27 master table was corrected: it had been pasted
  from the MPC row (164.75 px) instead of the recorded 115.00 px.
- The attribution Sections 10.27 and 10.42 make to a hard kinematic shell - that the
  discrete-kinematic consistency residual is identically zero *because of the clamps* -
  is corrected by Section 10.47: the property comes from the increment parameterisation,
  since every residual cell trained on its data term or behind the shell violates at
  $0.0000$-$0.0210$ while every state cell violates at $0.8470$-$0.9987$. The repository's implicit "more injected
  physics is better" is replaced by measured per-family signs (+$12.55$, +$558.99$,
  -$199.64$ px of closed-loop progress for DeepONet, FNO and MLP respectively).
- The Section 10.46 probe table reported the Physics-Constrained DeepONet ceiling as
  $25.50$ px/frame where `results/learned_structure_probe_metrics.json` measures $25.4946$.
  The cell was transcribed one rounding too high, and the quote gate had no row for it -
  every other cell of that table was reached by a neighbouring claim but not this one.
  Corrected in the prose and the table; the gate now covers all six ceilings, all six
  traction gains, the four uncovered surrogate-error rows and one tier separation, so a
  re-typed cell cannot survive again. An audit of every numeric cell in the four newest
  section tables (10.43.9, 10.44, 10.45, 10.46) against their artifacts found no others.
- Jump-impulse non-identifiability from single transitions is reported as a negative
  result (Section 10.37.1) rather than fitted away; Yoshi's Island 2 capture is reported
  as blocked with its diagnostics artifact (Section 10.36), not a fabricated state.

### Added

- `results/MANIFEST.md` artifact index with CI-enforced provenance (`_meta`), writer,
  command, README-section mapping and checkpoint-freshness gates.

[Unreleased]: https://github.com/PedroM2626/smw-pinn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PedroM2626/smw-pinn/releases/tag/v0.1.0
