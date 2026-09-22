# Results Manifest

Every number quoted in `README.md` comes from one of the JSON artifacts in this
directory. This file is the index that connects **artifact → the module that writes
it → the command that regenerates it → the README section that quotes it**, and
`tests/test_results_manifest.py` fails CI if any of those four links is broken
(an artifact with no owner, a command pointing at a module that never writes it, or
a file that exists but is not catalogued).

Checkpoints (`checkpoints*/`), figures (`figures/`) and run logs (`../runs/`) are
outputs too, but they are byte-artifacts rather than claims: only the JSON metric
files are gated here.

## Regenerating everything

```bash
# Hardware-free artifacts (safe on CI, CPU only):
python -m src.training.benchmark_experiment --config configs/benchmark.yaml
python -m src.evaluation.sample_efficiency_benchmark --config configs/sample_efficiency.yaml
python -m src.evaluation.multiseed_benchmark --config configs/multiseed.yaml
python -m src.evaluation.analytical_baselines --no-hardware
python -m src.training.train_pixel_estimator
python -m src.training.train_terminal_value
python -m src.evaluation.spatial_holdout_benchmark
python -m src.evaluation.plot_learning_curves

# Emulator-in-the-loop artifacts (need the Libretro core + your own ROM dump;
# see README section 11.2 for the SHA-1 and the SMW_ROM override):
python -m src.evaluation.mbrl_mpc_benchmark
python -m src.evaluation.mbrl_mpc_benchmark --reproduction-check
python -m src.evaluation.analytical_baselines --hardware
python -m src.evaluation.evaluate_pixel_mpc
python -m src.evaluation.evaluate_hierarchical_mpc
python -m src.evaluation.evaluate_policy_snes
# ... the remaining closed-loop commands are listed in README section 11.5.
```

On Windows, where `make` is usually unavailable, the same workflows are reachable
through the installed console script: `smw-pinn benchmark`, `smw-pinn multiseed`,
`smw-pinn baselines`, or generically `smw-pinn run src.evaluation.<module>`.

## Metric artifacts

| Artifact | Written by | Requires emulator | README section |
| :--- | :--- | :---: | :--- |
| `benchmark_metrics.json` | `src/training/benchmark_experiment.py` | no | 8.1-8.3, 10.1 |
| `sample_efficiency_metrics.json` | `src/evaluation/sample_efficiency_benchmark.py` | no | 8.3, 10.1 |
| `multiseed_benchmark_metrics.json` | `src/evaluation/multiseed_benchmark.py` | no | 8.4, 10.1 |
| `analytical_baseline_metrics.json` | `src/evaluation/analytical_baselines.py` | no | 10.37 |
| `oracle_mpc_metrics.json` | `src/evaluation/analytical_baselines.py` | yes | 10.37 |
| `mbrl_mpc_metrics.json` | `src/evaluation/mbrl_mpc_benchmark.py` | yes | 10.6 |
| `mpc_reproduction_metrics.json` | `src/evaluation/mbrl_mpc_benchmark.py --reproduction-check` | yes | 10.38 |
| `mpc_preamble_probe_metrics.json` | `src/evaluation/mbrl_mpc_benchmark.py --preamble-probe` | yes | 10.38 |
| `dyna_ppo_metrics.json` | `src/evaluation/evaluate_policy_snes.py` | yes | 10.7 |
| `sprite_perception_metrics.json` | `src/evaluation/evaluate_sprites_snes.py` | yes | 10.8 |
| `model_free_ppo_metrics.json` | `src/training/model_free_ppo.py` | yes | 10.9 |
| `pinn_ensemble_metrics.json` | `src/models/pinn_ensemble.py` | no | 10.10 |
| `online_mbpo_metrics.json` | `src/training/online_mbpo.py` | yes | 10.12 |
| `dyna_ppo_multi_entity_metrics.json` | `src/training/dyna_ppo_sprites.py` | yes | 10.13 |
| `online_mbpo_safe_metrics.json` | `src/training/online_mbpo.py --safe` | yes | 10.14 |
| `computational_profiling_metrics.json` | `src/evaluation/benchmark_computational_efficiency.py` | yes | 10.15 |
| `cross_level_generalization_metrics.json` | `src/evaluation/cross_level_benchmark.py` | yes | 10.16 |
| `set_multi_entity_metrics.json` | `src/training/train_set_multi_entity.py` | no | 10.22 |
| `distilled_policy_metrics.json` | `src/evaluation/evaluate_distilled_policy_snes.py` | yes | 10.23 |
| `extended_navigation_metrics.json` | `src/evaluation/evaluate_extended_navigation.py` | yes | 10.24 |
| `dagger_training_metrics.json` | `src/training/train_dagger.py` | yes | 10.25 |
| `dagger_policy_metrics.json` | `src/evaluation/evaluate_dagger_snes.py` | yes | 10.25 |
| `ablation_benchmark_metrics.json` | `src/evaluation/ablation_benchmark.py` | no | 10.26 |
| `cross_level_control_metrics.json` | `src/evaluation/evaluate_cross_level_control.py` | yes | 10.28 |
| `unified_ppo_metrics.json` | `src/training/train_unified_ppo.py` | no | 10.29.3 |
| `full_level_clearance_metrics.json` | `src/evaluation/evaluate_full_level_clearance.py` | yes | 10.29.4 |
| `full_level_trajectory_log.json` | `src/evaluation/evaluate_full_level_clearance.py` | yes | 10.29.4, 10.33 |
| `multi_entity_mpc_metrics.json` | `src/evaluation/evaluate_multi_entity_mpc.py` | yes | 10.19 |
| `multi_entity_hardware_metrics.json` | `src/evaluation/evaluate_multi_entity_snes.py` | yes | 10.19 |
| `tilemap_benchmark_metrics.json` | `src/training/train_tilemap.py` | no | 10.21 |
| `tilemap_mpc_metrics.json` | `src/evaluation/evaluate_tilemap_mpc.py` | yes | 10.34 |
| `unified_joint_metrics.json` | `src/training/train_unified_multimodal.py` | no | 10.34 |
| `mpc_reflex_ablation.json` | `src/evaluation/mpc_reflex_ablation.py` | yes | 10.33 |
| `terminal_value_metrics.json` | `src/training/train_terminal_value.py` | no | 10.33 |
| `learning_curve_metrics.json` | `src/evaluation/plot_learning_curves.py` | no | 10.35 |
| `spatial_holdout_metrics.json` | `src/evaluation/spatial_holdout_benchmark.py` | no | 10.35 |
| `pixel_estimator_metrics.json` | `src/training/train_pixel_estimator.py` | no | 10.31 |
| `pixel_mpc_metrics.json` | `src/evaluation/evaluate_pixel_mpc.py` | yes | 10.31 |
| `hierarchical_mpc_metrics.json` | `src/evaluation/evaluate_hierarchical_mpc.py` | yes | 10.32 |
| `yi2_capture_attempt.json` | `scripts/navigate_to_level.py` | yes | 10.36 |
| `obstacle_1000_diagnosis.json` | `src/evaluation/diagnose_obstacle_1000.py` | yes | 10.24 (X~1000 plateau) |
| `piml_mfrl_metrics.json` | `src/evaluation/piml_mfrl_study.py` | yes | 10.39.1 |

## Input freshness

A result file is only as current as the checkpoint it was recorded with. Each row
below declares the checkpoints a closed-loop artifact loads; the CI gate
`tests/test_results_manifest.py::test_declared_inputs_are_not_newer_than_the_result`
compares the git commit date of the artifact against the commit date of every
dependency and fails when they disagree with the `STALE` marker. Rows marked
`STALE` predate a checkpoint regeneration and are awaiting a hardware re-run; the
marker must be removed by actually re-running, never by editing. This gate is how
the 10.6 rows were found to have been recorded before the checkpoints were
regenerated (see README 10.38).

Commit dates only exist if the checkout carries history, so CI clones with
`fetch-depth: 0`; a shallow or single-commit clone skips this one gate instead of
judging every dependency pair a tie.

```freshness
      mbrl_mpc_metrics.json         <- pinn_hard_best.pt pinn_soft_best.pt mlp_best.pt
      oracle_mpc_metrics.json       <- pinn_hard_best.pt
      mpc_reproduction_metrics.json <- pinn_hard_best.pt pinn_soft_best.pt mlp_best.pt
      mpc_preamble_probe_metrics.json <- pinn_hard_best.pt
      hierarchical_mpc_metrics.json <- pinn_hard_best.pt
      pixel_mpc_metrics.json        <- pinn_hard_best.pt
      analytical_baseline_metrics.json <- pinn_hard_best.pt
      dyna_ppo_metrics.json           <- pinn_hard_best.pt
STALE cross_level_control_metrics.json <- pinn_hard_best.pt mlp_best.pt pinn_soft_best.pt dagger_policy_best.pt
```

`STALE` rows are the honest backlog: `mbrl_mpc_metrics.json`,
`oracle_mpc_metrics.json` and the §10.31-§10.32 rows were re-recorded on 2026-09-22
with the corrected `start_episode()` preamble (README 10.38.1). On the same date
`dyna_ppo_metrics.json` was re-run against the regenerated policy checkpoints: the
Hard-PINN headline row reproduced exactly (+115.0 px / 173 frames, Section 10.7.2), the
MLP and random rows were refreshed and the README table updated, so its `STALE` marker
was dropped. `cross_level_control_metrics.json` is deliberately left `STALE`: a
re-record confirmed its DAgger row reproduces (830.50 px / 34.66) but the MPC and random
rows are single-seed CEM draws and its latency/throughput columns are wall-clock, so
overwriting the published table from one loaded-machine draw would reduce fidelity
(cf. README 10.38.2). `evaluate_cross_level_control.py` now pins `set_global_seed` so a
clean, unloaded, multi-seed re-record is reproducible; until that re-record replaces the
table, the marker stays. The rule is unchanged: resolve a marker by re-running, never by
editing it alone.

## Provenance policy

Artifacts written from now on embed a `_meta` block (UTC timestamp, git SHA and
working-tree state, Python/NumPy/PyTorch versions, CUDA device, seed and the
regenerating command) via `src/utils/provenance.py:write_metrics`. Files produced
before that convention exists carry no `_meta`; they are grandfathered below and
the list may only shrink, never grow:

```text
ablation_benchmark_metrics.json        multiseed_benchmark_metrics.json
benchmark_metrics.json                 multi_entity_hardware_metrics.json
computational_profiling_metrics.json   multi_entity_mpc_metrics.json
cross_level_control_metrics.json       obstacle_1000_diagnosis.json
cross_level_generalization_metrics.json online_mbpo_metrics.json
dagger_policy_metrics.json             online_mbpo_safe_metrics.json
dagger_training_metrics.json           pinn_ensemble_metrics.json
distilled_policy_metrics.json          sample_efficiency_metrics.json
dyna_ppo_metrics.json                  set_multi_entity_metrics.json
dyna_ppo_multi_entity_metrics.json     spatial_holdout_metrics.json
extended_navigation_metrics.json       sprite_perception_metrics.json
full_level_clearance_metrics.json      terminal_value_metrics.json
full_level_trajectory_log.json         tilemap_benchmark_metrics.json
learning_curve_metrics.json            tilemap_mpc_metrics.json
model_free_ppo_metrics.json            unified_joint_metrics.json
mpc_reflex_ablation.json               unified_ppo_metrics.json
```

`mbrl_mpc_metrics.json` used to appear above; it was regenerated under the
provenance policy and must keep its `_meta`. The artifacts written by
`analytical_baselines.py`, `mbrl_mpc_benchmark.py`, `train_pixel_estimator.py`,
`evaluate_pixel_mpc.py`, `evaluate_hierarchical_mpc.py` and
`scripts/navigate_to_level.py` are new under that policy and must always carry
`_meta`; they are therefore absent from the allowance above.
