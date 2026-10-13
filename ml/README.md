# ml/ — model framework (40-run DOE)

Trains, cross-validates and reports models for the ME 323 maraging-steel DOE: 4 grades
(S1–S4 = 0 / 0.5 / 1 / 2 % C), 40 runs of 2400 s (10 tracks per specimen), in-situ WLI scans
at 0/300/600/900/1200/1800/2400 s. `ml/` only *reads* `data/processed/` (or the synthetic
stand-in `data/synthetic/`).

```bash
python -m ml.synthetic                                   # synthetic inputs -> data/synthetic/
python -m ml.train --data synthetic --quick              # smoke test (< 5 min)
python -m ml.train --data real                           # after `python -m tribo_extract extract`
python -m pytest ml/tests -q
```

Results: `ml/results/<timestamp>_<data>[_quick]/` with `metrics.csv` (mean ± std per
regime × target × model × feature set × split), `metrics_folds.csv`, `predictions.csv`
(out-of-fold), `hyperparameters.csv`, `best_models.csv`, `physics_route_params.csv`,
`figures/`, `interpretation/`, `config_snapshot.yaml`, `environment.json`, `run.log`,
`report.md`.

## CLI

```
python -m ml.train [--config ml/configs/default.yaml] [--data synthetic|real]
                   [--targets endpoint wear_checkpoints ...] [--models gpr svr ...]
                   [--quick] [--n-jobs N] [--out DIR] [--no-interpret]
python -m ml.synthetic [--out data/synthetic] [--seed 323]
```

`--quick` = one gkf5 repeat, inner 3 folds, first value of each grid list.

## Data regimes

| regime | source | rows | targets | extra input |
|---|---|---|---|---|
| `endpoint` | runs_targets.csv | 40 | log10 k, log10 V_w(t_end), COF_ss_mean | – |
| `trajectory_params` | runs_targets.csv | ≤ 40 (runs with an eq.-1 fit) | log10 V̇_steady, log10 ΔV_run, log10 τ_trans | – |
| `wear_checkpoints` | wear_trajectory.csv | ≤ 240 (t > 0) | log10 V_w(t) | log10(Time_s) |
| `cof_time` | cof_timeseries.csv → 120-s bins | ≤ 800 | mean COF per bin | Time_s (bin end) |

Run-level columns are joined from runs_targets.csv on `Experiment_ID`. `status == "error"`,
`TRIAL*` and rows lacking the target are excluded. `Wear_detected == False` is censoring:
excluded by default, LOD/2 with `censoring.mode: lod_half`; the report lists the count per
target. Feature sets: `base` = Carbon_pct, Load_N, Freq_Hz (+ Hardness_HV when logged);
`physics` = base + Speed_mps, Hertz_p_mean_MPa, PV. Scaling and imputation are fitted inside
training folds only.

## Validation

* Group = `Replicate_of` if set, else `Experiment_ID` (each centre replicate pair is one group).
* `gkf5`: `StratifiedGroupKFold(5, shuffle=True)` stratified by `Specimen_ID`, one repeat per
  seed in `[0, 1, 2, 3, 4]`; every fold holds 8 runs (asserted, redrawn with the next seed
  offset otherwise).
* `loao`: one fold per held-out specimen (grade).
* Row-level regimes use the run-level folds. Tuning is nested: inner `GroupKFold(4)` on the
  training groups.
* Model selection: lowest gkf5 RMSE, Ridge excluded (reference). LOAO is reported, not used.

## Models (registry: `ml/models/registry.py`)

| model | role | settings |
|---|---|---|
| `gpr` | primary | Constant × Matérn-5/2 (ARD, bounds 1e-2–1e3) + White(1e-2, 1e-6–1), normalize_y, 5 restarts, standardised X; mean + std |
| `tabpfn` | primary benchmark | defaults, CPU, no tuning; skipped loudly ("not run") if not installed |
| `svr` | | RBF; C {0.1, 1, 10, 100}, gamma {scale, 0.01, 0.1, 1}, epsilon {0.01, 0.05, 0.1}; y standardised |
| `rf` | | 300 trees; max_depth {2, 3}, min_samples_leaf {3, 5}, max_features {1.0, sqrt} |
| `xgb` | | n_estimators {200, 500}, max_depth {2, 3}, learning_rate {0.02, 0.05}, min_child_weight {3, 5}, subsample 0.8, reg_lambda 1 |
| `ridge` | reference only | alpha {0.1, 1, 10} |

Every model runs on every regime and both feature sets. MLP, KAN, multi-task networks, PySR
and the active-learning simulation are excluded by the DOE and live in `ml/archive/`
(not imported, not tested, not registered).

**Physics route** (`ml/trajectory.py`): in every gkf5 fold the best `trajectory_params` model
(by inner CV) predicts the eq.-1 parameters of the test runs;
`tribo_extract.physics.two_stage_wear` rebuilds V_w(t), which is scored on that fold's
`wear_checkpoints` rows next to the direct models.

## Report contents

R², RMSE, MAE (mean ± std) for gkf5 and LOAO; the noise floor (pooled SD of the centre pairs
R03/R04, R13/R14, R23/R24, R33/R34) next to every endpoint RMSE; LOAO error per held-out grade;
GPR 95 % coverage and calibration. Figures: parity of the best model per target, GPR response
surface (mean, σ) over Load × Freq per carbon level, GPR ARD length scales, LOAO per-grade
errors, trajectory reconstructions for one run per grade, GPR calibration.

## Assumptions

- Run_order: `numpy.random.default_rng(323)`, one `permutation(10) + 1` per specimen in the order S1→S4, assigned to the runs in table order.
- The TRIAL row of data/run_sheet.csv is unchanged except for two empty trailing fields for the new Role and Run_order columns.
- wear_trajectory.csv is built from reports/runs/<ID>/wear_scans.csv after each successful process_run, so a stale file from an earlier extraction of the same run would also be read.
- Pre-existing OCR test failure: tesseract 5.5 truncated a colour-bar label at ×3 scale, so `_ocr_number` now tries ×4–×6 before ×3 (tribo_extract/wli.py).
- gkf5 with fewer than 40 usable runs requires floor/ceil(n/5) runs per fold; redraws use random_state = seed + 1000·k, up to 500 tries.
- Folds are built once from all usable runs; a target that excludes runs (censored, no eq.-1 fit) keeps that run→fold map, so its folds can hold fewer than 8 runs.
- cof_time bins end at ceil(Time_s/120)·120; bins with fewer than 60 s of data are dropped.
- wear_checkpoints drops the t = 0 baseline scan; censored scans are excluded or imputed exactly like censored endpoints.
- A missing Specimen_ID is filled with one specimen per carbon level.
- Hardness_HV joins both feature sets when it has any non-null value.
- With a single grid candidate (GPR, TabPFN, every model in --quick) no inner fits are run.
- GPR fits on more than 400 training rows (cof_time) are memory-bandwidth bound, so they run in at most `cv.heavy_task_slots` (3) sequential chains while other tasks fill the remaining workers; results are unchanged.
- GPR hyperparameters use L-BFGS-B with ftol 1e-6 / gtol 1e-4 instead of sklearn's ~2e-9 (same optimum to 4 digits, about 2× faster); kernel, bounds and the 5 restarts are as specified.
- The GPR log marginal likelihood and gradient are evaluated by `FastLML` (same kernel, sklearn's 1e-10 jitter, restart starting points drawn as in sklearn; matches sklearn to 1e-6 / 1e-3, tested) and the result is fitted by sklearn with `optimizer=None`; about 3× faster.
- Physics route: each selectable model is tuned on the inner folds and the one with the lowest inner RMSE at its best grid point is chosen (tuning and choice share the inner folds); it predicts every test run that has checkpoint scans, including runs without an eq.-1 fit.
- The physics route is reported next to the direct models but is not eligible as "best model".
- Noise floor = sqrt(mean over pairs of (a − b)²/2) on the modelled scale (log10 for wear); a pair is skipped if either run lacks the target.
- GPR response surfaces use the base feature set and a GPR fitted on all usable runs (description, not evaluation); σ excludes the fitted white noise, whose SD is printed in the title.
- LOAO per-grade error bars are 95 % bootstrap intervals (200 resamples) of the RMSE over the held-out grade's runs, each model at its best LOAO feature set.
- Trajectory figure runs are the anchor_center run of each specimen (R03, R13, R23, R33), out-of-fold from gkf5 repeat 0.
- Coverage is reported for every model with a predictive std (GPR, TabPFN).
- The "unseen operating points" final refit was removed with the holdout logic, so results folders no longer contain models/.
- requirements.txt keeps torch because TabPFN depends on it; tabpfn itself stays an optional commented line.
- Synthetic data: the four inputs are runs_targets.csv, cof_timeseries.csv, wear_trajectory.csv and run_sheet.csv; COF is simulated at 10 Hz before extract_cof's 1 Hz binning; τ_trans is planted at about 200–450 s and replicate pairs share one specimen hardness.
