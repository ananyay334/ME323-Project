# ml/ — model framework

Trains, evaluates and reports models for **steady-state COF**, **COF(t)** and the
**specific wear rate** (log10 k) of carbon-modified maraging steel, from the tables
written by `tribo_extract`. `ml/` only *reads* `data/processed/` (or its synthetic
stand-in `data/synthetic/`). It never writes there.

```bash
python -m ml.synthetic                                   # synthetic tables -> data/synthetic/
python -m ml.train --data synthetic --quick              # smoke test, ~2-3 min on a laptop
python -m pytest ml/tests -q                             # ~1.5 min
```

When the campaign data has been extracted:

```bash
python -m tribo_extract extract                          # -> data/processed/*.csv
# put the unseen operating points under data.real.holdout_ids in ml/configs/default.yaml
python -m ml.train --data real                           # full run, ~20 min (timings below)
```

Everything lands in `ml/results/<timestamp>_<data>[_quick]/report.md`. Switching
from `--data synthetic` to `--data real` needs no code changes.

## Checklist for when the real data arrives

1. `python -m tribo_extract extract`, then look at `status`, `issues`, `*_flags` in
   `data/processed/runs_targets.csv`.
2. Make sure every replicate run has `Replicate_of` filled in the run sheet. Otherwise
   it is treated as an independent group, and the loader warns about conditions run twice
   without it.
3. List the 5–10 unseen operating points under `data.real.holdout_ids`. The loader
   warns about off-grid runs that are not listed.
4. `python -m ml.train --data real --quick` as a smoke test, then the full run.

## CLI

```
python -m ml.train --config ml/configs/default.yaml [--data synthetic|real]
                   [--targets wear cof] [--models gpr kan ...] [--quick]
                   [--n-jobs N] [--out DIR] [--no-interpret] [--no-al] [--no-trajectory]
python -m ml.active_learning --config ml/configs/default.yaml [--data ...] [--quick]
python -m ml.synthetic [--out data/synthetic] [--seed 323]
```

* `--targets` takes families (`wear` = `log10_k`, `log10_V`; `cof` = `COF_ss_mean`,
  `COF_t`) or target names.
* `--models` restricts to registry names: `linear ridge rf xgb svr gpr mlp kan mt_mlp mt_kan tabpfn pysr`.
* `--quick` merges the config's `quick:` block over it: 1 repeat, inner 2-fold,
  3 candidates per model, fewer epochs, ensembles and AL repetitions. It is a smoke test,
  not a result.

## Layout

```
ml/
  configs/default.yaml  every setting: data paths, holdout IDs, censoring mode, feature sets,
                        models on/off + search spaces, CV folds/repeats, AL, trajectory, quick
  config.py             config loading, seeds, logging, run snapshot
  data.py               load + clean tables, target definitions, censoring, holdout split
  features.py           feature sets, PV / log1p(time), leakage guards, physics-consistent inputs
  splits.py             groups (Replicate_of) and deterministic nested group K-fold
  metrics.py            R², RMSE, MAE, linear-k errors, interval coverage, calibration
  tuning.py             search spaces, inner-CV tuning
  evaluate.py           nested CV, joint (multi-task) CV, final refit + unseen points
  models/  registry.py  linear.py trees.py svr.py gpr.py mlp.py kan.py multitask.py
           tabpfn_model.py symbolic.py _torch.py (shared NN training)
  interpret.py          SHAP, partial dependence, KAN functions, GPR maps, Archard check
  active_learning.py    Extension A (also a CLI)
  trajectory.py         Extension B
  plotting.py           figure style (validated palette)
  report.py             report.md
  synthetic.py          synthetic tables (CLI)
  train.py              CLI entry point
  tests/                schema, leakage, reproducibility, models, synthetic recovery, CLI
  results/<timestamp>/  (gitignored) metrics.csv, metrics_folds.csv, predictions.csv,
                        predictions_timeseries.csv.gz, hyperparameters.csv, holdout_*.csv,
                        final_selection.csv, figures/, models/, interpretation/,
                        active_learning/, trajectory/, config_snapshot.yaml,
                        environment.json, run.log, report.md
```

## Inputs and data rules

| table | grain | used for |
|---|---|---|
| `runs_targets.csv` | 1 row per run | `log10_k`, `log10_V`, `COF_ss_mean`, eq.-4 trajectory parameters |
| `cof_timeseries.csv` | 1 row per second | `COF_t` (only `Experiment_ID, Time_s, COF, Phase` are read; run inputs are joined from the runs table) |
| `wear_checkpoints.csv` | optional, 1 row per scan | Extension B comparison (`Experiment_ID, t_s, Wear_Volume_mm3`) |

* Drop `status == "error"` and IDs starting with `TRIAL`. `partial` rows are kept and
  used only for the targets they have.
* Absent or all-NaN input columns are dropped and logged. Rows missing carbon, load or
  frequency are dropped and logged. Other missing inputs (e.g. one unmeasured hardness)
  are median-imputed inside each training fold.
* `Wear_detected == False` means censored, not zero. By default these runs are excluded
  from the wear targets (`censoring.mode: exclude`). A sensitivity run imputes LOD/2
  (`censoring.sensitivity`).
* Wear is modelled as log10. COF is modelled untransformed.
* `Zdepth_um` is discarded on load. A guard (`features.check_inputs`) rejects it, any
  target column, and any duration proxy (`Time_s`, `sliding_time_s`,
  `Sliding_distance_m`, `N_cycles`) as an input for the per-run targets.

## Leakage rules (enforced and tested)

* **Group K-fold everywhere.** Group = `Replicate_of` (followed to the root of a
  chain), else `Experiment_ID`. Folds are assigned per group from a seeded shuffle,
  stratified by carbon level. The same group→fold map serves every target, feature set
  and model, so comparisons are paired and the seconds of a run share its fold.
* **Nested CV.** Hyperparameters are chosen on inner group folds inside the outer
  training set. Outer folds are only for reporting and model selection.
* **In-fold preprocessing.** Imputers, scalers, the collinearity filter and the NN
  standardisation are all fitted inside the estimator on the training rows only.
* **Unseen points.** They are removed in `load_data` before any split exists. A holdout
  ID takes its whole replicate group with it.
* Tests: `test_leakage.py` asserts disjoint groups in every outer and inner split. It also
  runs a canary: a random value per run is unlearnable under group CV (R² < 0.1) but
  learnable under a row-level split (R² > 0.8), which shows the test can detect leakage.

## Features

`base` = carbon, hardness, load, frequency. `physics` = base + speed, Hertz mean
pressure, PV, plus Ra / ambient T / RH when logged for ≥ 80 % of runs. Both sets are
run for every model. At fixed stroke, speed is an exact multiple of frequency: plain OLS
drops it in-fold (`DropCollinear`), and the other models are regularised. COF(t) adds
`Time_s` and `log1p(Time_s)` (`features.time_encoding`).

## Models

| name | targets | notes |
|---|---|---|
| `linear` | all | OLS, collinear columns dropped in-fold |
| `ridge` | all | alpha tuned |
| `rf`, `xgb` | all | shallow trees, min leaf size / child weight, L1/L2; tuned |
| `svr` | all | RBF; C, γ, ε tuned (y standardised) |
| `gpr` | all (primary for wear) | ARD Matérn-5/2 + white noise, normalize_y; hyperparameters by marginal likelihood; predictive std |
| `mlp` | COF(t), wear | 1–2 layers < 32 units, AdamW weight decay, dropout, group-aware early stopping, 5-seed ensemble |
| `kan` | COF(t), wear, trajectory | [n_in, 4–8, 1], cubic B-spline edges, grid + L1/entropy reg. tuned; learned functions plotted |
| `mt_mlp`, `mt_kan` | COF(t) + log10 k jointly | shared trunk → wear head (once per run) + COF head (trunk embedding + time); uncertainty-weighted or fixed weights (tuned) |
| `tabpfn` | wear | optional (`pip install tabpfn`) |
| `pysr` | wear | optional, off by default (`pip install pysr` + Julia) |

Missing optional packages mean the model is skipped with a warning and listed in the
report. CPU only. Torch runs single-threaded per task, and the parallelism is across CV
tasks (joblib).

**Adding a model:** write `ml/models/<name>.py` with
`SPECS = [ModelSpec("<name>", build_fn, ...)]`, where `build_fn(params, seed, ctx)`
returns an sklearn-style estimator with `fit/predict` and optionally `predict_std`. Then
add the module to `MODEL_MODULES` in `models/registry.py` and give it a
`models: <name>:` block in the config. Splits, tuning, metrics, final test and report
need no changes.

## Evaluation and outputs

* Per target: R², RMSE, MAE on outer folds (mean ± std over folds and repeats), plus the
  pooled out-of-fold R². For log10 targets there are also linear-k errors (RMSE, MAE,
  median % error, multiplicative factor).
* For models with a std (GPR, NN ensembles, TabPFN): 95 % coverage, interval width,
  NLL, and calibration curves.
* COF(t) is scored per second, and as `COF_t@ss`, the predicted curve averaged over each
  held-out run's steady phase, comparable with `COF_ss_mean`.
* **Final test.** For each target, the `top_k` (model, feature set) by CV RMSE plus
  `final.always_include` / `interpret.refit_models` are re-tuned by group CV on all CV
  data, refit, and used to predict the unseen points. Intervals are Gaussian where a
  model has a std, and otherwise the 95 % quantile of that model's out-of-fold errors.
* **Interpretation.** SHAP (trees exact, others model-agnostic); partial dependence of
  load (per carbon level), carbon and frequency, with a linear-vs-curved verdict; KAN
  learned functions; GPR mean/uncertainty maps over load × frequency per carbon level;
  an Archard exponent check (log10 V on log10 F, log10 S).
* **Extension A** (`active_learning.py`). Pool = grid conditions. The seed is a maximin
  LHS snapped to the pool; GPR surrogate; variance or UCB acquisition vs random; stop at
  the budget or when max σ < ε. Learning curves over ≥ 20 repetitions, and N_AL vs the
  full-grid model within a tolerance.
* **Extension B** (`trajectory.py`). log10 of the eq.-4 parameters is modelled by group
  CV, and V(t) is rebuilt with `tribo_extract.physics.two_stage_wear` against the
  measured checkpoints. The module does nothing when those columns are absent.

## Assumptions (also listed in every report)

1. **Groups and replicates.** A replicate is identified only by `Replicate_of`. A
   condition run twice without it is treated as two groups (with a warning).
2. **Selection.** Only by held-out outer-fold RMSE. Ties are not broken by complexity.
3. **Censoring sensitivity.** The LOD/2 run is scored on detected runs only, so it is
   comparable with the main run. The report adds the fraction of censored runs predicted
   below their LOD.
4. **COF(t) training rows.** Models are fitted on `time_regime.train_points_per_run`
   seconds per run (half log-spaced, so the running-in is resolved), and kernel methods
   on fewer (`max_points_per_run`). All seconds are scored. `COF_t@ss` uses the
   extraction's `Phase` labels.
5. **Uncertainty.** NN ensemble std = sqrt(ensemble variance + mean validation MSE),
   i.e. epistemic plus an aleatoric estimate. GPR std includes the fitted noise. The GPR
   uncertainty maps and the AL acquisition use the *latent* (noise-free) std.
6. **Physically consistent inputs.** Partial dependence, maps and AL set carbon, load and
   frequency, and recompute speed and Hertz pressure with `tribo_extract.physics`.
   Hardness follows the mean of its carbon level. SHAP with an independent masker does
   not respect this coupling, so read it as "shared credit" between carbon and hardness.
7. **KAN.** A native torch implementation of the efficient-KAN layer, so only torch is
   needed. `models.kan.params.backend: efficient_kan` switches to that package if it is
   installed. pykan is not wired in: its training API differs and changes between
   versions.
8. **AL design variables.** Carbon, load and frequency on the nominal grid (hardness is
   measured, not chosen). Replicates are averaged per condition. The random baseline
   starts from the same LHS seed (`random_seed_design: shared`) so that only the
   acquisition differs. The stop point is recorded, but curves continue to the budget.
9. **Extension B QC rule.** Parameters pinned at a bound of the extraction's fit
   (τ ≤ 0.01 s, or V̇ / ΔV more than 3 decades below the median) are excluded from that
   parameter's model and counted.
10. **Synthetic data** (`ml/synthetic.py`) runs through `tribo_extract.cof.extract_cof`
    and `physics.fit_wear_trajectory`, so its targets follow the pipeline's definitions.
    The planted effects are in `data/synthetic/planted_effects.json`. It also writes
    `wear_checkpoints.csv`, a table the extraction does not produce yet.

## Findings from the synthetic dry run that matter for the campaign

* **τ_trans is not identifiable from checkpoints at 0/60/150/300/450/600 s** when the
  running-in lasts tens of seconds. Only one scan falls inside the transient, so the eq.-4
  fit often pins τ at its bound. V̇_steady and ΔV_run are recovered well, and so is the
  curve as a whole. Adding scans at about 15 and 30 s would fix τ_trans.
* For a direct V(t) comparison in Extension B, the per-scan volumes (currently in
  `reports/runs/<ID>/wear_scans.csv`) would need exporting to
  `data/processed/wear_checkpoints.csv`. Without that table the module compares against
  the measured eq.-4 fit instead.

## Timings (Apple-silicon laptop, 8 cores)

| command | time |
|---|---|
| `python -m ml.synthetic` | ~4 s |
| `python -m ml.train --data synthetic --quick` | ~2–3 min |
| `python -m ml.train --data synthetic` (full: 4 targets × 2 feature sets × 10 models, 5×2 outer folds, nested tuning, sensitivity, final test, interpretation, AL with 20 repetitions, trajectory) | ~20 min |
| `python -m pytest ml/tests -q` | ~1.5 min |

In the full run about a third of the CV compute goes to the COF(t) models (kernel methods
and networks on per-second rows). `--targets wear` or `--models ...` give faster focused
runs.

## Dependencies

Core: numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, joblib, torch (MLP, KAN,
multi-task), xgboost, shap. Optional: `tabpfn`, `pysr` (+ Julia), `efficient_kan`. Each
is skipped with a warning if missing. See `requirements.txt`.

**OpenMP note (macOS).** torch and xgboost each bundle their own OpenMP runtime. Two
runtimes running multi-threaded in one process segfault (reproduced with the conda
torch 2.12 / xgboost 3.1 builds). `import ml` therefore sets `OMP_NUM_THREADS=1` unless
you set it yourself. This costs nothing here, because the parallelism is across CV tasks
(joblib) and every model fits single-threaded. In a notebook, `import ml` before torch or
xgboost so that the cap takes effect. A regression test covers this
(`test_torch_and_xgboost_coexist_in_one_process`).

Plotting uses pyplot-free `Figure` objects inside an `rc_context`, so importing the
package changes neither the matplotlib backend nor your global style.
