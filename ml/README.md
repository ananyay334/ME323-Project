# ml/ — models for the 304 SS dry-reciprocating DOE

`ml/` reads `data/processed/runs_checkpoints.csv`, plus `cof_cycles.csv` for one figure, or the
synthetic stand-in in `data/synthetic/`. Assumptions are listed in the project `README.md` and
copied into every report.

```bash
python -m ml.synthetic                                   # synthetic inputs -> data/synthetic/
python -m ml.train --data synthetic --quick              # smoke test (< 1 min)
python -m ml.train                                       # CV on the training runs (real data)
python -m ml.topup --n 4                                 # top-up runs -> data/topup_candidates.csv
python -m ml.train --final                               # + test set R24-R29, scored once
python -m pytest ml/tests -q
```

## Data

- **Rows.** One row per run × checkpoint, keeping only `status != error` and `Include == Y`.
- **Training runs.** Roles completed, corner, interior, center_replicate and topup: R00–R23
  plus top-ups, 24 × 2 = 48 rows.
- **Test runs.** Role `test` (R24–R29). They are loaded only by `--final`.
- **Inputs.** ln(Load_N), ln(Freq_Hz), ln(Time_s), from nominal values, standardised inside
  the folds.
- **Targets.** log10(k_mm3_per_Nm) and COF_steady. A row missing one target still trains the
  other.
- **Censoring.** `Wear_detected == False` is censored: log10 k is excluded by default, or set
  to LOD/2 with `censoring.mode: lod_half`.

## Validation

- **Groups.** The group is `Replicate_of` if set, else `Experiment_ID`. R04/R14/R20 form one
  group, and both checkpoints of a run share their run's group.
- **Outer CV.** 5-fold `GroupKFold` with the groups shuffled by seed, repeated for seeds
  0–4, on the training rows only.
- **Inner tuning.** `GroupKFold(4)` on the outer training groups.
- **Selection.** Lowest mean CV RMSE per target; Ridge is a reference and is never selected.
- **`--final`.** Refits the selected model per target on all training rows, scores the test
  rows once and writes `ml/results/final/`. It refuses to run again unless `--force` is given.

## Models (`ml/models/registry.py`)

| model | settings |
|---|---|
| `ann` | numpy (default) or PyTorch (`backend: torch`), 3 → {3, 4} tanh → 2 outputs (log10 k or log10 width, COF), masked MSE on standardised targets, Adam lr 1e-2, weight decay {1e-4, 1e-3, 1e-2}, ≤ 3000 epochs, early stopping (patience 200) on 20 % of the training groups with best weights restored, 20-seed ensemble → mean ± SD; ≤ 26 weights |
| `gpr` | Constant × Matérn-5/2 (ARD, length scales 1e-2–1e3) + White(1e-2, 1e-6–1), normalize_y, 5 restarts |
| `svr` | RBF; C {0.1, 1, 10, 100}, gamma {scale, 0.1, 1}, epsilon {0.01, 0.05, 0.1} |
| `tabpfn` | TabPFNRegressor defaults, CPU; skipped loudly ("not run") if not installed |
| `xgb` | max_depth {1, 2}, n_estimators {200, 400}, learning_rate {0.03, 0.05}, min_child_weight {2, 4}, subsample 0.8 |
| `ridge` | alpha {0.1, 1, 10}; reference; its exponents d target / d ln x are reported with group-bootstrap 95 % CIs |

`--quick` runs one outer repeat, inner 3 folds, the first value of each grid, and an ANN
ensemble of 3. The ANN and the GPR also drive `ml.topup`.

## Report (`ml/results/<timestamp>/report.md`)

- **Metrics.** R², RMSE and MAE per target × model (mean ± SD over folds × repeats), with the
  noise floor (pooled SD of R04/R14/R20) next to the RMSEs, and 95 % interval coverage.
- **Which factor controls COF and wear.** GPR ARD length scales, group-permutation importance
  of the best model, and the Ridge exponents.
- **Figures.** CV parity, GPR mean ± σ over ln L × ln f at t = 60 and 120 s, and per-cycle
  COF of the replicates.
- **Top-ups.** Pre-registered predicted vs measured values for top-up runs.
- **`--final` only.** The test-set table and parity plot.

The output files are `metrics.csv`, `metrics_folds.csv`, `predictions.csv`,
`hyperparameters.csv`, `best_models.csv`, `importance_folds.csv`, `interpretation/`,
`figures/`, `config_snapshot.yaml`, `environment.json`, `run.log` and `report.md`.

## Archive

`ml/archive/` (not imported, not tested, not registered) holds:
- the 100-run models (MLP, KAN, multi-task, PySR) and the active-learning simulation;
- under `doe40/`: the 40-run DOE's carbon/specimen features, the stratified group K-fold by
  specimen and leave-one-alloy-out splits, the trajectory-parameter / physics-route regimes,
  and their tests.
