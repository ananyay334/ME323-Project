# ml/ — model framework (to be built)

Inputs come only from `data/processed/`, produced by `python -m tribo_extract extract`:

| table | grain | use |
|---|---|---|
| `runs_targets.csv` | 1 row per `Experiment_ID` | wear-rate head / GPR, SVR, TabPFN (N ≈ 100) |
| `cof_timeseries.csv` | 1 row per second of sliding | COF(t) head (≈ 600 rows × 100 runs) |

Contract:
* Features: `Carbon_pct, Hardness_HV, Load_N, Freq_Hz, Speed_mps, Hertz_p_mean_MPa`
  (+ `Time_s` for COF(t); + `Initial_Ra_um`, `Ambient_T_C`, `RH_pct` when logged).
* Targets: `COF_ss_mean`, `COF` (per second), `Specific_Wear_Rate_mm3_per_Nm`
  (use log10; rows with `Wear_detected == False` are censored at `…_LOD_…`).
* Rows with `status == "error"` are excluded; `issues` / `*_flags` columns are for QC.
* Cross-validation: `GroupKFold` on `Experiment_ID` (never split one run's seconds
  across train/test).
