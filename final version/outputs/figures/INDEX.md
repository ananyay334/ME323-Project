# Figure index

Figures are numbered in slide order. Each exists as PNG (300 dpi) and SVG.

| # | file | caption | data source |
|---|---|---|---|
| 1 | [fig01_pipeline.png](fig01_pipeline.png) | Pipeline overview: raw files -> extraction -> dataset -> models -> evaluation -> outputs. | code structure (src/tribo_final) |
| 2 | [fig02_test_matrix.png](fig02_test_matrix.png) | Test matrix of the 23 runs in L-f-t space; replicates ringed, missing DOE rows hollow. | outputs/data/runs.csv, inputs/run_sheet.csv |
| 3 | [fig03_raw_signal.png](fig03_raw_signal.png) | R04 raw Fz and Fx: static step, trimmed ramps, Fx offset, reversals and the per-cycle half-range COF. | testdata/R04/R04.csv |
| 4 | [fig04_cof_time.png](fig04_cof_time.png) | COF vs sliding time for all 23 runs (small multiples by load); first-10-s window shaded. | outputs/data/per_cycle_cof.csv |
| 5 | [fig05_height_map.png](fig05_height_map.png) | Height map R04 (m) with core, band and reference rows; across-track profile with groove and pile-up shading. | testdata/R04/R04.bcrf |
| 6 | [fig06_robust_levelling.png](fig06_robust_levelling.png) | Robust vs plain levelling on R24 (m), a scan with a deep pit. | testdata/R24/R24.bcrf |
| 7 | [fig07_wear_per_run.png](fig07_wear_per_run.png) | Specific wear rate k per run vs load with below-LOD upper bounds and not-measurable runs. | outputs/data/runs.csv |
| 8 | [fig08_c1_cof_definitions.png](fig08_c1_cof_definitions.png) | C1: COF definitions compared by replicate SD and physics-law R² (P1, P2). | outputs/tables/c1_cof_definitions.csv |
| 9 | [fig09_c2_wear_measures.png](fig09_c2_wear_measures.png) | C2: wear measures compared (runs measurable, replicate SD, best R² in P1/P2). | outputs/tables/c2_wear_measures.csv |
| 10 | [fig10_physics_fits.png](fig10_physics_fits.png) | Physics fits: COF_first10s and COF_end vs load (μ₀ + c·L^(-1/3)), track width vs load (log-log), wear volume vs L·S (Archard, slope 1), each with a 95 % run-bootstrap band. | outputs/data/runs.csv, outputs/tables/final_parameters.csv |
| 11 | [fig11_c3_models.png](fig11_c3_models.png) | C3: P1 (filled) and P2 (hollow) R² for every model and target, with constant and physics reference lines. | outputs/tables/metrics_all.csv |
| 12 | [fig12_stability.png](fig12_stability.png) | Stability: P1 vs P2 R² for every model x target; points in the shaded quadrant are unstable. | outputs/tables/metrics_all.csv |
| 13 | [fig13_parity.png](fig13_parity.png) | Parity plots (P1 out-of-fold) of the final model per target with 95 % prediction intervals. | outputs/data/oof_predictions.csv |
| 14 | [fig14_residuals.png](fig14_residuals.png) | Residuals of the final models vs load, frequency and time (P1 out-of-fold). | outputs/data/oof_predictions.csv |
| 15 | [fig15_confusion.png](fig15_confusion.png) | Tercile confusion matrices (P1): final vs old pipeline vs best plain ANN, for COF and wear. | outputs/data/oof_predictions.csv, outputs/data/c4_oof.csv |
| 16 | [fig16_uncertainty.png](fig16_uncertainty.png) | Uncertainty: 95 % interval coverage and relative width per model (P1). | outputs/tables/metrics_all.csv |
| 17 | [fig17_interpretation.png](fig17_interpretation.png) | Interpretation: GPR length scales and ANN permutation importance per input and target. | outputs/tables/gpr_length_scales.csv, ann_permutation_importance.csv |
| 18 | [fig18_surfaces.png](fig18_surfaces.png) | Prediction surfaces (mean and SD) of the selected COF and width models over L x f at the median run time. | outputs/data/runs.csv (models refitted on all runs) |
| 19 | [fig19_c4_old_vs_final.png](fig19_c4_old_vs_final.png) | C4: old pipeline vs final, metric table and P1 parity plots side by side. | outputs/tables/c4_old_vs_final.csv, outputs/data/c4_oof.csv |
| 20 | [fig20_c5_literature.png](fig20_c5_literature.png) | C5: relative COF and k vs load and speed, ours vs Ponnuswamy 2024, and the P1 scores of the pooling strategies. | inputs/ponnuswamy2024_table5.csv, outputs/tables/c5_trends.csv, c5_strategies.csv |
| 21 | [fig21_c6_ta_script.png](fig21_c6_ta_script.png) | C6: TA script - R24 orientation flip, replicate spread original/fixed/ours, COF ratio vs frequency. | inputs/analyze_wear.py on testdata/, outputs/tables/c6_ta_*.csv |
| 22 | [fig22_p3_holdout.png](fig22_p3_holdout.png) | P3 hold-out: measured vs predicted for the 5 test runs, final vs old pipeline. | outputs/data/c4_oof.csv |
