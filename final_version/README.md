# COF and wear of 304 stainless steel in dry reciprocating sliding (ME 323, IIT Bombay)

> Every number in this README is filled in by `python run_all.py` from the result tables in `outputs/tables/`.
> Do not edit the numbers by hand; rerun the pipeline (or `python run_all.py --stage readme`).

## 1. The project in one paragraph

We slid a hardened bearing-steel ball (Ø 6 mm) back and forth over a dry 304 stainless-steel plate in
23 tribometer runs at loads of 10–100 N, frequencies of
4–15 Hz and run times of 60–120 s, then scanned each
wear track with a white-light interferometer. This project turns the raw files into two predictions: the
**coefficient of friction (COF)** and the **wear** (wear volume, specific wear rate k and track width) as functions of
load, frequency and time. Physics laws (Hertz contact, adhesion friction, Archard wear) are compared with the
required machine-learning models (ANN, GPR, TabPFN), with hybrids of the two and with simpler baselines, under
grouped cross-validation. The short answer: **load is the only factor the data can resolve.** The early-run COF follows
μ₀ + c·L^(−1/3) (c = 0.098 [0.049, 0.142]), the track width grows with load
(w ∝ L^0.29), and the wear volume follows Archard's law with one material constant
k = 2.96e-05 mm³/(N·m) ×/÷ 2.4. No ML model beats these laws robustly; the final model for every target is the physics law.

## 2. Quick start

```bash
pip install -r requirements.txt          # core: numpy pandas scipy scikit-learn matplotlib pyyaml joblib reportlab pytest
pip install xgboost                      # optional comparison model
pip install tabpfn                       # optional required model; needs torch and a one-time licence (below)
python run_all.py                        # everything: extraction -> README
python run_all.py --stage figures        # one stage (repeatable: --stage report --stage readme)
python run_all.py --from compare         # a stage and everything after it
pytest                                   # synthetic tests + real-data acceptance tests
```

- **Runtime:** about 24 min for a full run from a clean `outputs/` (no cache) on an 8-core laptop CPU.
  The per-run extraction cache makes reruns fast: unchanged runs are not re-extracted.
- **Raw data is included** in `testdata/` (one folder per run, 230 files, 214 MB): only the files the
  pipeline reads (`.csv`, `.lrx`, `.bcrf`, `.info`). The vendor `.rtec` / `.Raw` containers and `.png` previews are
  never opened and are not shipped. `config.yaml` (`paths.testdata`) can point elsewhere; nothing in `testdata/` is
  ever written.
- **Outputs:** `outputs/data/` (dataset, out-of-fold predictions), `outputs/tables/` (every table as CSV),
  `outputs/figures/` (22 figures, PNG + SVG, listed in `INDEX.md`), `outputs/report/` (`final_report.pdf`,
  `slides_data.md`), `outputs/logs/run.log`, `outputs/environment.json`.
- **TabPFN status in the last run:** not run (tabpfn installed but cannot load weights: TabPFNLicenseError: TabPFN requires a one-time license acceptance to download).
  TabPFN downloads its weights on first use and asks for a one-time licence acceptance: log in at
  https://ux.priorlabs.ai, accept the licence, copy your API key and `export TABPFN_TOKEN=<key>` before
  `python run_all.py --from models`. Without it, TabPFN and TabPFN + physics are logged as "not run" and skipped
  everywhere; nothing else needs torch.

## 3. Folder structure

```
final version/
├── README.md                this file (generated)
├── requirements.txt         Python dependencies
├── config.yaml              every parameter: paths, extraction settings, model grids, CV and test settings
├── run_all.py               one command for everything; --stage / --from run parts
├── inputs/                  run_sheet.csv (DOE), ponnuswamy2024_table5.csv (literature), analyze_wear.py (TA script); read-only copies
├── testdata/                raw data: R01 … R29 run folders (.csv friction log, .lrx recipe, .bcrf height maps, .info)
├── src/tribo_final/
│   ├── io/                  csv_reader (RTEC log), lrx (recipe), bcrf (height map), info (scan metadata), layout (runs, locations, DOE match)
│   ├── extract/             cof (per-cycle COF), wear_band (band method), dataset (one row per run), cache
│   ├── models/              physics, ann, gpr, tabpfn_model, hybrid, baselines, registry (targets + model catalogue)
│   ├── evaluate/            cv (P1/P2/P3), metrics, selection (final-model rule), permutation, interpret
│   ├── compare/             c1_cof … c6_ta (comparison studies)
│   ├── plots/               style (palette, one colour per model), figures (one function per figure)
│   ├── report/              pdf, slides, readme, numbers (all quoted numbers come from the tables)
│   └── stages.py            the pipeline stages
├── tests/                   pytest: synthetic tests and real-data acceptance tests
└── outputs/                 generated: data/, tables/, figures/, report/, logs/, cache/
```

## 4. The experiment

- **Specimen and counter-body.** 304 stainless-steel plate, dry, room conditions; hardened bearing-steel ball,
  Ø 6 mm (R = 3 mm), a new ball for every run.
- **Motion.** Reciprocating ball-on-flat, stroke amplitude A = 5 mm (track length 10 mm). One cycle is 4A = 20 mm of
  sliding, so the sliding distance is S = 4·A·N for N cycles.
- **Protocol per run.** A static loading step (2 s in R01, R02, R04, R05; 10 s in the other 19 runs; ball pressed down, not moving), then
  one continuous sliding step with 1 s acceleration and 1 s deceleration ramps. After the run the track is scanned
  with a white-light interferometer (10× lens) at 4 locations along the track (m, a, b, c).
- **Design of experiments.** The run sheet planned R00–R29 over L = 10–100 N, f = 4–15 Hz, t = 60–120 s. Folders exist
  for **23 runs**: R01, R02, R04, R05, R06, R07, R08, R09, R10, R11, R12, R13, R14, R17, R19, R20, R21, R22, R24, R25, R26, R28, R29. Missing (skipped): R00, R03, R15, R16, R18, R23, R27. The folder `TRAIL'NEW`
  is not a run and is ignored.
- **Roles.** 18 training runs and 5 test runs (R24, R25, R26, R28, R29).
- **Replicates.** R04, R14 and R20 share the same parameters (32 N, 7.5 Hz, 84 s); they are detected from the values
  read from the files and always kept in **one** cross-validation group. Their scatter is the noise floor.
- **Two test days** (2026-10-13: 9 runs, 2026-10-16: 14 runs). The day is recorded in `runs.csv` for
  documentation only. As instructed by the teaching team, it is **never** a model input or correction.

## 5. Raw data formats

| file | content | used? |
|---|---|---|
| `R04.csv` | RTEC friction log, 100 Hz | yes |
| `R04.lrx` | recipe (XML): load, step times, frequency | yes |
| `R04.bcrf`, `R04a.bcrf`, `R04_b.bcrf`, `R01a_13.bcrf` … | height maps, one per scan location | yes |
| `R04.info` … | scan metadata (XML): pixel size, lens, stage position | yes (metadata) |
| `*.png` | previews | no |
| `*.Raw` (≈ 1.2 GB each), `*.rtec`, `*.enc` | raw interferometer frames and vendor containers | **never opened** (≈ 150 GB in total; everything needed is in the .bcrf maps) |

- **Friction CSV.** Line 1 holds the metadata names, a bare carriage return, then the values (`Averaging` = 10 and
  `LogFrequency` = 1000, so the log is 100 Hz; `TestDate`). The file is read in binary and split on CR/LF. The
  channel header follows: `Step, Timestamp, RecipeStep, DAQ.Fz (N), DAQ.COF (Fx), DAQ.Fx (N), XYZ.Z Depth (mm)`; R01–R10
  also log `XYZ.X Velocity` and `Reciprocating.Frequency`. The trailing `TotalTime` lines are dropped. `Timestamp`
  restarts in every RecipeStep, so a continuous clock is built. `XYZ.Z Depth` is a servo signal, never wear.
- **Recipe (.lrx).** `<StepDetail>` texts such as `2> Fz:32 N, t:1.4 min R - Continuous Frequency (Hz): 7.5 …`.
  Load = Fz of the sliding step, run time = sum of the sliding steps' t, frequency from "Frequency (Hz)". These file
  values are the truth: each run is matched to its DOE row **by value** (tolerances L 2 %, f 2 %, t 3 %), never by
  folder name alone. Without an `.lrx`, L, f and t are measured from the CSV.
- **Height map (.bcrf).** A 4096-byte UTF-16-LE header (`xpixels = 960`, `ypixels = 600`, lengths in nm,
  `zunit = um`, `voidpixels`), then float32 heights (µm), row-major. Voids become NaN. Pixel = 1.875 µm, field of view
  1.8 × 1.125 mm. **The wear track always runs along x**, so each row is a line along the track.
- **Scan location** from the file name: the letter right after the run ID, with or without an underscore
  (`R04` → m, `R04a`/`R04_a` → a, `R01a_13` → a, `R01_13` → m; a trailing number is a scan counter).

## 6. Extraction, step by step

### 6.1 COF per cycle (offset-free)

1. **Sliding step.** Rows with logged frequency > 0.5 Hz; without that channel, the RecipeStep block(s) whose Fx
   half-range / Fz is at least 30 % of the largest. The static step never counts.
2. **Trim** 1 s at both ends (the acceleration and deceleration ramps).
3. **Cycles from Fx reversals.** A 20 ms moving median of Fx, centred on (P10 + P90)/2 so the offset cannot bias it;
   sign changes with ±0.2 × half-range hysteresis; two reversals = one cycle. The total cycle count (ramps included)
   integrates the logged frequency, else counts reversals. Check: every run is within
   0.6 % of f·t (R04: 628 cycles vs f·t = 630).
4. **Per-cycle COF = (P90(Fx) − P10(Fx)) / 2 / mean(Fz)** within the cycle, for cycles fully inside the trimmed window.

**Why the half-range and not the instrument column?** `DAQ.COF (Fx)` is |Fx|/Fz sample by sample.
- In the static step it shows the sensor's sideways offset: R01 reads 0.049 with the ball not
  moving.
- During sliding Fx carries an offset (R01: +0.10 N; stroke levels +1.38 /
  -1.18 N), and the mean of |Fx| includes the reversal samples where Fx passes through zero.
- The instrument mean is therefore 13 % below the offset-free COF on
  average, and the bias grows with frequency (ratio vs f: r = -0.70), because
  faster runs have fewer samples per stroke.
- The half-range of each cycle cancels any constant offset, because the forward and return strokes are symmetric about it.

**Why the first 10 s is the primary COF target.** The runs last 238–897 cycles and never
reach steady state: the COF is still rising at the end in **19 of 23 runs** (figure 4). Runs differ in
length and frequency, so the end-of-run COF compares runs at different stages of running-in. The first 10 s
compares every run at the same stage, and it follows the contact-mechanics law (section 10): physics-law R²
+0.36 (P1) vs +0.08 for the end-of-run COF.

### 6.2 Wear from the height maps: the band method with robust levelling

The plates are rough (unworn Sa 3.1–10.4 µm, with long-wavelength
waviness), and the worn track is a **smoother**, slightly sunken band (about 1–2 µm deep) with pile-up ridges at its
edges. A "deepest point" search finds texture valleys and pits. The track is therefore found by its **roughness**, not
its depth (figure 5). All parameters are in `config.yaml` (`wear:`).

1. **First levelling.** Robust plane over the whole map: 3 clipping passes that drop pixels more than
   2.0 SD from the median residual, then a final fit.
2. **Row roughness.** Each row is high-passed along the track (minus a 15-px moving mean, NaN-aware); its SD
   is smoothed across rows (Gaussian, σ = 3 rows).
3. **Contrast.** R_unworn = median roughness of the outer 15 % of rows on each side; R_track = 5th
   percentile; contrast = (R_unworn − R_track)/R_unworn. Below 0.3: no track found.
4. **Core and band.** Core = contiguous rows around the smoothest row with roughness below
   R_track + 0.5 (R_unworn − R_track); **track width = core width**. Band = core ± 40 % of its width, each edge tightened
   to the outer foot of its pile-up ridge (where the smoothed mean profile returns to 0); a side without a ridge ends
   at the core edge.
5. **Reference plane.** Robust plane through all rows outside the band plus a 10 µm margin. It is valid
   only with **≥ 110 µm of unworn rows on both sides**; otherwise the scan is flagged `one_sided` and left
   out of the wear volumes (high-load tracks fill the 1.125 mm field of view).
6. **Profile.** Per row, the mean along the track after 2σ clipping of that row (removes pits and debris; figure 6).
7. **Areas.** Groove = ∫ max(−profile, 0) over the band; pile-up = ∫ max(profile, 0) over the band (µm²).
8. **Uncertainty and LOD.** n_eff = unworn length / 50 µm (texture correlation length); u = SD(reference profile)
   / √n_eff × band width; **LOD = 2u**. The scan is "detected" if its groove area exceeds its LOD.
9. **Per run.** The **most-worn valid, detected location** gives the wear volume V = area × 10 mm and the
   **specific wear rate k = V/(L·S)** (Archard; profilometric cross-section method, cf. ASTM G133). Status:
   `detected`; `below_LOD` (valid scans, none above its LOD; reported as the upper bound
   k < max LOD/(L·S)); `not_measurable` (no valid scan).

**Result:** 16 runs detected, 6 below the LOD (R04, R06, R07, R09, R10, R29), 1 not measurable
(R12). R04: track width 424 µm, below the LOD.

**Two details that differ from the earlier implementation, and why.**
- *Clipping centre.* The 2σ clipping (plane and rows) is centred on the median residual, as in the TA's
  `robust_level`. The texture is valley-dominated (negatively skewed), and median-centred clipping keeps the
  reference on the load-bearing surface.
- *High-pass.* The roughness high-pass is a 1-D moving mean along each row, as specified. The earlier
  implementation that produced the reference numbers used a 15 × 15 2-D box. It is kept as a switch
  (`wear.highpass_mode: box`) and as a C2 sensitivity row. With the box filter this pipeline reproduces the
  reference wear status exactly: 15 detected, R12 and R19 not measurable.
  With the per-row filter, **R19 location b** keeps more than 110 µm of unworn rows on both sides after ridge-foot
  tightening. It becomes measurable (a borderline case at 89 N), giving 16 detected runs instead of 15.
- *Narrow cores on rough faces.* On low-contrast scans of the rougher plates the smooth core is narrow, so the width is
  probably under-read; the runs furthest below the width law are R24, R28, R25.

## 7. The dataset

`outputs/data/runs.csv` has one row per run (23 rows). Also: `per_cycle_cof.csv` (every cycle of every run),
`per_location_wear.csv` (every scan, every wear variant), `wear_variants.csv` (run-level wear for each C2 variant),
`oof_predictions.csv` (every out-of-fold prediction).

| column | meaning |
|---|---|
| run_id | DOE Experiment_ID, matched to the folder by the parameters read from the files (not by folder name) |
| folder_id | name of the folder in testdata/ that holds the run |
| role | `training` (18 runs) or `test` (DOE role test: R24, R25, R26, R28, R29) |
| replicate_group | runs with identical (L, f, t) read from the files share one group (R04/R14/R20 -> R04); CV group |
| is_replicate | True for runs in a group of two or more |
| doe_role | Role column of the run sheet (corner, interior, center_replicate, test, ...) |
| test_date | date from the CSV header (13 or 16 Oct 2026); documentation only, never a model input |
| Load_N, Freq_Hz, Run_time_s | normal load L, reciprocating frequency f and sliding time t from the .lrx recipe |
| params_source | `lrx`, or `csv` when the parameters had to be measured from the friction log |
| cycles | cycles over the whole sliding step, ramps included (integrated frequency channel, else Fx reversals / 2) |
| cycles_ratio_ft | cycles / (f t); QC requires |ratio - 1| <= 2 % |
| S_m | sliding distance S = 4 A N (A = 5 mm), m |
| v_mean_m_s | mean sliding speed 4 A f, m/s |
| COF_first10s | **primary COF target**: mean per-cycle COF in the first 10 s of the trimmed window |
| COF_0.5_2m | mean per-cycle COF for 0.5 m <= s <= 2 m |
| COF_end | median per-cycle COF over the second half of the cycles (the old 'steady' COF) |
| COF_last10s, COF_whole | mean per-cycle COF in the last 10 s / over the whole trimmed window |
| COF_instrument_mean | mean of the instrument `DAQ.COF (Fx)` channel over the trimmed window (comparison only) |
| COF_instrument_static | mean instrument 'COF' in the static loading step (= sensor offset, e.g. R01 0.049) |
| Fx_offset_N, Fx_forward_N, Fx_return_N | Fx offset (P10+P90)/2 and the P90 / P10 stroke levels, N |
| Fz_mean_N | mean normal force in the trimmed window, N |
| COF_slope_end_per_s, cof_rising_end | slope of the per-cycle COF over the second half of the run, and whether it is > 0 |
| n_cycles_window, sliding_time_s, cycles_method | cycles inside the trimmed window, sliding-step duration, how cycles were counted |
| wear_status | `detected`, `below_LOD` (valid scans, none above its detection limit) or `not_measurable` (no valid scan) |
| wear_location | scan location (m, a, b, c) of the most-worn valid, detected location |
| groove_area_um2, groove_LOD_um2 | groove cross-section area at that location and its detection limit, µm² |
| V_mm3 | wear volume = groove area x 10 mm track length, mm³ (detected runs) |
| k_mm3_Nm | specific wear rate k = V / (L S), mm³/(N·m) (detected runs) |
| k_upper_mm3_Nm, V_upper_mm3 | upper bound for below-LOD runs: max LOD over valid scans / (L S) |
| k_mean_loc_mm3_Nm, V_mean_loc_mm3 | k and V from the mean over valid locations (old definition, comparison only) |
| track_width_um, track_width_sd_um | mean (SD) core width over locations where a track was found, µm |
| plasticity_ratio | sum of pile-up / sum of groove areas over valid locations (detected runs only) |
| depth_max_um | deepest point of the across-track profile in the core, over valid locations, µm |
| Sa_unworn_um | mean absolute height of the unworn reference rows after levelling, µm |
| n_scans, n_valid_scans, n_detected_scans | scan counts: all, two-sided reference, above LOD |
| qc_flags | QC notes: invalid scans, folder holding another DOE row, cycle-count check |

## 8. Models

Inputs: **ln L, ln f, ln t**, standardised inside each training fold. No day or session input anywhere.

| target | runs | scale | physics baseline (fitted parameters) |
|---|---|---|---|
| `COF_first10s` (primary) | 23 | COF | COF = μ₀ + c·L^(−1/3); Hertz area ∝ L^(2/3) makes the adhesion term ∝ L^(−1/3) (μ₀, c by OLS) |
| `COF_end` (secondary) | 23 | COF | same law |
| log10 wear volume V | 16 (detected) | log10 mm³ | Archard: log10 V = log10 k̄ + log10(L·S), slope fixed at 1 (intercept only) |
| log10 track width | 23 | log10 µm | log10 w = a + b·ln L; Hertz predicts b = (1/3)/ln 10 = 0.145 |
| log10 k | 16 (detected) | log10 mm³/(N·m) | Archard: k is a constant (intercept only) |

**Required ML models** (teaching team: ANN, GPR and TabPFN):
- **ANN ensemble.** sklearn `MLPRegressor`, one hidden layer of [3, 4] tanh units, lbfgs, `max_iter` =
  5000. L2 `alpha` ∈ [0.001, 0.01, 0.1, 1.0]; hidden size and alpha are chosen by inner GroupKFold(5) on
  the training fold, with 5-net ensembles for speed. The prediction is the mean of
  20 seeds, the uncertainty their SD. Inputs and target are standardised.
- **GPR.** Constant × Matérn(ν = 2.5, one length scale per input) + White noise, `normalize_y`,
  5 optimiser restarts, standardised inputs; uncertainty = predictive SD.
- **TabPFN.** `TabPFNRegressor(device="cpu")` with default settings, plugged into the same evaluation code.
  Status: not run (tabpfn installed but cannot load weights: TabPFNLicenseError: TabPFN requires a one-time license acceptance to download).
- **Hybrids ("ANN + physics", "GPR + physics", "TabPFN + physics").** The physics baseline is fitted on the training
  fold, the ML model on its residuals (inputs ln L, ln f, ln t); prediction = physics + residual model.

**Comparison models** (all on ln L, ln f, ln t): constant (training mean); linear power law (OLS); Ridge (alpha by
inner CV); SVR (RBF; C ∈ [0.1, 1, 10, 100], γ ∈ ['scale', 0.1, 1], ε ∈ [0.01, 0.05, 0.1]);
shallow XGBoost (depth [1, 2], [200, 400] trees, learning rate
[0.03, 0.05], subsample 0.8; skipped if not installed); ANN, GPR and TabPFN on
the raw inputs (not physics-informed).

## 9. Validation

| protocol | data | folds | role |
|---|---|---|---|
| **P1 (primary)** | all runs of the target | leave-one-group-out (R04/R14/R20 = one group; 21 groups for 23 runs) | main score |
| **P2 (stability)** | the 18 training runs only | same LOGO | a real gain must appear in P1 **and** P2 |
| **P3 (hold-out check)** | train on the training runs, predict the 5 test runs once | – | a check, not a sealed test: the test runs were inspected during development |

**Why grouped CV.** With 23 runs there is no data to spare for a fixed test set, so every run is predicted once by a
model that never saw it. The three replicates are one group: leaving one out while its twins stay in training would
let a model "predict" a run it has effectively seen, and inflate every score.

**Metrics** (on each target's modelling scale, pooled out-of-fold): n, R², RMSE, MAE, max |error|, bias,
Pearson r, Spearman ρ, calibration slope (measured on predicted; ideal 1), % of runs within ±1 and ±2 replicate SD,
95 % interval coverage and mean width (ANN ensemble, GPR, OLS prediction interval), and the tercile confusion matrix
with accuracy and Cohen's κ. Glossary in section 14.

**Final-model selection rule** (implemented in `evaluate/selection.py`; the README reports its output):
1. Among the required ML models (ANN, GPR, TabPFN; raw or hybrid), keep those that beat the constant model (lower
   RMSE) in **both** P1 and P2.
2. Among those, keep the ones tied with the best P1 RMSE (1-SE rule over all models), prefer hybrids, then take the
   best P2 R².
3. Report it next to the physics baseline.
4. The selected model replaces the physics law only if it **beats it robustly**: lower RMSE than physics in P1 and
   P2, and not tied with physics by the 1-SE rule in either.

Two clarifications and one addition to the rule as specified, all in code and all reported:
- *Tie tolerance.* Two models whose RMSEs differ by less than 2 % also count as tied. This matters for near-identical
  models such as "ANN + physics" on a constant-physics target vs the raw ANN, where the per-run differences are
  tiny and consistent, so their SE is tinier still.
- *"Robustly"* is read with the same 1-SE rule: a P2 edge of under 1 % RMSE is a tie, not a win.
- *Added safeguard: one-run influence check.* The selected model is refitted (P1) without the single run that helps
  it most against the constant. It is not promoted over physics if it then loses to the constant or to physics.
  This was added because on log10 k the ANN's apparent gain depends on one run (section 10).

**Permutation tests.** The target is shuffled and the whole P1 LOGO evaluation is repeated; p = share of shuffles
scoring at least the real R². The physics baselines use 2000 shuffles, exactly and vectorised. The
hybrids refit every fold of every shuffle with their hyper-parameters fixed at the values chosen on the real data.
To keep the full run under 30 min, they use 300 shuffles (GPR + physics, no optimiser
restarts) and 100 shuffles (ANN + physics, 3-net ensemble) instead of 2000. The
p-value resolution is then 1/(n + 1). The counts are in `config.yaml` (`evaluate:`).

## 10. Results

### 10.1 Final model per target

| target | runs | final model | R² P1 / P2 / P3 | RMSE P1 | perm. p | selected ML model (P1 / P2 R²) | noise floor |
|---|---|---|---|---|---|---|---|
| COF, first 10 s (primary) | 23 | μ₀ + c·L^(−1/3) | +0.36 / +0.23 / +0.70 | 0.0095 | 0.0005 | ANN + physics (+0.43 / +0.04) | 0.0056 |
| COF, end of run (second-half median) | 23 | μ₀ + c·L^(−1/3) | +0.08 / +0.05 / -0.38 | 0.0115 | 0.0215 | none | 0.0062 |
| log10 wear volume V [mm³] | 16 | Archard (slope 1) | +0.29 / +0.15 / +0.38 | 0.387 | 0.0090 | ANN (+0.74 / +0.16) | 0.067 |
| log10 track width [µm] | 23 | a + b·ln L | +0.20 / +0.39 / -1.54 | 0.150 | 0.0050 | GPR + physics (+0.26 / +0.55) | 0.062 |
| log10 k [mm³/(N·m)] | 16 | Archard constant k | -0.15 / -0.20 / -2.34 | 0.387 | n/a | ANN + physics (+0.40 / +0.39) | 0.067 |

How the rule decided:
- **COF, first 10 s (primary)** → **μ₀ + c·L^(−1/3)**. ANN + physics is the selected ML model (beats the constant in P1 and P2, tied with the best P1 RMSE, hybrid preferred) but does not beat the physics baseline robustly (lower RMSE and not 1-SE-tied in both P1 and P2).
- **COF, end of run (second-half median)** → **μ₀ + c·L^(−1/3)**. Required ML models beat the constant in P1 and P2, but none is tied with the best P1 RMSE.
- **log10 wear volume V [mm³]** → **Archard (slope 1)**. ANN is the selected ML model (beats the constant in P1 and P2, tied with the best P1 RMSE) but does not beat the physics baseline robustly (lower RMSE and not 1-SE-tied in both P1 and P2).
- **log10 track width [µm]** → **a + b·ln L**. GPR + physics is the selected ML model (beats the constant in P1 and P2, tied with the best P1 RMSE, hybrid preferred) but does not beat the physics baseline robustly (lower RMSE and not 1-SE-tied in both P1 and P2).
- **log10 k [mm³/(N·m)]** → **Archard constant k**. ANN + physics beats the constant and, beyond the 1-SE tie, the physics baseline in both P1 and P2, and is tied with the best P1 RMSE; but without its most influential run (R11) it no longer beats the constant in P1, so the gain is not robust.

The one-run influence check for the selected ML models:
- COF first 10 s: ANN + physics without R02: R² +0.33 (constant -0.10, physics +0.36)
- log10 V: ANN without R24: R² +0.33 (constant -0.16, physics +0.18)
- log10 width: GPR + physics without R12: R² +0.05 (constant -0.10, physics +0.10)
- log10 k: ANN + physics without R11: R² -0.32 (constant -0.17, physics -0.17)

### 10.2 The fitted laws (95 % run-bootstrap intervals, resampling replicate groups)

- **COF, first 10 s:** COF = μ₀ + c·L^(−1/3) with μ₀ = 0.099 [0.086, 0.114] and c = 0.098 [0.049, 0.142].
  R² +0.36 (P1), +0.23 (P2),
  +0.70 (P3); permutation p = 0.0005.
  Over 10 → 100 N the law predicts a COF drop of
  0.024, against a replicate SD of 0.0056.
- **COF, end of run:** the same law gives R² +0.08 (P1) and +0.05 (P2).
  By then the running-in stage differs from run to run, and load no longer orders the COF.
- **Wear:** k = **2.96e-05 mm³/(N·m) ×/÷ 2.36** (geometric mean ×/÷ geometric SD of
  16 detected runs; 95 % CI of the geometric mean 1.86e-05–4.38e-05).
  As a volume model (Archard, slope fixed at 1) this gives R² +0.29 (P1),
  +0.15 (P2), permutation p = 0.0090. A free fit gives
  V ∝ L^0.96 [0.41, 1.59] · S^-0.07
  [-1.32, 1.61]; Archard expects 1 and 1.
- **Track width:** w ∝ L^0.29
  [0.12, 0.47]
  (Hertz: 1/3), R² +0.20 (P1), +0.39 (P2).

### 10.3 Noise floor (replicates R04/R14/R20)

| target | replicate SD | replicate runs | spread over runs (SD) | signal / noise |
|---|---|---|---|---|
| COF, first 10 s (primary) | 0.0056 | R04,R14,R20 | 0.0121 | 2.2 |
| COF, end of run (second-half median) | 0.0062 | R04,R14,R20 | 0.0123 | 2.0 |
| log10 wear volume V [mm³] | 0.0668 | R14,R20 | 0.4739 | 7.1 |
| log10 track width [µm] | 0.0625 | R04,R14,R20 | 0.1717 | 2.7 |
| log10 k [mm³/(N·m)] | 0.0670 | R14,R20 | 0.3727 | 5.6 |

The wear replicate SD comes from R14 and R20 only, because R04 is below the LOD. A signal-to-noise ratio near 1–2 means
the runs differ only a little more than identical repeats do. That caps what any model can explain.

### 10.4 What the models say

- **Load dominates.** It has the shortest GPR length scale in 4 of 5 targets (COF first 10 s: L 2.24, f 2.0, t 3.1; COF end: L 1.10, f 1.5, t 1.5; log10 V: L 0.44, f 1.8, t 1000.0; log10 width: L 0.01, f 1000.0, t 1000.0; log10 k: L 0.33, f 1.1, t 985.7; standardised
  inputs, shorter = stronger effect; 1000 = upper bound, i.e. no effect). It is the most important ANN input for
  5 of 5 targets.
- **Frequency and time are not detectable.** Power-law
  exponents (d ln y / d ln x, 95 % CI):

| target | load | frequency | time |
|---|---|---|---|
| COF first 10 s | -0.09 [-0.12, -0.04] | -0.05 [-0.15, +0.05] | -0.05 [-0.18, +0.08] |
| COF end | -0.06 [-0.10, +0.01] | -0.02 [-0.13, +0.10] | -0.04 [-0.17, +0.15] |
| log10 V | +1.02 [+0.44, +1.74] | -0.26 [-1.62, +1.44] | +0.70 [-1.55, +3.36] |
| log10 width | +0.35 [+0.15, +0.57] | +0.29 [-0.13, +0.81] | +0.49 [-0.28, +1.28] |
| log10 k | +0.02 [-0.56, +0.74] | -1.26 [-2.62, +0.44] | -0.31 [-2.55, +2.35] |

- **k is constant, per Archard.** Load, frequency and time do not move k beyond the replicate scatter in any model that
  survives P2 and the influence check. Every apparent trend in k rests on one or two runs (10.5).
- **COF follows the adhesion law early in the run.** It is lower at higher load (c > 0, 95 % CI above 0),
  as μ₀ + c·L^(−1/3) predicts, in the first 10 s (R² +0.36); at the end of the run the
  same law explains R² +0.08, lost in running-in differences.

### 10.5 Fragile ML gains (read this before quoting an ML score)

- **Unstable models** (beat the constant in P1, lose to it in P2): XGBoost on COF first 10 s (P1 +0.09, P2 -0.21), GPR on COF end (P1 +0.37, P2 -0.38), ANN + physics on COF end (P1 +0.16, P2 -0.59), GPR + physics on COF end (P1 +0.20, P2 -0.22).
- **log10 k and log10 V.** The rule selects ANN + physics for log10 k
  (P1 R² +0.40, P2 +0.39),
  and ANN for log10 V (P1 +0.74).
  These gains do not survive scrutiny:
  - Without its most influential run, the k model falls below the constant (10.1).
  - Its P3 RMSE is 0.44 dex, against
    0.34 dex for the constant.
  - The pattern it learns is mostly a frequency effect (k exponent on f: -1.26 [-2.62, +0.44]). All
    4 runs at f ≥ 11 Hz were made on 2026-10-13, and the unworn plate roughness differs by day
    (2026-10-13: Sa 3.1–3.4 µm; 2026-10-16: Sa 5.7–10.4 µm). So the frequency
    "effect" cannot be told apart from the day/plate difference, which must not be modelled.
  - Sensitivity of the P1 R² on log10 k to single runs and to the wear-list definition (`tables/c3_k_fragility.csv`):

| dataset | n | R² constant | R² ann | R² ann_phys |
|---|---|---|---|---|
| all detected runs | 16 | -0.15 | +0.41 | +0.40 |
| without R11 | 15 | -0.17 | -0.33 | -0.32 |
| without R19 | 15 | -0.16 | +0.12 | +0.13 |
| without R22 | 15 | -0.16 | -0.37 | -0.42 |
| 2-D box high-pass wear list | 15 | -0.17 | -0.28 | -0.28 |

- **Permutation tests:**

| target | model | shuffles | R² (test statistic) | 95th pct. of shuffled R² | p |
|---|---|---|---|---|---|
| COF first 10 s | Physics | 2000 | +0.36 | -0.01 | 0.0005 |
| COF first 10 s | GPR + physics | 300 | +0.08 | +0.05 | 0.0399 |
| COF first 10 s | ANN + physics | 100 | +0.42 | +0.08 | 0.0099 |
| COF end | Physics | 2000 | +0.08 | +0.00 | 0.0215 |
| COF end | GPR + physics | 300 | +0.20 | +0.06 | 0.0133 |
| COF end | ANN + physics | 100 | -0.75 | -0.39 | 0.1584 |
| log10 V | Physics | 2000 | +0.29 | -0.01 | 0.0090 |
| log10 V | GPR + physics | 300 | +0.19 | +0.16 | 0.0498 |
| log10 V | ANN + physics | 100 | +0.58 | +0.05 | 0.0099 |
| log10 width | Physics | 2000 | +0.20 | +0.01 | 0.0050 |
| log10 width | GPR + physics | 300 | +0.21 | +0.06 | 0.0100 |
| log10 width | ANN + physics | 100 | +0.00 | -0.06 | 0.0495 |
| log10 k | GPR + physics | 300 | -0.32 | +0.14 | 0.6213 |
| log10 k | ANN + physics | 100 | +0.28 | -0.23 | 0.0198 |

### 10.6 Comparison with the expected reference numbers

| check | expected | this pipeline | within tolerance | note |
|---|---|---|---|---|
| R04 parameters | 32 N, 7.5 Hz, 84 s | 32 N, 7.5 Hz, 84 s | yes |  |
| R04 cycles | 626 ± 10 | 628 | yes | integrated frequency channel |
| R04 COF_end | 0.155 ± 0.015 | 0.152 | yes |  |
| R04 track width | 414 ± 60 µm | 424 µm | yes |  |
| R01 static instrument COF | ≈ 0.049 | 0.049 | yes | sensor offset |
| wear: detected runs | 15 ± 1 | 16 | yes |  |
| wear: below LOD | R04, R06, R07, R09, R10, R29 | R04, R06, R07, R09, R10, R29 | yes |  |
| wear: not measurable | R12, R19 | R12 | no | R19 location b has > 110 µm of unworn rows on both sides with the per-row high-pass (see section 6) |
| wear replicate SD | ≈ 0.06 dex (R14/R20) | 0.07 dex | yes | R14 and R20 differ by a factor 1.2 |
| COF replicate SD | ≈ 0.006 | 0.0056 (first 10 s), 0.0062 (end) | yes |  |
| COF_first10s physics R² | +0.36 / +0.22 | +0.36 / +0.23 | yes | P1 / P2 |
| COF_end physics R² (P1) | +0.03 | +0.08 | yes |  |
| track width physics R² | +0.26 / +0.40 | +0.20 / +0.39 | yes | P1 / P2 |
| wear volume Archard R² (P1) | +0.19 | +0.29 | yes | 16 detected runs instead of 15 |
| typical k | ≈ 2.9e-5 ×/÷ 2.4 | 3.0e-05 ×/÷ 2.4 | yes | mm³/(N·m) |
| plain GPR on COF_end | ≈ +0.36 / −0.40 | +0.37 / -0.38 | yes | P1 / P2 |
| no model beats a constant for k | yes | no robust winner | see text | the rule selects ANN + physics (P1 +0.40), but without R11 it falls to -0.32 |
| literature COF trends | paper L^+0.52 v^−0.69, ours L^−0.06 v^−0.02 | paper L^+0.52 v^-0.69, ours L^-0.09 v^-0.05 | yes |  |

Outside tolerance or needing comment: **wear: not measurable** (R19 location b has > 110 µm of unworn rows on both sides with the per-row high-pass (see section 6)); **no model beats a constant for k** (the rule selects ANN + physics (P1 +0.40), but without R11 it falls to -0.32). The ANN's apparent skill on log10 k (P1 R² +0.41 on all detected runs) changes to -0.33 without R11, +0.12 without R19, -0.37 without R22 and to -0.28 on the reference implementation's wear list (constant: -0.15); this is why the reference result 'no model beats a constant for k' and ours differ in the raw ML scores but agree after the robustness checks.

## 11. Why these choices (the comparison studies, in plain language)

- **C1 – Why not the end-of-run COF or the instrument COF?** Use COF_first10s: it follows the adhesion law best among the definitions (P1 R² +0.36, P2 +0.23; end-of-run COF +0.08) with signal-to-noise 2.2; the instrument mean reads 13 % low and its ratio to ours falls with frequency (r = -0.70). The end-of-run COF mixes runs at different
  stages of running-in, so the law that holds early is lost (R² +0.08). The
  instrument mean is biased low, and more so at high frequency. (`tables/c1_*.csv`, figure 8)
- **C2 – Why the most-worn location, and why robust levelling?** Use the most-worn valid location with robust levelling: 16 runs measurable with an LOD, replicate SD 0.07 dex, Archard R² P1 +0.29 / P2 +0.15; the old mean-of-locations, plain-levelling measure gives 0.34 dex and R² +0.08 / +0.01. Exponents: V ~ L^0.96 [0.41, 1.59] S^-0.07 [-1.32, 1.61] (Archard: 1, 1). The TA depth rule scores a higher R² (+0.52) only because it drops the shallow part of every groove: runs at <= 25 N keep a median 11 % of their groove, runs at >= 60 N 91 %, which inflates the load exponent to 1.9; it is not a volume. The mean over locations averages
  detected grooves with locations below their LOD, and plain levelling lets pits and debris tilt the reference.
  (`tables/c2_*.csv`, figure 9)
- **C3 – Why not plain ANN or GPR?** COF first 10 s: final = physics law (selected ML: ANN + physics); COF end: final = physics law (selected ML: none); log10 V: final = physics law (selected ML: ANN); log10 width: final = physics law (selected ML: GPR + physics); log10 k: final = physics law (selected ML: ANN + physics). The flexible models can win P1 by fitting a few influential runs and
  lose it on the 18 training runs (P2). The hybrids stay close to the physics law but add no robust skill. Table for
  the primary target (all targets in `tables/c3_models.csv`, figures 11–12):

| model | P1 R² | P2 R² | parameters | uncertainty | physically consistent | verdict |
|---|---|---|---|---|---|---|
| Constant | -0.09 | -0.12 | 1 | yes | no | reference |
| Physics | +0.36 | +0.23 | 2 | yes | yes | physics baseline (final model) |
| Power law (OLS) | +0.19 | -0.01 | 4 | yes | yes | worse than physics |
| Ridge | +0.15 | -0.11 | 4 | no | yes | worse than physics |
| SVR | -0.21 | -0.12 | 23 SV | no | yes | no better than constant |
| XGBoost | +0.09 | -0.21 | 200 trees, depth 1 | no | yes | unstable (beats constant in P1 only) |
| ANN | +0.14 | +0.15 | 21 | yes | yes | worse than physics |
| GPR | -0.17 | -0.42 | 5 | yes | yes | no better than constant |
| TabPFN | – | – | – | – | – | not run (tabpfn installed but cannot load weights: TabPFNLicenseError: TabPFN requires a one-time license acceptance to download) |
| ANN + physics | +0.43 | +0.04 | 23 | yes | yes | chosen (selected ML model); not better than physics |
| GPR + physics | +0.15 | +0.02 | 7 | yes | yes | worse than physics |
| TabPFN + physics | – | – | – | – | – | not run (tabpfn installed but cannot load weights: TabPFNLicenseError: TabPFN requires a one-time license acceptance to download) |

- **C4 – Why replace the old pipeline?** Replace the old pipeline. COF: the old ANN on end-of-run COF scores R² +0.40 in P1 but +0.00 on the training runs (P2); the final model on COF_first10s scores +0.36 / +0.23 with two parameters. Wear: R² -0.48 / -0.47 (ANN on mean-of-locations k, plain levelling, replicate SD 0.34 dex) -> +0.29 / +0.15 (final model on the most-worn-location volume, replicate SD 0.07 dex). The old wear target had no detection limit and mixed locations,
  so its noise floor was larger; the old COF target compared runs at different running-in stages.
  (`tables/c4_old_vs_final.csv`, figures 15, 19, 22)
- **C5 – Why not the literature data?** Do not use the literature data: paper COF ~ L^+0.52 v^-0.69 vs ours L^-0.09 v^-0.05; log10_COF: best with paper data GPR + study flag R² +0.24 vs ours alone +0.28; log10_k: best with paper data pooled + study offset: power law R² -0.31 vs ours alone +0.41. The published COF (0.002-0.007) and volumes (up to 1620 mm³ in 200 s) are implausible as printed.
  - The paper's COF rises with load (L^+0.52 [+0.08, +0.93]) and
    falls with speed (v^-0.69 [-0.94, -0.44]). Ours
    falls with load (L^-0.09 [-0.12, -0.04]) and
    shows no detectable trend with speed (v^-0.05 [-0.15, +0.05]).
  - Its COF (0.0019–0.0072) is 18–70 times lower than ours, far
    below any dry steel pair, and its wear volumes (up to 1620 mm³ in 200 s, about the whole
    pin) are implausible as printed.
  - The best strategy that uses the paper scores R² +0.24 on our COF (GPR + study flag),
    against +0.28 for our data alone; for k, -0.31 (pooled + study offset: power law) against
    +0.41. (`tables/c5_*.csv`, figure 20)
- **C6 – Why not the TA script?** Do not use the TA script as written: 25 of 92 scans are read along the track, and the R04/R14/R20 replicates spread 30x in k (SD 0.84 dex); fixed copy 0.13 dex, ours 0.07 dex (R14/R20 only). Archard R² (P1): original -0.03, fixed -0.19, ours +0.29. The bugs, with evidence from our data (`tables/c6_ta_bugs.csv`,
  figure 21):

1. **Orientation chosen from the unlevelled map (tilt decides).** 25 of 92 scans transposed, so the 'profile' runs along the track. R24.bcrf: transposed, reported depth 22.5 um (a pit), groove area 4352 um^2; with the correct axis the same code gives depth 7.4 um, groove area 0 um^2.
2. **Points shallower than the cutoff zeroed one by one.** 10 scans (correct axis) have groove area 0 although the depth is >= 3 um, e.g. R05b_13.bcrf (3.4 um), R11.bcrf (4.2 um), R13.bcrf (4.1 um), R14_a.bcrf (5.9 um).
3. **Pile-up summed over the whole image outside the groove.** median TA pile-up / band pile-up = 1.5 over 65 valid scans (texture peaks counted as pile-up).
4. **Shoulder (reference) rows never checked to be unworn.** in 23 of 90 scans with a track, the worn band reaches into the outer 15 % rows that the TA uses as the unworn baseline.
5. **Scan picked by the deepest point (texture valleys).** the deepest-point scan differs from the largest-groove valid scan in 12 of 20 runs with a detected groove.
6. **COF = mean of the instrument |Fx|/Fz channel; deceleration second not trimmed.** TA COF / offset-free COF = 0.873 on average (13 % low); the ratio falls with frequency (r = -0.70, p = 0.000).
7. **Hard-coded R01 parameters; friction log = first *.csv; outputs written into the data folder.** R04 with the default arguments (10 N, 6 Hz, 120 s): k = 6.53e-06 instead of 2.33e-06 mm^3/(N m) (2.8x); run_pipeline() writes all_scans_breakdown.csv and single_scan_summary.csv into each run folder (avoided here).

## 12. Limitations

- **Short runs that never reach steady state.** 238–897 cycles; the COF is still rising at the end
  in 19 of 23 runs. "Steady-state COF" cannot be measured here; the models describe running-in.
- **Rough plates and shallow grooves.** Unworn Sa 3.1–10.4 µm against
  grooves only about 1 µm deep. 6 runs are below the detection limit (R04, R06, R07, R09, R10, R29) and 1 not
  measurable (R12). Low-contrast scans on the rougher plates give narrow cores, so some track
  widths are probably under-read (furthest below the width law: R24, R28, R25).
- **Only 23 runs.** With 5 targets, 12 candidate models and 21 CV groups, single runs can sway a
  score (see the influence check). The noise floor is set by one replicate triplet.
- **Two test sessions, not modelled as instructed.** The day, the plate face and the high-frequency runs are
  partly confounded, so a small frequency effect cannot be ruled out; this design cannot detect it.
- **High-load tracks exceed the field of view.** At 80–100 N the tracks leave too little unworn surface for a two-sided
  reference at most locations (valid locations: R11 (100 N): 3 of 4, R12 (100 N): 0 of 4, R19 (89 N): 1 of 4, R22 (80 N): 2 of 4).
- **TabPFN** is reported as "not run (tabpfn installed but cannot load weights: TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)" unless its licence token is set; the evaluation code runs it unchanged once it is.
- **Reduced permutation counts** for the ML hybrids (300 / 100 shuffles) keep the
  runtime under 30 min; physics baselines use 2000.

## 13. Reproducibility

- **Seeds.** One seed (`seed: 42` in `config.yaml`) drives every model, the CV inner folds, the bootstraps and
  the permutation tests. ANN ensembles use seeds seed … seed + 19.
- **Versions.** Written to `outputs/environment.json` on every run (last run: Python 3.13.5,
  numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1,
  xgboost 3.1.1, tabpfn 9.2.0).
- **Caching.** Per-run extraction results are cached in `outputs/cache/extract/`. The key covers the input files' size
  and modification time, the extraction config and a hash of the reader/extraction source code. Model predictions,
  permutation tests and the C4–C6 studies are cached by a hash of the dataset and config. Delete `outputs/` (or use
  `--force-extract`) for a cold run.
- **Rerun one stage:** `python run_all.py --stage <name>` with name in extract, dataset, models, evaluate, compare,
  figures, report, readme. Each stage reads the previous stages' outputs from `outputs/`.
- **Tests:** `pytest` (synthetic tests always; real-data acceptance tests skip if `testdata/` is absent).

## 14. Glossary

| term | meaning |
|---|---|
| COF | coefficient of friction, friction force / normal force |
| Half-range COF | (P90 - P10)/2 of Fx within one cycle divided by mean Fz; cancels any constant Fx offset |
| Running-in | the early phase of sliding in which friction and wear change as the surfaces adapt |
| Archard's law | V = k L S: wear volume proportional to load and sliding distance; k is the specific wear rate |
| Hertz contact | elastic sphere-on-flat contact; contact radius ~ L^(1/3), area ~ L^(2/3) |
| LOD | limit of detection: 2 x the uncertainty of the groove area from the reference-surface noise |
| Band method | finds the worn track as the smoother band of rows, then measures its groove against an unworn reference plane |
| Robust levelling | plane fit that drops pixels more than 2 SD from the fit (3 passes), so pits and debris do not tilt it |
| LOGO | leave-one-group-out cross-validation: each replicate group is held out once |
| P1 / P2 / P3 | validation protocols: all runs LOGO / training runs LOGO / train on training, predict test runs once |
| R² | 1 - SSE / SST on pooled out-of-fold predictions; 0 = as good as the mean, < 0 = worse than the mean |
| RMSE / MAE | root-mean-square / mean absolute error on the target's modelling scale |
| dex | one decade on a log10 scale (0.1 dex = a factor of 1.26) |
| Coverage | share of runs whose measured value lies inside the model's 95 % interval (ideal 95 %) |
| Tercile confusion matrix | measured and predicted values binned into low / mid / high by the measured terciles |
| Cohen's κ | agreement of the tercile classes beyond chance (0 = chance, 1 = perfect) |
| 1-SE rule | two models are tied if the mean per-run squared-error difference is within one standard error |
| Permutation test | refit on shuffled targets; p = share of shuffles that score at least as well as the real data |
| Hybrid | physics law + an ML model fitted to its residuals |
| Noise floor | SD of the identical replicate runs: no model can be expected to predict better than this |
| GSD | geometric standard deviation; k ×/÷ GSD covers about 68 % of runs on a log scale |
