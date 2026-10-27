# Prompt: build the final version of the ME 323 tribology project from scratch

Copy everything below the line into a new session that has access to the `323_project` folder.

---

## 0. Your task

You are building, **from scratch**, the final, clean version of a university project (IIT Bombay ME 323). It turns
raw tribometer data from 23 dry reciprocating sliding tests on 304 stainless steel into:
- a coefficient-of-friction (COF) model;
- a wear model;
- every result, comparison table and plot needed for a presentation;
- a README that explains the whole pipeline to someone who has never seen it.

Work only inside a new folder **`323_project/final version/`** (on this Mac: `/Users/ananyay/Desktop/323_project/final version/`;
create it if absent). The folder name contains a space: build every path with `pathlib`, quote paths in shell
commands, and keep the Python package name space-free (`tribo_final`). Read the inputs listed in section 2, but never
modify, move or delete anything outside `final version/`. Do not reuse code from the older folders (`tribo_extract/`, `ml/`,
`tribo304/`); write fresh, tidy code. All the domain knowledge you need is in this prompt.

Constraints:
- Python ≥ 3.10.
- Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, joblib, reportlab, pytest.
- Optional: xgboost; tabpfn (needs torch). Skip each cleanly with a logged "not run" if it is not installed. Never
  require torch for anything else.
- One command must reproduce everything: `python run_all.py`. Target under 30 min on a laptop CPU.
  - Cache extraction results per run.
  - Parallelise with joblib.
  - Fix all random seeds.
- Pandas ≥ 3 note: `Series.astype(str)` keeps missing values as NaN (not the string "nan"); use `fillna("")` before
  string comparisons.

The teaching team requires **ANN, GPR and TabPFN** to be used. They also said the **test day must NOT be used as a
model input or correction**, so do not add any day/session factor anywhere in the models.

## 1. The experiment (domain background)

- **Specimen.** 304 stainless steel plate, dry, at room conditions.
- **Counter-body.** A hardened bearing-steel ball, Ø 6 mm (R = 3 mm), new for every run.
- **Motion.** Reciprocating ball-on-flat. Stroke amplitude A = 5 mm (±5 mm), so the track length is 10 mm and one
  cycle is 4·A = 20 mm of sliding. Sliding distance S = 4·A·N, where N is the number of cycles.
- **Factors.** Load L = 10–100 N, frequency f = 4–15 Hz, run time t = 60–120 s.
- **Protocol per run.**
  1. A 2 s static loading step (ball pressed down, not moving).
  2. ONE continuous sliding step, with 1 s acceleration and 1 s deceleration ramps.
  3. After the run, the track is scanned with a white-light interferometer (10× lens) at 4 locations along the
     track (m, a, b, c).
- **Two test days** (13 and 16 Oct 2026). This is known, but must not be modelled.
- **Runs.**
  - The DOE planned R00–R29. Folders exist for **23 runs**: R01, R02, R04–R14 (no R03), R17, R19, R20, R21, R22,
    R24, R25, R26, R28, R29.
  - Missing runs (R00, R03, R15, R16, R18, R23, R27) are simply skipped. Ignore any folder whose name is not a run
    ID (e.g. `TRAIL'NEW`).
  - Role "test" in the DOE: R24, R25, R26, R28, R29. The other 18 are "training".
  - R04, R14 and R20 are replicates (32 N, 7.5 Hz, 84 s). Detect replicates by identical parameters read from the
    files, and keep them in **one CV group**.

## 2. Inputs

| path (relative to `323_project/`) | content |
|---|---|
| `testdata/<RunID>/` | one folder per run (formats in section 3) |
| `data/run_sheet.csv` | DOE table: Experiment_ID, Session, Face, X_mm, Y_mm, Load_N, Freq_Hz, Run_time_s, Cycles_nominal, Role, Replicate_of, Include, … |
| `data/external/ponnuswamy2024_table5.csv` | 18 literature runs (Ponnuswamy & Manoharan 2024, Mater. Res. Express 11 066515, Table 5): Run, Track_diameter_mm, Load_N, Speed_m_s, Time_s (200), Wear_volume_mm3, COF |
| `external/ta_analyze_wear/analyze_wear.py` | the TA's wear/COF script (comparison only; it has known bugs, section 8) |

At setup, copy the last three small files into `final version/inputs/`, so the project is self-contained except
for `testdata/`. `config.yaml` points to `../testdata` by default.

## 3. File formats (confirmed on the real files)

Per run folder, e.g. `testdata/R04/`:

| file | use |
|---|---|
| `R04.csv` | friction log |
| `R04.lrx` | recipe (load, time, frequency) |
| `R04.bcrf`, `R04a.bcrf`, `R04_b.bcrf`, `R01a_13.bcrf` … | height maps, one per scan location |
| `R04.info`, … | scan metadata |
| `*.png` | previews (unused) |
| `*.Raw` (≈1.2 GB each), `*.rtec`, `*.enc` | **never open** |

**Scan location from the file name.** The letter right after the run ID, with or without an underscore:
- `R04` → m;
- `R04a` / `R04_a` → a;
- `R01a_13` → a;
- `R01_13` → m (a trailing number is a scan counter).

Regex: `^[RT]\d{2}_?([a-c])(?=$|[_. ])`, else m. All scans are post-run.

**`.lrx` (XML).** `<StepDetail>` texts such as:
- `1> Fz:32 N,  t:0.03333 min  None` (static step);
- `2> Fz:32 N,  t:1.4 min  R - Continuous  Frequency (Hz): 7.5  Acceleration (s): 1  Deceleration (s): 1`.

Load_N = Fz of the sliding step. Run_time_s = sum of the sliding steps' t (min → s, rounded). Freq_Hz from
"Frequency (Hz)". These file values are the truth: match each run to its DOE row **by value** (tolerances L 2 %,
f 2 %, t 3 %), never by folder name alone. If there is no `.lrx`, measure from the CSV:
- mean Fz rounded to 0.5 N;
- f = reversals / 2 / sliding time;
- t = sliding duration.

**Friction CSV (RTEC export).**
- **Metadata (line 1).** Line 1 holds metadata names, then a **bare CR**, then metadata values. Read the file in
  binary and split on CR/LF, or use universal newlines. Metadata includes `Averaging` (= 10, so the log is
  **100 Hz**), `TestDate` (e.g. `10/13/2026 2:10:25 AM`) and `LogFrequency` (1000).
- **Channels.** The next line is the channel header: `Step, Timestamp, RecipeStep, DAQ.Fz (N), DAQ.COF (Fx),
  DAQ.Fx (N), XYZ.Z Depth (mm)`. R01–R10 also have `XYZ.X Velocity (mm/s)` and `Reciprocating.Frequency (Hz)`.
- **Trailing lines.** The file ends with `TotalTime` / `00:01:30.0`; drop them.
- **Timestamp** restarts at about 0 in each RecipeStep. Build a continuous clock.
- **`DAQ.COF (Fx)`** is |Fx|/Fz sample by sample. **Do not use it as the COF:**
  - In the static step it shows the sensor's sideways offset (e.g. R01: Fx ≈ 0.49 N at 10 N gives "COF" 0.049).
  - During sliding, its average includes the reversal samples where Fx passes through zero. That biases it about
    12 % low, more at high frequency.
- **Fx offset.** During sliding Fx has an offset (e.g. +0.11 N for R01), so the forward and return strokes differ
  (R01: +1.59 N / −1.36 N).
- `XYZ.Z Depth` is a servo signal, never wear.

**`.bcrf` height map (BCR-F unicode).**
- Header: UTF-16-LE, 2048 characters = 4096 bytes, `key = value` lines: `xpixels = 960`, `ypixels = 600`,
  `xlength`/`ylength` (nm), `zunit = um`, `voidpixels = 3.402823E+38`, `bit2nm`.
- Data: float32 little-endian, row-major (ny = 600, nx = 960), heights in µm. Void values (> 1e30) become NaN.
- Pixel size = xlength / nx / 1000 = **1.875 µm**. Field of view 1.8 × 1.125 mm.
- **The wear track always runs along x**, so each row is a line along the track and the across-track profile is
  the row means. Do not "auto-detect" the orientation on unlevelled data (see bug 1 in section 8).
- `.info` (XML `CRunDetail`): PixelSizeuM, Lens (I-10x), Mode (WLI), XPos/YPos (stage mm).

## 4. Extraction (`src/tribo_final/extract/`)

### 4.1 COF per cycle (offset-free)

1. **Sliding step.**
   - If the frequency channel exists: rows with frequency > 0.5 Hz.
   - Otherwise: the RecipeStep block whose Fx half-range / Fz is ≥ 30 % of the largest. Skip blocks under 10 rows.
   - The 2 s static step must never count as sliding.
2. **Trim** 1 s after the start and 1 s before the end of the sliding step (the ramps).
3. **Cycles from Fx reversals.**
   - Take a 20 ms moving median of Fx, centred on (P10 + P90)/2 of the step so the offset cannot bias it.
   - Count sign changes with hysteresis ±0.2 × half-range; every second reversal is one cycle.
   - Count total cycles over the whole sliding step, ramps included: integrate the logged frequency if available,
     else count reversals.
   - Check: within 2 % of f·t.
4. **Per-cycle COF** = (P90(Fx) − P10(Fx)) / 2 / mean(Fz) within the cycle. Use only cycles fully inside the trimmed
   window. Record t (cumulative sliding time at mid-cycle) and s = 4·A × cycles so far.
5. **Run-level COF targets:**
   - `COF_first10s` (**primary COF target**): mean of the per-cycle COF in the first 10 s of the trimmed window.
   - `COF_end`: median over the second half of the cycles (the old "steady" COF).
   - `COF_last10s`, `COF_whole`, `COF_0.5_2m` (cycles with 0.5 ≤ s ≤ 2 m).
   - `COF_instrument_mean`: the mean of the `DAQ.COF (Fx)` channel, for the comparison only.
6. **Sliding distance:** S = 4·A·N_counted; cycles = N_counted.

Why the first 10 s is the primary target: the runs (240–900 cycles) never reach steady state. COF is still rising
at the end in 18 of 23 runs. Runs differ in length and frequency, so the end-of-run COF compares runs at different
stages of running-in. The first 10 s compares every run at the same stage, and it follows the contact-mechanics
law (section 5).

### 4.2 Wear from the height maps: "band" method with robust levelling

The plates are rough (unworn Sa 3–9 µm, with long-wavelength waviness). The worn track is a **smoother**, slightly
sunken band (about 1–2 µm deep) with pile-up ridges at its edges. Detect it by roughness, not depth. Parameters go in
`config.yaml`.

1. **First levelling.** Fit a plane to the whole map with robust levelling: 3 passes, dropping pixels more than
   2 SD from the fit (`clip_sigma = 2.0`).
2. **Local roughness per row.** High-pass each row along the track (subtract a 15-px moving mean, NaN-aware). Take
   the SD per row, then smooth across rows with a Gaussian (σ = 3 rows).
3. **Contrast.**
   - R_unworn = median roughness of the outer 15 % of rows on each side.
   - R_track = 5th percentile of the row roughness.
   - contrast = (R_unworn − R_track) / R_unworn.
   - If contrast < 0.30: no track found.
4. **Core and band.**
   - Core = contiguous rows around the smoothest row with roughness < R_track + 0.5·(R_unworn − R_track).
     Track width = core width × pixel size.
   - Band = core ± 40 % of the core width.
   - Tighten each band edge to the outer foot of its pile-up ridge, i.e. where the smoothed mean profile returns
     to 0.
5. **Reference plane.**
   - Fit a plane (robust, 2σ clipping) to all rows outside the final band plus a 10 µm margin.
   - Valid only if there are at least **110 µm of unworn rows on BOTH sides**. Otherwise flag `one_sided` and
     exclude the scan from wear volumes. This happens for the 89–100 N tracks, which fill the field of view.
6. **Profile.** For each row, the mean along the track after 2σ clipping of that row (removes pits and debris).
7. **Areas.** Groove area = ∫ max(−profile, 0) over the band (µm²). Pile-up area = ∫ max(profile, 0) over the band.
8. **Uncertainty and detection limit.**
   - n_eff = (unworn length)/50 µm, with a texture correlation length of 50 µm.
   - u = SD(profile outside the band)/√n_eff × band width.
   - LOD = 2u. The scan is "detected" if groove area > LOD.
9. **Per run:**
   - **Most-worn valid location:** the valid, detected location with the largest groove area. Wear volume
     V = area × 10 mm (µm² × 1e-6 → mm², × 10 mm → mm³). Specific wear rate **k = V/(L·S)** in mm³/(N·m).
   - Status:
     - "detected";
     - "below_LOD" (valid scans exist, none detected): report the upper bound k < max(LOD)/(L·S);
     - "not_measurable" (no valid scan, R12 and R19).
   - Also report:
     - k from the mean over valid locations (old definition, comparison only);
     - track width = mean over locations where a track was found;
     - plasticity ratio = Σ pile-up / Σ groove;
     - max wear depth.

### 4.3 Dataset

One row per run, saved to `outputs/data/runs.csv` and documented in the README data dictionary. Columns:
- run ID, role, replicate group, test date (kept for documentation only);
- L, f, t, cycles, S, mean sliding speed v = 4·A·f;
- all COF targets;
- k, V, status, upper bound, track width, plasticity ratio, unworn Sa;
- QC flags.

Also save `per_cycle_cof.csv` and `per_location_wear.csv`.

## 5. Models (`src/tribo_final/models/`)

Model **targets**:

| target | runs | scale |
|---|---|---|
| `COF_first10s` (primary) | 23 | COF |
| `COF_end` (secondary) | 23 | COF |
| log10 wear volume V | detected runs only (expected 15) | log10 |
| log10 track width | 23 | log10 |

The final wear-rate result is k reported as one material constant (geometric mean ×/÷ geometric SD, with a bootstrap
CI), because Archard's law expects k to be independent of the test settings. Still run log10 k (detected runs) through
every model in C3, so the presentation can show the evidence: no model beats a constant for k.

Inputs: ln L, ln f, ln t (standardised inside each training fold). No day/session input.

### 5.1 Physics baselines

| target | physics model | fitted parameters |
|---|---|---|
| COF | COF = μ₀ + c·L^(−1/3) (Hertz: contact area ∝ L^(2/3) gives the adhesion term ∝ L^(−1/3)) | μ₀, c (OLS) |
| Wear volume | Archard: log10 V = log10 k̄ + log10(L·S), slope fixed at 1 | the intercept only |
| Track width | log10 w = a + b·ln L (Hertz predicts b ≈ 1/3 × ln-to-log10 factor) | a, b |

Report 95 % run-bootstrap intervals for the parameters, resampling replicate groups.

### 5.2 Required ML models

1. **ANN ensemble.** sklearn `MLPRegressor`, 1 hidden layer (3 or 4 tanh units), solver lbfgs, `max_iter=5000`.
   - L2 `alpha` ∈ {1e-3, 1e-2, 1e-1, 1}, with hidden size and alpha chosen by inner GroupKFold(5) on the training
     fold.
   - 20 seeds: prediction = mean, uncertainty = SD.
   - Standardise inputs and target.
2. **GPR.** sklearn `GaussianProcessRegressor`, kernel ConstantKernel × Matérn(ν = 2.5, one length scale per input)
   + WhiteKernel, `normalize_y=True`, `n_restarts_optimizer=5`, standardised inputs. Uncertainty = predictive SD.
3. **TabPFN** (`TabPFNRegressor`, default settings, `device="cpu"`). At setup, try `pip install tabpfn`; it pulls
   in torch and downloads its model weights on first use, so it needs internet once. If it still cannot be imported,
   log "not run (install tabpfn)" and carry on. It must
   plug into the same evaluation code.
4. **Physics-informed hybrids (the final ML models).**
   - Fit the physics baseline on the training fold.
   - Fit the ANN ensemble / GPR (and TabPFN if available) on its **residuals** with inputs ln L, ln f, ln t.
   - Prediction = physics + residual model.
   - Name them "ANN + physics" and "GPR + physics".

### 5.3 Comparison models

All on ln L, ln f, ln t:
- constant (training mean);
- linear power law (OLS);
- Ridge (alpha by inner CV);
- SVR (RBF; grid C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.1, 1}, ε ∈ {0.01, 0.05, 0.1});
- shallow XGBoost (max_depth ∈ {1, 2}, n_estimators ∈ {200, 400}, lr ∈ {0.03, 0.05}, subsample 0.8), if installed;
- ANN, GPR and TabPFN on the raw inputs (not physics-informed).

## 6. Evaluation (`src/tribo_final/evaluate/`)

Validation protocols, each applied to every model × target:

| protocol | data | groups | role |
|---|---|---|---|
| **P1 (primary)** | all 23 runs | leave-one-group-out; R04/R14/R20 = one group (21 groups) | main score |
| **P2 (stability)** | the 18 training runs only | same LOGO | a real gain must appear in P1 **and** P2; models that only win in P1 are flagged "unstable" |
| **P3 (hold-out check)** | train on the 18, predict the 5 test runs once | – | report the scores, and state that these runs were inspected during development, so P3 is a check, not a sealed test |

Metrics per model × target × protocol (on the target's modelling scale):
- n, R² (pooled out-of-fold), RMSE, MAE, max |error|, bias;
- Pearson r, Spearman ρ, calibration slope (measured on predicted; ideal 1);
- % within ±1 and ±2 replicate SD (noise floor = SD of R04/R14/R20);
- 95 % interval coverage and mean width (ANN ensemble, GPR, OLS prediction interval);
- **tercile confusion matrix**: bin measured and predicted values into low/mid/high by the terciles of the measured
  values; report accuracy and Cohen's κ.

Also compute:
- **1-SE tie rule:** a model is "tied with the best" if its mean squared-error difference with the best model is
  within one standard error.
- **Permutation test** (2000 shuffles of the target) for the physics baselines and the final hybrids → p-value.
- **Replicate noise floor** per target, and the "spread over runs / replicate SD" ratio (signal-to-noise).
- **Interpretation:**
  - GPR length scales;
  - ANN permutation importance (group-permuted);
  - power-law exponents with bootstrap CIs.

**Final model selection rule** (implement it in code, so the README states the winner from the results, not from
this prompt):
1. Among the required ML models (ANN, GPR, TabPFN, raw or hybrid), keep those that beat the constant model in **both**
   P1 and P2.
2. Among those, choose the one tied with the best P1 RMSE that has the most stable P2 score, preferring hybrids
   (physics-consistent, never far from the physics law).
3. Report it next to the physics baseline.
4. If no ML model beats the physics baseline robustly, say so plainly. The ML models then serve to:
   - confirm the physics trend;
   - quantify uncertainty;
   - show that there is no further structure in the data.

## 7. Comparison studies (each gives a CSV table, one figure, and a one-line verdict written into the README)

| ID | question | what to compute |
|---|---|---|
| C1 | Which COF definition? | first 10 s, 0.5–2 m, end (second half), last 10 s, whole run, instrument mean: replicate SD, signal-to-noise, physics-law R² (P1/P2), frequency bias of the instrument mean (its ratio to ours vs f) |
| C2 | Which wear measure? | most-worn location vs mean of locations; with vs without robust levelling; TA depth rule; max depth; track width: runs measurable, replicate SD, best R² (P1/P2), wear-volume exponents vs Archard (L¹, S¹) |
| C3 | Which model? | all models × targets × P1/P2/P3 + parameter count + uncertainty yes/no + physically consistent yes/no → verdict per model ("chosen", "unstable", "no better than constant", "worse than physics", …) |
| C4 | Old pipeline vs final | old = ANN on ln L, ln f, ln t for COF_end and log10 k (mean of locations, no robust levelling), evaluated under P1 vs the final models: metric table + parity plots side by side |
| C5 | Does the literature data help? | load the paper table (k_paper = V/(L·v·200 s); v_ours = 4·A·f). Compare trends (power-law exponents of COF and k vs load and speed, ours vs paper, with CIs). Strategies, scored only on our runs under P1 with the paper rows always in training: naive pooling (power law / GPR / ANN), pooled with a study offset, our power law shrunk toward the paper's slopes (strength by inner CV), paper slopes imposed, GPR/ANN with a study flag. Note that the paper's COF (0.002–0.007) and wear volumes (up to 1620 mm³ in 200 s, about the whole pin volume) are implausible as published |
| C6 | TA script | see section 8 |

## 8. The TA script (C6)

Run `inputs/analyze_wear.py`'s functions unchanged on all runs (without writing files into `testdata/`). Then run a
fixed copy, and compare both with ours. Known bugs to demonstrate with numbers:

| # | bug | evidence on our data |
|---|---|---|
| 1 | Orientation chosen from the unlevelled map (tilt decides) | 25 of 92 scans transposed, so the "profile" runs along the track. Make a figure for R24.bcrf: the reported 22.5 µm "groove" is a pit; with the correct axis the same code gives groove 0 |
| 2 | Points shallower than the cutoff zeroed one by one | groove area 0 although depth 3–5 µm, e.g. R05, R21 |
| 3 | Pile-up summed over the whole image outside the groove | – |
| 4 | Shoulder (reference) rows never checked to be unworn | – |
| 5 | Scan picked by deepest point (texture valleys) | – |
| 6 | COF = mean of the instrument \|Fx\|/Fz channel | about 12 % low; the ratio to ours falls with frequency (r ≈ −0.68); deceleration second not trimmed |
| 7 | Hard-coded R01 parameters; friction log = first `*.csv` (can be its own output); outputs written into the data folder | – |

Repeat runs R04/R14/R20 give k = 2.3e-6 / 7.0e-5 / 2.6e-6 with the original script, a 30× spread for identical
tests. Quantify replicate SD and model R² for original, fixed and ours.

## 9. Figures (presentation pack)

Save to `outputs/figures/` as PNG (300 dpi, 16:9-friendly) and SVG, numbered in slide order.
- **Style:**
  - one message per figure, with the takeaway in the title;
  - fonts ≥ 12 pt at slide size;
  - colour-blind-safe palette;
  - the same colour for the same model or method everywhere;
  - no dual y-axes.
- **Every figure** is listed in `outputs/figures/INDEX.md`, with a one-line caption and its data source.

1. Pipeline overview diagram (raw files → extraction → dataset → models → evaluation).
2. Test matrix: the 23 runs in L–f–t space (training/test/replicates marked; missing runs shown hollow).
3. Raw friction signal of one run (R04): Fx and Fz vs time. Annotate the static step, ramps, offset, and reversals,
   and show how the per-cycle half-range COF is taken.
4. COF vs sliding time for all 23 runs (small multiples ordered by load; per-cycle dots + running median; the
   first-10 s window shaded).
5. Height-map example (R04 location m) with the detected core, band and reference rows, plus the across-track
   profile with groove/pile-up shading.
6. Robust levelling effect: one scan with a pit (R24), plain vs robust profile.
7. Wear per run: k (most-worn location) vs load, with below-LOD upper bounds and not-measurable runs marked.
8. C1: COF definitions (replicate SD and physics-law R², P1 and P2).
9. C2: wear measures (runs measurable, replicate SD, R²).
10. Physics fits:
    - COF_first10s vs load with μ₀ + c·L^(−1/3) and its 95 % band;
    - COF_end vs load for contrast;
    - track width vs load (log-log) with the fit;
    - wear volume vs L·S with the Archard line.
11. C3 model comparison: dot plot of P1 R² per model per target, with P2 as a second marker; constant and physics
    baselines as reference lines.
12. Stability: P1 vs P2 R² scatter for every model × target (unstable models stand out).
13. Parity plots for the final models (out-of-fold), with 95 % intervals.
14. Residuals vs load, frequency and time for the final models.
15. Tercile confusion matrices: final vs old pipeline vs best plain ANN.
16. Uncertainty: interval coverage vs nominal 95 %, and interval width per model.
17. Interpretation: GPR length scales and ANN permutation importance (load dominates).
18. Prediction surfaces of the final COF and width models over L × f at a mid t, with uncertainty.
19. C4: old pipeline vs final (metric table + parity side by side).
20. C5: literature vs our trends (relative COF and k vs load and vs speed), and the scores of the pooling strategies.
21. C6: TA script: the R24 flip figure; replicate spread original/fixed/ours; COF ratio vs frequency.
22. P3 hold-out: predictions vs measured for the 5 test runs (final vs old).

## 10. Reports

- **`outputs/report/final_report.pdf`** (reportlab; tables and figures, minimal prose):
  - title page;
  - data summary;
  - the final model card per target (equation, parameters with CIs, P1/P2/P3 metrics, uncertainty);
  - comparison tables C1–C6 with verdicts;
  - all figures.
- **`outputs/report/slides_data.md`.** One section per planned slide (about 12–15 slides), each with 3–5 key numbers
  copied from the result tables and the figure to use. Example: "Slide 7 – COF model: COF_first10s = μ₀ + c·L^(−1/3),
  μ₀ = …, c = …, R² P1 = …, P2 = …, permutation p = …; figure 10".
- **`outputs/tables/`.** Every table as CSV:
  - runs;
  - per-model metrics for each protocol;
  - final parameters;
  - each comparison;
  - confusion matrices;
  - permutation tests;
  - noise floors.

## 11. README.md (written for a newcomer; auto-fill the numbers from the result tables at the end of `run_all.py`)

1. **Project in one paragraph.** What is predicted and why (COF and wear of 304 SS vs load, frequency, time).
2. **Quick start.** Install, the `run_all.py` command, the expected runtime, and where the outputs are.
3. **Folder structure** (tree with one line per item).
4. **The experiment.** DOE, protocol, which runs exist and which are missing, and the replicates.
5. **Raw data formats.** CSV quirks, .lrx, .bcrf, .info, and which files are ignored and why (the 150 GB of .Raw).
6. **Extraction explained step by step.**
   - COF: why half-range per cycle, why the instrument column is not used, why the first 10 s.
   - Wear: the band method with a diagram reference, robust levelling, validity, LOD, most-worn location, k = V/(F·S)
     (Archard specific wear rate, profilometric cross-section method; cf. ASTM G133).
7. **Dataset.** Data dictionary for `runs.csv`.
8. **Models.** Physics baselines, the required ML models, the hybrids, and the comparison models, each with
   settings.
9. **Validation.** P1/P2/P3, metrics glossary (R², RMSE, coverage, tercile confusion matrix, 1-SE rule, permutation
   test), and why grouped CV.
10. **Results.** Final model per target with numbers, the noise floor, and what the models say (load dominates;
    frequency and time not detectable; k constant per Archard; COF follows the adhesion law early in the run).
11. **Why these choices** (the C1–C6 verdicts in plain language: why not the end-of-run COF, why not mean-of-locations
    wear, why not plain ANN/GPR, why not the literature data, why not the TA script, why the old pipeline is
    replaced).
12. **Limitations.**
    - Short runs that never reach steady state.
    - Rough plates and only about 1 µm-deep grooves; 6 runs below the detection limit and 2 not measurable.
    - Only 23 runs.
    - Two test sessions, not modelled as instructed.
    - The high-load tracks exceed the field of view.
13. **Reproducibility.** Seeds, versions (write `outputs/environment.json`), caching, and how to rerun one stage.
14. **Glossary.**

## 12. Code structure

```
final version/
  README.md  requirements.txt  config.yaml  run_all.py
  inputs/           run_sheet.csv, ponnuswamy2024_table5.csv, analyze_wear.py (copied, read-only)
  src/tribo_final/
    io/        csv_reader.py, lrx.py, bcrf.py, info.py, layout.py (run discovery, location parsing, DOE matching)
    extract/   cof.py, wear_band.py, dataset.py, cache.py
    models/    physics.py, ann.py, gpr.py, tabpfn_model.py, hybrid.py, baselines.py, registry.py
    evaluate/  cv.py (P1/P2/P3, groups), metrics.py, selection.py, permutation.py, interpret.py
    compare/   c1_cof.py, c2_wear.py, c3_models.py, c4_old.py, c5_literature.py, c6_ta.py
    plots/     style.py, figures.py (one function per figure)
    report/    pdf.py, slides.py, readme.py
  tests/
  outputs/   (generated: data/, tables/, figures/, report/, logs/)
```

`run_all.py` stages: `extract → dataset → models → evaluate → compare → figures → report → readme`. Each stage can be
run alone (`python run_all.py --stage figures`) and reuses cached results. Log to `outputs/logs/run.log`.

## 13. Tests (pytest; must pass before you finish)

**Synthetic tests:**
- 100 Hz CSV with a static step + one sliding step + Fx offset + reversal spikes: the static step is excluded, the
  offset cancels (COF within 3 %), cycles within ±2.
- BCR-F round trip.
- `.lrx` parsing.
- Location naming (`R01a_13` → a, `R01_13` → m).
- DOE matching by value (a folder named R09 holding the R24 condition maps to R24).
- Band method on 6 synthetic rough surfaces (Sq 3 µm, 1 µm-deep flattened band, pile-up ridges): median groove bias
  < 20 %, width bias < 10 %, no track found on an unworn surface, a one-sided reference flagged.
- Robust levelling ignores a 25 µm pit.
- CV never puts R04/R14/R20 in different folds.
- A hybrid with zero residual equals the physics model.
- The ANN ensemble is deterministic for fixed seeds.

**Real-data acceptance tests** (skip if `testdata/` is absent; tolerances in brackets). If a value is outside its
tolerance, investigate before relaxing it.

| check | expected (tolerance) |
|---|---|
| run folders found | 23; parameters from .lrx agree with the DOE row of the same ID for all 23 |
| R04 parameters | 32 N, 7.5 Hz, 84 s |
| R04 cycles | 626 (± 10) |
| R04 COF_end | 0.155 (± 0.015) |
| R04 track width | 414 µm (± 60) |
| R01 static step | instrument COF ≈ 0.049 |
| wear status (robust levelling) | 15 detected (± 1); below LOD: R04, R06, R07, R09, R10, R29; not measurable: R12, R19 |
| replicate noise floor | R04 is below LOD, so the wear replicate SD comes from R14/R20 only (≈ 0.06 dex); COF replicate SD ≈ 0.006 |

**Expected headline results** (P1, approximately):

| result | P1 | P2 |
|---|---|---|
| COF_first10s physics law R² | +0.36 | +0.22 |
| COF_end physics law R² | +0.03 | – |
| track width physics R² | +0.26 | +0.40 |
| wear volume (Archard, constant k) R² | +0.19 | – |
| typical k | ≈ 2.9e-5 mm³/(N·m) ×/÷ 2.4 | – |
| plain GPR on COF_end (example of an unstable model) | ≈ +0.36 | ≈ −0.40 |

- No model beats a constant for k.
- Literature trends: paper COF ∝ L^+0.52 v^−0.69 vs ours L^−0.06 v^−0.02; every pooling strategy scores worse than
  our data alone.

If your numbers differ a lot from these anchors, find out why and explain it in the README.

## 14. Definition of done

- `python run_all.py` runs end to end from a clean `outputs/` and finishes under 30 min.
- `pytest` passes.
- Generated:
  - all 22 figures;
  - `final_report.pdf`, `slides_data.md`, all tables;
  - the README with auto-filled numbers.
- The README states the selected final model per target, with the reason taken from the selection rule. It also
  gives an honest account of what the data can and cannot support.
- Nothing outside `final version/` was changed.
