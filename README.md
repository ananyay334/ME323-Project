# ME 323 — 304 stainless steel, dry reciprocating, ANN modelling

Per-cycle COF and specific wear rate of 304 stainless steel at two checkpoints per run,
extracted from the raw tribometer and height-scan files (`tribo_extract/`) and modelled with
a shallow ANN ensemble plus benchmarks (`ml/`).

> **Campaign data (13 and 16 Oct 2026)** is in `testdata/<RunID>/`, and the results are in
> [`RESULTS.md`](RESULTS.md). The real files differ from the plan below in a few ways, all
> handled automatically:
> - one continuous sliding step per run, split at t_run/2 for the two COF checkpoints;
> - post-run scans only, so wear exists for checkpoint 2 only;
> - logging at 100 Hz;
> - BCR-F height maps;
> - load, frequency and time read from each run's own `.lrx`, not from the DOE or the folder name.
>
> Runs with no folder (R00, R03, R15, R16, R18, R23, R27) are skipped.

```
323_project/
├── config/extraction.yaml      thresholds + test geometry (A = 5 mm, Ø6 mm ball, 304 SS)
├── config/file_layout.yaml     file roles, scan location, checkpoint ladder (confirmed on the real files)
├── docs/FILE_FORMATS.md        what each delivered file is (.csv .lrx .bcrf .info; .Raw never read)
├── testdata/<RunID>/           the campaign run folders (raw_dir in config/extraction.yaml)
├── RESULTS.md                  campaign results: extraction summary, CV and test scores
├── data/
│   ├── run_sheet.csv           DOE: R00–R29 (+ top-ups T01… appended with Role = topup)
│   ├── raw/TRIAL_2026-09-22/   practice run (test fixture)
│   ├── raw/_unassigned.csv     files `organize` could not place
│   ├── processed/              ML tables (written by the pipeline, don't edit)
│   │   ├── runs_checkpoints.csv   one row per run × checkpoint  → ML (24 training runs × 2)
│   │   ├── cof_cycles.csv         one row per sliding cycle
│   │   └── wear_locations.csv     one row per run × checkpoint × scan location
│   └── topup_candidates.csv    pre-registered top-up runs (`python -m ml.topup`)
├── reports/runs/<ID>/          QC figures + summary.json for every run — look at these
├── tribo_extract/              extraction package
├── tests/                      regression tests (+ synthetic CSV / height-map generators)
└── ml/                         models (see ml/README.md)
```

## The DOE

- **Specimen.** 304 stainless steel plate, 42 × 42 mm, two polished faces F1/F2. The
  counter-body is a new hardened bearing-steel ball (Ø6 mm, R = 3 mm) for every run.
- **Factors.** Load L = 10–100 N, frequency f = 4–15 Hz, run time t_run = 60–120 s.
- **Kinematics.** Stroke amplitude A = 5 mm (±5 mm, track length ℓ = 10 mm); N = f·t,
  S = 4·A·N = 0.02·f·t m.
- **Run sequence.** Each run is two equal slide steps of t_run/2. The track is scanned inline
  before sliding (base), after step 1 (checkpoint 1, 50 %) and after step 2 (checkpoint 2,
  100 %), at 4 locations along the track (m, a, b, c).
- **Logging.** 1000 Hz, averaging 1. There is a 1 s acceleration and a 1 s deceleration ramp at
  every step start and stop.
- R00/R01 are completed pilot runs. R04/R14/R20 are the centre replicates. R24–R29 are the
  sealed test runs.

## Setup (once)

```bash
cd ~/Desktop/323_project
pip install -r requirements.txt
python -m pytest -q            # extraction + ML tests, ~1-2 min
```

## Per-run workflow

1. **Run sheet.** The run is already in `data/run_sheet.csv`. After the run, fill in the manual
   columns: `Hardness_HV`, `Ra_um`, `Ambient_T_C`, `RH_pct`, `Ball_scar_d1_um`,
   `Ball_scar_d2_um`, `Notes`. `Include = N` excludes a run everywhere.
2. **Files per run.** There are about 10–12 files per run (details in `docs/FILE_FORMATS.md`):

   | file | example | role |
   |---|---|---|
   | COF CSV (RTEC export, both steps) | `R02.csv` or `<load>_<freq>.csv` (e.g. `10_10.csv`) | `cof_csv` |
   | height scans, 4 locations × base/cp1/cp2 | `R02.raw`, `R02_a.raw`, `R02_b.raw`, `R02_c.raw` | `height` |
   | previews / metadata of a scan | `R02_a.png`, `R02_a.txt` | `preview`, `metadata` |

   A name without a suffix is location `m`; `_a`/`_b`/`_c` are the other three locations. The
   checkpoint comes from the first rule that works: a name token (`_t60s`, `base`, `cp1`,
   `_50`, `cp2`, `_100`), then a sub-folder name, then the metadata timestamp, then the file
   mtime order (3 scans: base, cp1, cp2; 2: cp1, cp2; 1: cp2).
3. **Organize a dump.** This copies a flat dump into the run folders:
   ```bash
   python -m tribo_extract organize ~/Downloads/session3_dump
   ```
   The run ID is read from the name (`^([RT]\d{2})(?:_([a-c]))?`). A `<load>_<freq>.csv` is
   matched to the run sheet only when the pair is unique; 32 N / 7.5 Hz is R04, R14 and R20, so
   those files need the run ID in their name. Unmatched, ambiguous or clashing files are listed
   in `data/raw/_unassigned.csv` and are never guessed. Sub-folders and mtimes are kept.
4. **First real run only (Part 0).** This prints every file's role, location and checkpoint,
   and how each RAW file decodes:
   ```bash
   python -m tribo_extract discover data/raw/R00
   ```
   Then put the RAW format (nx, ny, dtype, header, z scale, pixel size) in the `raw:` block of
   `config/file_layout.yaml`, fill the TBDs in `docs/FILE_FORMATS.md`, and set
   `wear.track_angle_deg` in `config/extraction.yaml` to the measured track angle.
5. **Extract.**
   ```bash
   python -m tribo_extract extract               # all runs; or --run R02 R03
   ```
6. **Check.** Look at `reports/runs/<ID>/cof_cycles.png` and `wear_<loc>_<cp>.png`, and at the
   `status`, `issues` and `flags` columns of `runs_checkpoints.csv`.

Single files, no run sheet needed: `python -m tribo_extract inspect R02.csv --load 15 --freq 4`
or `python -m tribo_extract inspect R02_a.raw`.

## How the targets are computed

**COF (per cycle, offset-independent).**
- Sliding steps are the RecipeStep blocks whose median |COF| is at least 30 % of the highest
  step median, in time order: the first is checkpoint 1, the second checkpoint 2.
- Cycles are counted over the full step (achieved frequency = cycles / step duration). Each
  cycle runs between upward zero-crossings of the drive position or, without a position
  channel, between every second sign reversal of the 20 ms moving median of Fx (hysteresis
  ±0.2 × half-range).
- After trimming the 1 s ramps, `COF_c = (P90(Fx) − P10(Fx)) / 2 / mean(Fz)` per cycle.
  Without an Fx channel, the median |COF column| is used instead (flag `cof_from_cof_column`).
- `COF_steady` = median of COF_c over the last 50 % of the step's cycles.
- `Distance_m` = 4 · A_achieved · Cycles_cum / 1000. A_achieved comes from the position
  channel, or is the nominal 5 mm (flag `stroke_nominal`).

**Wear (groove, per checkpoint).**
- Every location's scan is read with `tribo_extract/height_io.read_height`: RAW via the format
  ladder, text grids directly, and screenshots only for PNG-only scans.
- The location's base scan is subtracted after alignment. `measure_wear_scar` then gives the
  groove area below the shoulder reference and the pile-up area beside it.
- `Groove_volume_mm3` = mean groove area × ℓ, with the SE over locations.
  `Pileup_volume_mm3` is computed the same way, and `Net_volume_mm3` = groove − pile-up.
- `k_mm3_per_Nm` = Groove_volume / (L · Distance_m); `k_net_mm3_per_Nm` uses the net volume.
- Check: V_width = ℓ·w̄³/(12R); flag `width_model_mismatch` if it differs by more than 50 %.
- Ball wear: V_ball = π·d⁴/(64R) from the two ball-scar diameters (checkpoint-2 row).
- A groove shallower than the detection limit gives `Wear_detected = False`, an empty volume
  and `Wear_Volume_LOD_mm3` (flag `below_LOD`). Treat it as "< LOD", not as zero.
- `XYZ.Z Depth` in the CSV is a servo diagnostic (in `summary.json`), never wear.

**Campaign data: what changes (config `wear.method: band`, `cof.split_single_step: true`).**
- *Parameters.* `run_parameters` reads Load_N, Freq_Hz and Run_time_s from the run's `.lrx`
  (else measures them from the CSV). `match_doe` then finds the DOE row with the same values
  (within 2 % / 2 % / 3 %); only the role, face and position come from it. If the folder name and
  the values disagree, `issues` says `files_match_doe_row_<ID>`. Replicates (`Replicate_of`) are
  runs with identical values in their files.
- *Sliding steps.* A step slides if the logged reciprocating frequency is above 0.5 Hz or, without
  that channel, if its Fx half-range / Fz is at least 30 % of the largest. This keeps the 2 s
  static loading step, with its ≈1 N Fx offset, out. A single sliding step is split at its half:
  cp1 = first half and cp2 = whole run (flag `single_step_split_at_half`).
- *Wear on rough plates (`measure_track_band`).* The unworn plate is rough (Sa ≈ 3 µm on
  13 Oct, 6–9 µm on 16 Oct, mostly long-wavelength waviness). The track is a *smoother*,
  slightly sunken band with pile-up ridges, so it is found by local row roughness, not by depth:
  - the core is the rows smoother than halfway between unworn and track roughness;
  - the band is the core ± 40 %, tightened to the outer feet of the pile-up ridges;
  - the reference is a plane fitted to the unworn rows, then re-fitted outside the final band;
  - groove = area below the reference inside the band, pile-up = area above it;
  - `Track_width_um` = core width.

  A location needs ≥ 110 µm of unworn surface on *both* sides; otherwise the tilt is extrapolated
  and the groove is not trusted (flag `reference_one_sided`). This happens on the 89–100 N tracks,
  which fill the 1.125 mm field of view. The run's groove is "detected" if its mean over valid
  locations exceeds 2 × its uncertainty. The uncertainty comes from the unworn texture
  (50 µm correlation length). Below that the run is censored (< LOD), and with no valid location it
  is not measurable (NaN).
- *Track width* (`Track_width_um`, ML target `log10_w`) is the robust wear measure here:
  - replicate SD 0.07 dex, against 0.36 dex for the groove volume;
  - available for every run;
  - not affected by the plate's waviness.
- `Test_date` comes from the CSV header. `python -m ml.session_effect` checks the test-day effect
  on the training runs.

### Validation (tests/test_extraction.py, tests/test_campaign.py)

| check | result |
|---|---|
| synthetic 1000 Hz CSV: square-wave Fx, μ = 0.15, 0.29 N offset, noise, reversal spikes, 1 s ramps | COF_steady within 2 %, cycles within ±1 per step |
| 4 locations × (base, cp1, cp2) height maps with known parabolic grooves | groove volume within 5 %, k = V/(L·S) |
| headerless float32 RAW, with and without companion metadata | decoded exactly (r > 0.999) |
| scars painted into the real practice screenshot (single scan / baseline difference) | area within ±10 % / ±3 % |
| synthetic BCR-F file; location names `R01a_13`, `R04c`, `R01_b`; `.lrx` parameters | read exactly |
| static step with 1 N Fx offset + one continuous sliding step (100 Hz) | static step excluded; split at t/2; COF within 3 % |
| folder named R09 holding the R24 condition; CSV-only parameter fallback; replicates by value | matched to R24 / R04; replicates found |
| band method on 6 rough synthetic surfaces (Sq 3 µm, 1 µm-deep flattened track) | median groove bias < 20 %, width < 10 %; no track on unworn surface; one-sided reference flagged |
| real `testdata/R04` | 32 N / 7.5 Hz / 84 s; COF 0.13–0.17; ≈625 cycles; width 350–470 µm |

## Output tables (`data/processed/`)

- **`runs_checkpoints.csv`** (one row per run × checkpoint; the ML table):
  - run-sheet columns: `Experiment_ID, Role, Replicate_of, Session, Face, X_mm, Y_mm, Load_N, Freq_Hz, Run_time_s`
  - kinematics and COF: `Checkpoint` (1, 2), `Time_s` (= t_run/2, t_run), `Cycles_cum, Distance_m, Freq_achieved_Hz, Stroke_amp_mm, Fz_mean_N, COF_steady, COF_std`
  - wear: `Groove_volume_mm3, Groove_volume_se_mm3, Pileup_volume_mm3, Net_volume_mm3, k_mm3_per_Nm, k_net_mm3_per_Nm, Wear_detected, Wear_Volume_LOD_mm3, V_width_mm3, V_ball_mm3, n_locations`
  - manual measurements: `Hardness_HV, Ra_um, Ambient_T_C, RH_pct`
  - QC: `status, issues, flags`
- **`cof_cycles.csv`**: `Experiment_ID, Step, Cycle, t_s, COF_c, Fz_N, Halfrange_N, Amp_mm`.
- **`wear_locations.csv`** (one row per run × checkpoint × location): the file used, read
  method, wear method, groove / pile-up / net area, width, depth, LOD, track angle, preview r
  and flags.

## Modelling (`ml/`, details in `ml/README.md`)

```bash
python -m ml.synthetic && python -m ml.train --data synthetic --quick   # smoke test (< 1 min)
python -m ml.train                        # CV on the training runs -> ml/results/<timestamp>/
python -m ml.train --config ml/configs/width.yaml          # same with track width instead of k
python -m ml.train --config ml/configs/width.yaml --final  # its sealed test -> ml/results/final_width/
python -m ml.session_effect               # test-day effect on the training runs
python -m ml.campaign_model               # final model: all 23 runs, test day as blocking factor
                                          #   -> ml/results/campaign_model/report.md
python -m ml.topup --n 4                  # next runs -> data/topup_candidates.csv (pre-registered)
python -m ml.train --final                # refit, score R24-R29 once -> ml/results/final/
```

- **Models.** Inputs are ln L, ln f, ln t_ckpt. Targets are log10 k and COF_steady at each
  checkpoint, or log10 track width and COF_steady with `ml/configs/width.yaml`. The ANN is an
  ensemble of 20 seeds (3 → 3–4 tanh → 2), trained with a numpy Adam backend (or torch if
  installed: `backend: torch`). Benchmarks: GPR (Matérn-5/2), SVR, TabPFN (skipped if not
  installed), shallow XGBoost, and Ridge (reference).
- **Validation.** 5-fold group CV by run on the training runs (R04/R14/R20 count as one
  group), repeated over 5 seeds, with nested tuning.
- **Top-up workflow.** `ml.topup` picks up to 10 runs where the GPR σ and the ANN ensemble
  spread are largest, gives them free plate positions and writes their predictions before
  the runs are made. Append the chosen rows (T01…) to `run_sheet.csv` with `Role = topup`,
  run them, `organize`, `extract`. The next report then shows predicted vs measured for them,
  and they join the training set.
- **Campaign model (`ml/campaign_model.py`).** With no further runs possible, this is the model to report.
  It uses all 23 runs, one row per run, and leave-one-group-out CV. The test day, read from the CSV header, is
  a blocking factor; plate roughness is offered as the physical alternative. It compares input sets × models
  (linear, ridge, GPR, ANN ensemble) and flags candidates tied with the best (1-SE rule). It fits the
  physics-shaped models COF = μ₀ + c·L^(−1/3) + day offset and w = A·L^b·(day factor), with run-bootstrap
  intervals. The sealed-test result from `--final` stays the pre-registered number.
- **Test seal.** The test runs are loaded only by `--final`. It refuses to run again once
  `ml/results/final/` exists, unless `--force` is given.

## Assumptions

- Campaign data (Oct 2026): config/file_layout.yaml and docs/FILE_FORMATS.md are confirmed on the real files. RTEC `.Raw` (interferogram stacks), `.rtec` and `.enc` are role `ignore`. The generic RAW ladder below is kept for other instruments.
- Run parameters come from the run's files (`.lrx`, else CSV) and the DOE row is matched by value (Load 2 %, frequency 2 %, time 3 %; a same-ID row is preferred, then a center_replicate). If the matched row is another ID, its position and manual columns are not copied. Without a match, Role = interior (issue `not_in_doe`).
- Every run folder matching the run-ID pattern is processed, unless the run sheet has Include = N for it. Missing runs are simply absent.
- One sliding step is split at t_run/2. COF_steady at cp1 is the median over the second quarter of the run, at cp2 over the second half.
- All scans are post-run (cp2); cp1 rows carry COF only.
- Band wear method parameters (config `wear.band`): high-pass window 15 px, unworn reference = outer 15 % of rows, core level 0.5, minimum contrast 0.30, band extension 40 %, texture correlation length 50 µm, minimum one-sided reference 110 µm, re-levelling margin 10 µm.
- Run-level wear: track width is the mean over locations where a track was found. Groove, pile-up and net areas are means over locations with a two-sided reference. LOD = 2·sqrt(mean u²)/sqrt(n).
- The ANN numpy backend uses full-batch Adam (β = 0.9/0.999, ε = 1e-8, L2 weight decay added to the gradient, as torch does). Its hand-written gradients are checked against finite differences in ml/tests.
- Lateral pixel size falls back to wli.fov_um["10x"] / nx (the old WLI's 10x field of view) when neither file_layout.yaml nor companion metadata gives it.
- wear.track_angle_deg stays `auto` until it is measured on the first real scan.
- The CSV `position_mm` pattern is provisional (a position / displacement / velocity channel that is not the XYZ stage or Z depth). A channel moving less than 10 % of A is ignored (flag position_static), and a velocity channel counts cycles but gives no amplitude.
- Unknown z scale: the unit factor (µm, nm, mm, m) that puts the levelled height range within 0.1–100 µm is used (`z_scale_source = unit_guess`).
- Brute force ranks candidates by the median absolute row and column difference over the 5–95 % value spread. This is a robust form of normalised total variation; a plain std normalisation picked a wrong offset when header bytes were read as heights.
- Preview validation tries the four flips of the RAW map and keeps the best r. A preview that read_wli_screenshot cannot decode falls back to the 0.1–100 µm range check.
- A scan is the set of files that share a name up to the first "." (preview or metadata suffixes such as `_preview` are stripped). A location's PNG-only scans are used only when it has no height file.
- A checkpoint rule is used for a location only if it assigns every scan of that location uniquely. More than 3 scans, or ties, leave the location unassigned (issue checkpoint_unresolved). `_t<sec>s` maps 0 s to base and any other time to the nearer of t_run/2 and t_run.
- organize also accepts top-up IDs T01… (`^([RT]\d{2})`). Dump sub-folders are kept below the run folder, mtimes are preserved (shutil.copy2), an existing different file of the same name is never overwritten (listed as name_clash), and `_unassigned.csv` is rewritten on every call.
- COF: sliding steps are contiguous RecipeStep blocks; with more than two, the first two are used (flagged). A cycle's COF is used only if the whole cycle lies inside the trimmed window. With no reversal signal, cycles are counted from the nominal frequency (flag cycles_nominal).
- Fx reversal detection centres the 20 ms moving median on (P10 + P90)/2 before the sign test, so the Fx offset cannot bias it.
- cof_cycles.csv: Step = sliding-step index (1 → checkpoint 1, 2 → checkpoint 2); t_s = cumulative sliding time at the middle of the cycle; a cycle's Fz falls back to Load_N when no Fz channel exists.
- Several COF CSVs in one run folder are appended in mtime order (noted in `issues`).
- A checkpoint without counted cycles (no CSV) uses the nominal Cycles_cum = f·t and A = 5 mm (flag distance_nominal).
- Wear per checkpoint: volumes and SE are means over the locations where the groove was detected, and n_locations counts all measured locations. Partial detection is flagged `detected_<n>_of_<m>_locations`. Wear_detected is left empty when no scan was measured.
- Both cp1 and cp2 of a location are measured against that location's base scan.
- V_ball_mm3 is reported on the checkpoint-2 row only, because the ball scar is measured after the run.
- process_all writes all three tables, with headers only when there is no data. The TRIAL folder does not match the run-ID pattern and is skipped.
- The old 1 Hz `extract_cof` is kept because test_cof_trial (practice run) depends on it. process_all no longer writes COF(t).
- The old tables runs_targets.csv, cof_timeseries.csv and wear_trajectory.csv were removed from data/processed; they remain in git history.
- runs_checkpoints.csv has no Include column (it is not in the A6 list). Include = N runs are skipped at extraction, so the ML Include filter only acts when the column is present.
- log10 k LOD (used by censoring.mode lod_half) = Wear_Volume_LOD_mm3 / (Load_N · Distance_m).
- ANN: each member draws its own validation groups (round(20 %) of the training groups; none with fewer than 5 groups). The members are trained as one batched network, which is equivalent to separate training. ANN interval coverage uses the ensemble SD only.
- ANN inner-CV score (two outputs) = mean over the targets of RMSE / target SD.
- Group-permutation importance: in every outer fold of the first repeat, 10 permutations on the held-out rows. Each row takes the input value of the same checkpoint in the group that its own group is mapped to.
- Ridge exponents use the alpha chosen by inner GroupKFold on all training rows; every bootstrap resample is refitted with that alpha (closed form, identical to StandardScaler + Ridge).
- Noise floor = sqrt(mean over the checkpoints of the variance (ddof 1) of R04/R14/R20).
- --final reruns the CV to select the models, then refits them. Results go to ml/results/final/ (synthetic: final_synthetic/), and --force deletes and rewrites that folder.
- Top-up details:
  - distances use the mean and SD of ln L, ln f and ln t_run over the training runs;
  - "existing runs" means every run-sheet row, including the test runs and earlier top-ups;
  - the GPR σ is the latent σ, without the fitted noise;
  - candidate L and f are rounded to 0.01 before the constraint check;
  - free positions are taken in the order F1 → F2, then X, then Y;
  - the pre-registered predictions are the ANN ensemble mean ± SD at both checkpoints;
  - synthetic data writes to data/synthetic/topup_candidates.csv.
- The random forest is dropped, because it is not one of the DOE's models.
- --quick keeps 1000 bootstrap resamples and 10 permutation repeats, since the quick spec lists neither.
- load_data(split="train") reads the whole runs_checkpoints.csv and drops the test-role rows right after parsing, before any other step.
- Synthetic data: a = −4.3 (k ≈ 1e-4 mm³/(N·m)) and LOD 3e-4 mm³. The 2 censored runs are the lowest-wear non-replicate training runs, censored at both checkpoints with LOD = 2 × their checkpoint-2 volume.
- requirements.txt: torch is optional (numpy ANN backend by default); tabpfn stays an optional commented line.
