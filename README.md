# ME 323 — Triboinformatics of carbon-modified maraging steel

Target extraction (COF and specific wear rate) from each tribometer run, feeding
the ML framework that will live in `ml/`.

```
323 proj/
├── config/extraction.yaml      all thresholds + test defaults (stroke, ball, FOV, …)
├── data/
│   ├── run_sheet.csv           DOE log: one row per Experiment_ID (inputs you control)
│   ├── raw/<Experiment_ID>/    the files the TA hands over for that run
│   └── processed/              ML-ready tables (written by the pipeline, don't edit)
│       ├── runs_targets.csv    one row per run   → wear-rate regime (N ≈ 100)
│       └── cof_timeseries.csv  one row per second → COF(t) regime (≈ 600 rows/run)
├── reports/runs/<ID>/          QC figures + summary.json for every run — look at these
├── tribo_extract/              the extraction package
├── tests/                      regression tests (+ synthetic wear-scar generator)
└── ml/                         model framework (next step)
```

## Setup (once)

```bash
cd "~/Desktop/323 proj"
pip install -r requirements.txt
# optional, lets the code read the colour-bar numbers off the screenshot by itself:
brew install tesseract && pip install pytesseract
python -m pytest -q            # 14 tests, ~1-2 min
```

## Per-run workflow

1. The run should already be in `data/run_sheet.csv` (R001–R100 are the 4×5×5 DOE
   grid). Fill in anything measured on the day: `Hardness_HV`, `Ball_ID`,
   `Ambient_T_C`, `RH_pct`, and the actual `Load_N` / `Freq_Hz` if they differ.
   **`Freq_Hz` must be there — the CSV does not record it**, and it sets the sliding distance.
2. Drop the files into `data/raw/<Experiment_ID>/` (any file names):
   * the tribometer `.csv`
   * the WLI screenshot `.png` of the wear track
   * *(recommended)* a baseline scan of the same spot before sliding, named `…_t0s.png`,
     and any checkpoint scans as `…_t60s.png`, `…_t300s.png`, …
3. Run `python -m tribo_extract extract` (or `--run R017` for one run).
4. Check `reports/runs/<ID>/cof.png` and `wear_*.png`, and the `status`, `issues`,
   `cof_flags`, `wear_flags` columns in `runs_targets.csv`.

Without tesseract, type the colour-bar min/max from the screenshot into the run sheet
(`cbar_min_um`, `cbar_max_um`), or per scan in `data/raw/<ID>/wli_scans.csv`
(columns `file, t_s, cbar_min_um, cbar_max_um`).

Quick look at a single file, no run sheet needed:
```bash
python -m tribo_extract inspect "data/raw/TRIAL_2026-09-22/Trail_Tribo.csv"
python -m tribo_extract inspect some_scan.png --stroke 6.25      # figure → reports/inspect/
```

## How the targets are computed

**COF** — from the `DAQ.COF (Fx)` column (the tribometer already divides Fx/Fz).
The sliding step is found automatically (the approach/loading step has COF ≈ 0.01).
The end of running-in is the last time the rolling mean leaves a band around the
second-half median; `COF_ss_mean` is the mean after that. Also written: median, std,
running-in time and peak, COF at 60/150/300/450/600 s, and a 1 Hz `COF(t)` series.

**Wear rate** — the screenshot's height map is rebuilt pixel by pixel: the software
shades the rainbow colours with the optical image, but the hue survives shading, so
each pixel's hue is looked up on the colour bar to get its height (overlay lines,
markers and text are masked out). Then:

1. level the map, find the sliding direction (or use `track_angle_deg`), rotate the
   track horizontal and average along it → one low-noise cross-section;
2. fit the unworn surface on the shoulders (tilt, plus curvature only when the data
   support it) and integrate the depression → worn area *A* (µm²);
3. `Wear_Volume_mm3` = *A* × track length (ASTM G133 approach);
4. `Specific_Wear_Rate_mm3_per_Nm` = *V* / (*F* · *S*), with *S* = 2 · stroke · f · t.

If a `t0s` baseline scan exists, the pipeline aligns the two scans and measures
the difference map (worn − baseline) instead. Surface texture and specimen form
cancel out, which makes a big difference for shallow scars. With ≥ 2 checkpoint scans it also fits
the slope of V(t) and the two-stage model of DOE eq. 4 (`Vdot_steady`, `dV_run`, `tau_trans`).

A scar shallower than the surface roughness can't be told apart from texture in a single
scan. Those runs get `Wear_detected = False` and `Wear_Volume_mm3` left empty, with the
detection limit in `Wear_Volume_LOD_mm3`. Treat them as "< LOD", not as zero.

### Validation (tests/test_extraction.py)
Scars of known size were painted into the real practice screenshot, keeping its
shading and overlays, and measured back:

| method | cases | area error |
|---|---|---|
| single scan | 0.4–2.5 µm deep, 80–400 µm wide, 10–120° | within ±10 % (typ. 2–5 %) |
| baseline difference | incl. 0.25–0.3 µm deep, 5×9 px misalignment | within ±1 % |
| unworn practice scan | 5 directions | no false scar |

`reports/validation/synthetic_scar_60deg.png` shows one of these checks.

## gng we need to discuss this with TA

1. **Stroke definition.** The DOE text uses V ≈ 4·stroke·f (i.e. stroke = half-travel)
   but eq. 3 uses S = 2·stroke·N (stroke = full travel). The two differ by a factor 2 in
   speed and distance. Set `test.stroke_definition` in the config after confirmation of
   what the machine's stroke setting means.
2. **Take a baseline WLI scan at t = 0 of the same spot.** It is the cheapest accuracy gain available.
3. **Raw height export.** If the WLI software can export the height matrix
   (csv/txt/asc), save it as `<name>_height.csv` in the run folder. It is used in place of
   the screenshot: no colour quantisation, no clipping.
4. **Screenshot hygiene.** Keep the whole scar inside the field of view with unworn surface
   on both sides, keep the colour bar visible, and don't let the scar bottom saturate the
   colour scale (the `colourbar_clipping_low` flag).
5. **Sliding direction in the camera view** is fixed by the rig. After the first
   real scar, put its angle (see `wear_*.png`) in `wear.track_angle_deg` in the config.
6. `XYZ.Z Depth` in the CSV is logged as a diagnostic (`Zdepth_drift_um`) but **not**
   used as wear: in the practice run it rose then fell, which real wear cannot do.
7. Wear of the ball is not measured. `k` refers to the flat (maraging steel) specimen.

## The practice run (TRIAL_2026-09-22)
COF_ss = 0.107 ± 0.004 after 1.5 s of running-in (3.2 s of sliding at 2.00 N).
No wear scar was detected. The screenshot shows the unworn machining lay (Sa ≈ 0.34 µm).
That fits ~3 s of sliding, or a baseline scan. With no frequency logged, *k* can't be
computed for this run.
