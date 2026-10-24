# Campaign results: 304 SS dry reciprocating sliding (runs of 13 and 16 Oct 2026)

Produced by:
```bash
python -m tribo_extract extract                              # testdata/ -> data/processed/
python -m ml.train && python -m ml.train --final             # COF + log10 k   -> ml/results/final/
python -m ml.train --config ml/configs/width.yaml --final    # COF + log10 width -> ml/results/final_width/
python -m ml.session_effect                                  # -> ml/results/session_effect.md
```

## 1. Runs used

There are 23 run folders: 18 training runs and 5 test runs. The 7 runs with no folder are skipped:
R00, R03, R15, R16, R18 and R23 (training) and R27 (test). The `TRAIL'NEW` practice folder is also
skipped.
- **Parameters.** Load, frequency and time were read from every run's own `.lrx` recipe. All 23
  agree with the DOE row of the same ID, so no run was renamed or re-assigned. The replicate
  condition 32 N / 7.5 Hz / 84 s (R04, R14, R20) was found from the file values.
- **Two test days** (CSV `TestDate`): R01–R10 on 13 Oct; R11–R22 and all test runs on 16 Oct.
- **Protocol as run.** One continuous sliding step, split at t/2 for the COF checkpoints, logged
  at 100 Hz. The tracks were scanned after the run only (4 locations), so wear is available at
  the end of the run only.

| run | set | day | L (N) | f (Hz) | t (s) | cycles | COF cp1 | COF cp2 | track width (µm) | groove k (mm³/N·m) |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | train | 10-13 | 10 | 6 | 120 | 717 | 0.143 | 0.145 | 396 ± 43 | 3.8e-05 |
| R02 | train | 10-13 | 15 | 4 | 120 | 478 | 0.163 | 0.165 | 307 ± 61 | 1.3e-05 |
| R04 (replicate) | train | 10-13 | 32 | 7.5 | 84 | 626 | 0.145 | 0.155 | 414 ± 24 | < LOD |
| R05 | train | 10-13 | 10 | 15 | 60 | 893 | 0.160 | 0.151 | 399 ± 119 | 1.3e-05 |
| R06 | train | 10-13 | 24.5 | 4.5 | 116 | 520 | 0.143 | 0.151 | 341 ± 13 | < LOD |
| R07 | train | 10-13 | 10.5 | 12.5 | 68 | 844 | 0.136 | 0.140 | 241 ± 15 | < LOD |
| R08 | train | 10-13 | 54 | 12 | 72 | 858 | 0.138 | 0.149 | 426 ± 132 | 3.2e-06 |
| R09 | train | 10-13 | 20 | 11 | 62 | 677 | 0.156 | 0.161 | 332 ± 47 | < LOD |
| R10 | train | 10-13 | 40.5 | 6.5 | 72 | 465 | 0.136 | 0.140 | 377 ± 24 | < LOD |
| R11 | train | 10-16 | 100 | 4 | 60 | 238 | 0.132 | 0.144 | 636 ± 54 | 1.5e-05 |
| R12 | train | 10-16 | 100 | 7.5 | 120 | 896 | 0.124 | 0.117 | 705 ± 33 | not measurable |
| R13 | train | 10-16 | 16 | 5.5 | 82 | 448 | 0.123 | 0.130 | 216 ± 56 | < LOD |
| R14 (replicate) | train | 10-16 | 32 | 7.5 | 84 | 627 | 0.131 | 0.147 | 356 ± 34 | 1.7e-05 |
| R17 | train | 10-16 | 15 | 6.5 | 100 | 647 | 0.128 | 0.134 | 278 ± 71 | < LOD |
| R19 | train | 10-16 | 89 | 7.5 | 68 | 506 | 0.124 | 0.120 | 619 ± 57 | not measurable |
| R20 (replicate) | train | 10-16 | 32 | 7.5 | 84 | 627 | 0.133 | 0.143 | 308 ± 70 | 8.0e-06 |
| R21 | train | 10-16 | 47.5 | 7 | 108 | 753 | 0.122 | 0.127 | 336 ± 49 | 4.8e-06 |
| R22 | train | 10-16 | 80 | 4 | 70 | 278 | 0.129 | 0.129 | 424 ± 65 | 3.8e-05 |
| R24 | test | 10-16 | 23 | 5 | 64 | 318 | 0.128 | 0.132 | 144 ± 39 | < LOD |
| R25 | test | 10-16 | 12 | 9 | 86 | 770 | 0.130 | 0.131 | 185 ± 45 | 1.1e-05 |
| R26 | test | 10-16 | 73 | 8 | 102 | 812 | 0.127 | 0.129 | 407 ± 35 | 3.2e-06 |
| R28 | test | 10-16 | 64 | 6 | 64 | 381 | 0.123 | 0.132 | 241 ± 130 | 7.4e-06 |
| R29 | test | 10-16 | 13.5 | 5 | 108 | 538 | 0.145 | 0.153 | 221 ± 35 | < LOD |

In the table:
- Width is the mean ± SD over the scan locations where a track was found.
- `< LOD` means the groove is below its detection limit (censored, never zero).
- `not measurable` means the 89–100 N tracks fill the 1.125 mm field of view, so there is no
  unworn reference on both sides.

## 2. Cross-validation on the 18 training runs

5-fold grouped CV repeated 5 times, with nested tuning; the replicates form one group. The
pooled R² uses all out-of-fold predictions together. "Spread" is the SD of the target over the
training runs, which is the error of always predicting the mean.

| target | rows | best model (pooled) | pooled R² | CV RMSE | spread | noise floor (replicates) |
|---|---|---|---|---|---|---|
| COF_steady | 36 (both checkpoints) | ANN | −0.04 to −0.12 | 0.013 | 0.013 | 0.007 |
| log10 k (groove) | 9 (7 censored, 2 not measurable) | ANN | −0.16 | 0.37 dex | 0.34 dex | 0.23 dex |
| log10 track width | 18 | XGB / ANN | **+0.34 / +0.33** | 0.107 dex | 0.133 dex | 0.064 dex |

## 3. Sealed test set (R24, R25, R26, R28, R29), scored once

The model with the lowest CV RMSE was selected and refitted on all training runs. The null
model predicts the training mean.

| target | model | test RMSE | null-model RMSE | bias (pred − meas.) | 95 % interval coverage |
|---|---|---|---|---|---|
| COF_steady (10 rows) | ANN | **0.0069** (0.0063 in the width run) | 0.0105 | +0.004 | 90 % |
| log10 k (3 runs; R24, R29 < LOD) | GPR | 0.42 dex | 0.37 dex | +0.35 dex | 67 % |
| log10 track width (5 runs) | XGB | 0.215 dex | 0.269 dex | **+0.18 dex** (widths over-predicted ×1.5) | – |

## 4. What the data say

1. **Track width grows with load and nothing else is detectable.** The fitted relation is
   w ∝ L^0.2–0.3: OLS 0.29, shrunk Ridge 0.19, 95 % CI excluding 0. The Hertz contact predicts
   L^1/3. Frequency and time have no effect within the noise: the GPR length scales are at their
   upper bound and the CIs include 0.
2. **COF is not explained by load, frequency or time.** COF is 0.12–0.165 everywhere, and the run-to-run
   spread (0.013) is only twice the replicate noise (0.007). It rises slightly during the run
   (cp1 → cp2, +0.005 on average, up in 20 of 23 runs).
3. **The test day matters more than any DOE factor for COF.** On the training runs only:
   - COF was 0.151 on 13 Oct and 0.132 on 16 Oct, a difference of −0.018, about 2.6× the replicate noise;
   - with leave-one-group-out CV, the test day alone gives R² = +0.34, while L, f, t give −0.23;
   - the 16 Oct plates are also about twice as wavy (unworn Sa 5.5–9 µm against 3.0–3.4 µm); their fine
     texture is similar.

   All test runs are from 16 Oct. That explains the positive COF bias on the test set. It also
   means the replicate "noise floor" (R04 on 13 Oct, R14/R20 on 16 Oct) contains the day effect.
   Details are in `ml/results/session_effect.md`.
4. **The specific wear rate k cannot be learned from these scans.**
   - The plates are rough (Sa 3–9 µm), and the track is a flattened band only about 1 µm below
     the surface, with pile-up larger than the groove in 8 of the 12 runs where both are measured: 304 deforms
     more than it loses material at these pass counts.
   - The groove volume is below its detection limit in 9 of 23 runs and not measurable in 2 more.
   - The two detected replicates (R14, R20) differ by a factor of 2; R04 is below the LOD.

   Use the track width, or ball-scar diameters if they are measured, as the wear response.

## 5. Limitations found during QC (not corrected, to keep the test set clean)

- **Width under-read on faint tracks on the 16 Oct plates.** For faint, low-load tracks on the
  rougher 16 Oct surfaces, the smooth-core finder sometimes locks onto only the smoothest strip
  of the band. For example, `reports/runs/R24/wear_a_cp2.png` shows a core of 184 µm where the
  smooth band is visibly about twice as wide; R13 c looks similar. This contributes to the
  test-set width bias.
  - A fix would set the core threshold from a robust track roughness instead of its minimum.
  - Any such change should be validated on new runs, not on R24–R29, which have now been scored.
- Scans of the 89–100 N tracks need a wider field of view (e.g. a stitched or 5× scan) to get
  a two-sided reference.
- The 100 Hz logging gives 7–25 samples per cycle; the windowed half-range COF handles this.
- TabPFN was not run (not installed); torch is not needed (numpy ANN backend).
