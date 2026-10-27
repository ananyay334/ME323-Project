# Slide data

Numbers copied from outputs/tables by `run_all.py` (stage report). Figures are in outputs/figures.

## Slide 1 – Question and pipeline

- Predict COF and wear of 304 SS (dry, ball-on-flat, A = 5 mm) from load L, frequency f and time t
- 23 runs with data (of 30 planned); missing: R00, R03, R15, R16, R18, R23, R27
- Models must include ANN, GPR and TabPFN; the test day is never an input
- **Figure:** fig01_pipeline

## Slide 2 – Test matrix

- L 10–100 N, f 4–15 Hz, t 60–120 s
- 18 training, 5 test runs; replicates R04/R14/R20 (32 N, 7.5 Hz, 84 s) = one CV group
- cycles 238–897; sliding distance 4.8–17.9 m
- **Figure:** fig02_test_matrix

## Slide 3 – COF extraction

- R04: 628 cycles counted (f·t = 630), COF_end 0.152
- R01 static step: instrument 'COF' 0.049 = sensor offset only
- instrument mean / offset-free COF = 0.87; falls with f (r = -0.70)
- COF still rising at the end in 19 of 23 runs
- **Figure:** fig03_raw_signal, fig04_cof_time

## Slide 4 – Which COF? (C1)

- COF_first10s: replicate SD 0.0056, physics R² P1 +0.36 / P2 +0.23
- COF_end: physics R² P1 +0.08 / P2 +0.05
- Use COF_first10s: it follows the adhesion law best among the definitions (P1 R² +0.36, P2 +0.23; end-of-run COF +0.08) with signal-to-noise 2.2; the instrument mean reads 13 % low and its ratio to ours falls with frequency (r = -0.70).
- **Figure:** fig08_c1_cof_definitions

## Slide 5 – Wear extraction

- band method: 16 detected, 6 below LOD (R04, R06, R07, R09, R10, R29), 1 not measurable (R12)
- R04 track width 424 µm; unworn Sa 3.1–10.4 µm
- wear replicate SD (R14/R20) 0.07 dex
- **Figure:** fig05_height_map, fig06_robust_levelling

## Slide 6 – Which wear measure? (C2)

- most-worn + robust: 16 runs, replicate SD 0.07 dex, Archard R² P1 +0.29
- old mean-of-locations, plain: replicate SD 0.34 dex, R² P1 +0.08
- V ~ L^0.96 [0.41, 1.59] · S^-0.07 [-1.32, 1.61] (Archard 1, 1)
- **Figure:** fig09_c2_wear_measures

## Slide 7 – COF model

- COF_first10s = μ₀ + c·L^(−1/3): μ₀ = 0.099 [0.086, 0.114], c = 0.098 [0.049, 0.142]
- R² P1 +0.36, P2 +0.23, P3 +0.70; permutation p = 0.0005
- noise floor 0.0056, S/N 2.2
- COF_end with the same law: R² P1 +0.08
- **Figure:** fig10_physics_fits

## Slide 8 – Wear model

- k = 2.96e-05 mm³/(N·m) ×/÷ 2.36 (95 % CI 1.86e-05–4.38e-05, n = 16)
- Archard log10 V: R² P1 +0.29, P2 +0.15, permutation p = 0.0090
- track width ~ L^0.29 [0.12, 0.47] (Hertz 1/3); R² P1 +0.20, P2 +0.39
- **Figure:** fig07_wear_per_run, fig10_physics_fits

## Slide 9 – Model comparison (C3)

- COF first 10 s: final μ₀ + c·L^(−1/3); selected ML ANN + physics (P1 +0.43, P2 +0.04)
- COF end: final μ₀ + c·L^(−1/3); selected ML none (P1 –, P2 –)
- log10 V: final Archard (slope 1); selected ML ANN (P1 +0.74, P2 +0.16)
- log10 width: final a + b·ln L; selected ML GPR + physics (P1 +0.26, P2 +0.55)
- log10 k: final Archard constant k; selected ML ANN + physics (P1 +0.40, P2 +0.39)
- TabPFN: not run (tabpfn installed but cannot load weights: TabPFNLicenseError: TabPFN requires a one-time license acceptance to download)
- **Figure:** fig11_c3_models

## Slide 10 – Stability and fragility

- unstable (win P1, lose P2): xgb on COF_first10s, gpr on COF_end, ann_phys on COF_end, gpr_phys on COF_end
- plain GPR on COF_end: P1 +0.37, P2 -0.38
- selected k model ANN + physics: P1 +0.40, without R11 -0.32 (constant -0.17)
- **Figure:** fig12_stability

## Slide 11 – Final models: parity, residuals, uncertainty

- COF first 10 s: R² P1 +0.36, coverage 96 %, tercile κ +0.21
- COF end: R² P1 +0.08, coverage 96 %, tercile κ -0.02
- log10 V: R² P1 +0.29, coverage 94 %, tercile κ +0.07
- log10 width: R² P1 +0.20, coverage 91 %, tercile κ +0.54
- log10 k: R² P1 -0.15, coverage 94 %, tercile κ +0.00
- ANN-ensemble coverage 35-100 % (spread of seeds only); GPR 75-96 %
- **Figure:** fig13_parity, fig14_residuals, fig15_confusion, fig16_uncertainty

## Slide 12 – What the models say

- COF_first10s exponents: L -0.09 [-0.12, -0.04], f -0.05 [-0.15, +0.05], t -0.05 [-0.18, +0.08]
- width exponents: L +0.35 [+0.15, +0.57], f +0.29 [-0.13, +0.81], t +0.49 [-0.28, +1.28]
- load dominates (shortest GPR length scale, largest ANN importance); f and t not detectable
- **Figure:** fig17_interpretation, fig18_surfaces

## Slide 13 – Old pipeline vs final (C4) and hold-out (P3)

- Replace the old pipeline. COF: the old ANN on end-of-run COF scores R² +0.40 in P1 but +0.00 on the training runs (P2); the final model on COF_first10s scores +0.36 / +0.23 with two parameters. Wear: R² -0.48 / -0.47 (ANN on mean-of-locations k, plain levelling, replicate SD 0.34 dex) -> +0.29 / +0.15 (final model on the most-worn-location volume, replicate SD 0.07 dex).
- P3 COF_first10s physics R² +0.70; log10 V Archard +0.38
- **Figure:** fig19_c4_old_vs_final, fig22_p3_holdout

## Slide 14 – Literature data (C5)

- paper COF ~ L^+0.52 [+0.08, +0.93] v^-0.69 [-0.94, -0.44]
- ours COF ~ L^-0.09 [-0.12, -0.04] v^-0.05 [-0.15, +0.05]
- Do not use the literature data: paper COF ~ L^+0.52 v^-0.69 vs ours L^-0.09 v^-0.05; log10_COF: best with paper data GPR + study flag R² +0.24 vs ours alone +0.28; log10_k: best with paper data pooled + study offset: power law R² -0.31 vs ours alone +0.41. The published COF (0.002-0.007) and volumes (up to 1620 mm³ in 200 s) are implausible as printed.
- **Figure:** fig20_c5_literature

## Slide 15 – TA script (C6)

- bug 1: 25 of 92 scans transposed, so the 'profile' runs along the track. R24.bcrf: transposed, reported depth 22.5 um (a pit), groove area 4352 um^2; with the correct axis the same code gives depth 7.4 um, groove area 0 um^2
- bug 2: 10 scans (correct axis) have groove area 0 although the depth is >= 3 um, e.g. R05b_13.bcrf (3.4 um), R11.bcrf (4.2 um), R13.bcrf (4.1 um), R14_a.bcrf (5.9 um)
- bug 3: median TA pile-up / band pile-up = 1.5 over 65 valid scans (texture peaks counted as pile-up)
- Do not use the TA script as written: 25 of 92 scans are read along the track, and the R04/R14/R20 replicates spread 30x in k (SD 0.84 dex); fixed copy 0.13 dex, ours 0.07 dex (R14/R20 only). Archard R² (P1): original -0.03, fixed -0.19, ours +0.29.
- **Figure:** fig21_c6_ta_script

