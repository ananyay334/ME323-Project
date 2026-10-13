"""Synthetic stand-ins for the ML inputs, in the exact pipeline schema (40-run DOE).

    python -m ml.synthetic [--out data/synthetic] [--seed 323]

Writes ``runs_targets.csv``, ``cof_timeseries.csv``, ``wear_trajectory.csv`` and
``run_sheet.csv`` (plus the planted ground truth) into ``data/synthetic/``; never into
``data/processed/``.

* Design: the fixed 40-run matrix (S1-S4 = 0/0.5/1/2 % C, 10 tracks per specimen, anchors
  (2,2) (10,10) (6,6), one centre replicate, 6 interior points), 2400 s per run, stroke
  amplitude 6.25 mm, plus the TRIAL row (dropped by the loader).
* COF: a 10 Hz signal COF_ss + A·exp(-t/τ) + AR(1) + white noise behind a 5 s approach
  step, passed through :func:`tribo_extract.cof.extract_cof` (1 Hz series, steady state);
  COF_ss depends on frequency and carbon.
* Wear: Archard-like k(carbon, hardness, load) with a carbon x load interaction and ~15 %
  log-normal noise; V_w(t) follows eq. 1 with run-level noise in ΔV_run and τ_trans;
  in-situ scans at 0/300/600/900/1200/1800/2400 s carry 2 % scatter, and scans below the
  run's detection limit are censored. The eq.-1 parameters are fitted with
  :func:`tribo_extract.physics.fit_wear_trajectory`, as the pipeline does. About 3 runs end
  below the LOD. Replicate pairs share specimen and conditions and differ only by noise.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from tribo_extract import physics
from tribo_extract.config import load_config as load_extraction_config
from tribo_extract.cof import extract_cof
from tribo_extract.pipeline import FEATURE_COLUMNS, LEAD_COLUMNS, RUN_SHEET_COLUMNS, TRAIL_COLUMNS
from tribo_extract.tribometer_csv import TriboRun

from .config import PROJECT_ROOT, resolve_path

DEFAULT_OUT = "data/synthetic"
DURATION_S = 2400.0
SAMPLE_HZ = 10.0
STROKE_MM = 6.25
CHECKPOINTS_S = [0, 300, 600, 900, 1200, 1800, 2400]
DESIGN = """R01,S1,0.0,2,2,anchor_min,
R02,S1,0.0,10,10,anchor_max,
R03,S1,0.0,6,6,anchor_center,
R04,S1,0.0,6,6,center_replicate,R03
R05,S1,0.0,2,6,interior,
R06,S1,0.0,4,3,interior,
R07,S1,0.0,4,8,interior,
R08,S1,0.0,8,4,interior,
R09,S1,0.0,8,9,interior,
R10,S1,0.0,10,5,interior,
R11,S2,0.5,2,2,anchor_min,
R12,S2,0.5,10,10,anchor_max,
R13,S2,0.5,6,6,anchor_center,
R14,S2,0.5,6,6,center_replicate,R13
R15,S2,0.5,3,4,interior,
R16,S2,0.5,3,9,interior,
R17,S2,0.5,5,7,interior,
R18,S2,0.5,7,3,interior,
R19,S2,0.5,9,4,interior,
R20,S2,0.5,9,8,interior,
R21,S3,1.0,2,2,anchor_min,
R22,S3,1.0,10,10,anchor_max,
R23,S3,1.0,6,6,anchor_center,
R24,S3,1.0,6,6,center_replicate,R23
R25,S3,1.0,3,7,interior,
R26,S3,1.0,5,2,interior,
R27,S3,1.0,5,10,interior,
R28,S3,1.0,7,8,interior,
R29,S3,1.0,8,2,interior,
R30,S3,1.0,9,6,interior,
R31,S4,2.0,2,2,anchor_min,
R32,S4,2.0,10,10,anchor_max,
R33,S4,2.0,6,6,anchor_center,
R34,S4,2.0,6,6,center_replicate,R33
R35,S4,2.0,2,8,interior,
R36,S4,2.0,4,5,interior,
R37,S4,2.0,6,2,interior,
R38,S4,2.0,7,5,interior,
R39,S4,2.0,8,7,interior,
R40,S4,2.0,10,3,interior,"""

PLANTED = {
    "log10_k": "-4.80 - 1.5 log10(HV/600) - 0.20 C + 0.05 C^2 + 0.22 xL + 0.10 xL^2 "
               "- 0.08 C xL + 0.04 xF  (+ N(0, 0.065) ~ 15 % log-normal noise)",
    "COF_ss": "0.62 - 0.05 C + 0.012 C^2 + 0.035 xF - 0.015 xF^2 - 0.012 xL  (+ N(0, 0.012))",
    "trajectory": "V(t) = Vdot t + dV (1 - exp(-t/tau)); dV = 0.25 (L/6)^0.3 LN(0, 0.2) V_end, "
                  "tau = 250 (6/L)^0.5 LN(0, 0.2) s, Vdot from V(2400) = V_end; scans +-2 %",
    "running_in": "COF(t) = COF_ss + A exp(-t/tau), A ~ 0.15 LN(0, 0.2), "
                  "tau = 60 (6/L)^0.5 (6/F)^0.3 LN(0, 0.15) s; AR(1) noise (8 s, sd 0.012)",
    "hardness": "HV = 520 + 95 C - 18 C^2 + N(0, 6) per specimen",
    "scaling": "xL = (Load_N - 6)/4, xF = (Freq_Hz - 6)/4, C = Carbon_pct",
}
NOISE_LOG10_K = 0.065
NOISE_COF_SS = 0.012
LOD_V_MM3 = 3.0e-3              # median per-run detection limit (log-normal scatter 0.25)


def design_table() -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(DESIGN)))
    df = pd.DataFrame(rows, columns=["Experiment_ID", "Specimen_ID", "Carbon_pct", "Load_N",
                                     "Freq_Hz", "Role", "Replicate_of"])
    for c in ("Carbon_pct", "Load_N", "Freq_Hz"):
        df[c] = df[c].astype(float)
    df["Replicate_of"] = df["Replicate_of"].replace("", np.nan)
    rng = np.random.default_rng(323)                 # same Run_order as data/run_sheet.csv
    df["Run_order"] = 0
    for spec in ["S1", "S2", "S3", "S4"]:
        m = df["Specimen_ID"] == spec
        df.loc[m, "Run_order"] = rng.permutation(int(m.sum())) + 1
    return df


def hardness_mean(carbon):
    c = np.asarray(carbon, float)
    return 520.0 + 95.0 * c - 18.0 * c ** 2


def true_log10_k(carbon, hardness, load, freq):
    """Planted noise-free log10 specific wear rate (mm^3/Nm)."""
    c, hv = np.asarray(carbon, float), np.asarray(hardness, float)
    xl, xf = (np.asarray(load, float) - 6) / 4, (np.asarray(freq, float) - 6) / 4
    return (-4.80 - 1.5 * np.log10(hv / 600.0) - 0.20 * c + 0.05 * c ** 2
            + 0.22 * xl + 0.10 * xl ** 2 - 0.08 * c * xl + 0.04 * xf)


def true_cof_ss(carbon, load, freq):
    c = np.asarray(carbon, float)
    xl, xf = (np.asarray(load, float) - 6) / 4, (np.asarray(freq, float) - 6) / 4
    return 0.62 - 0.05 * c + 0.012 * c ** 2 + 0.035 * xf - 0.015 * xf ** 2 - 0.012 * xl


def _simulate_cof_signal(rng, cof_ss, amp, tau, load):
    n_app, n = int(5 * SAMPLE_HZ), int(DURATION_S * SAMPLE_HZ)
    t = np.arange(n) / SAMPLE_HZ
    phi = np.exp(-1.0 / (SAMPLE_HZ * 8.0))
    eta = rng.normal(0, 0.012 * np.sqrt(1 - phi ** 2), n)
    eta[0] = rng.normal(0, 0.012)
    mu = cof_ss + amp * np.exp(-t / tau) + lfilter([1.0], [1.0, -phi], eta) + rng.normal(0, 0.02, n)
    df = pd.DataFrame({
        "t_step_s": np.r_[np.arange(n_app) / SAMPLE_HZ, t],
        "recipe_step": np.r_[np.ones(n_app), 2 * np.ones(n)],
        "cof": np.r_[0.01 + rng.normal(0, 0.002, n_app), mu],
        "fz_N": np.r_[np.linspace(0.2, 1.0, n_app) * load, load * (1 + rng.normal(0, 0.01, n))],
        "z_depth_um": 420.0 + np.cumsum(rng.normal(0, 0.02, n_app + n))})
    df.insert(0, "t_s", np.arange(n_app + n) / SAMPLE_HZ)
    return TriboRun(path=Path("synthetic.csv"), data=df, sample_rate_hz=SAMPLE_HZ)


def generate(out_dir: str | Path = DEFAULT_OUT, seed: int = 323, verbose: bool = True) -> dict:
    """Generate the synthetic tables and return their paths."""
    out = resolve_path(out_dir)
    if out.resolve() == (PROJECT_ROOT / "data" / "processed").resolve():
        raise ValueError("synthetic data must never be written into data/processed/")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    xcfg = load_extraction_config()
    tc = xcfg["test"]
    sdef = tc["stroke_definition"]
    design = design_table()
    hv_spec = {s: float(hardness_mean(c) + rng.normal(0, 6))
               for s, c in design.groupby("Specimen_ID")["Carbon_pct"].first().items()}
    t_cp = np.array(CHECKPOINTS_S, float)

    rows, series, scans, truth = [], [], [], []
    for d in design.to_dict("records"):
        eid, c, load, f = d["Experiment_ID"], d["Carbon_pct"], d["Load_N"], d["Freq_Hz"]
        hv = hv_spec[d["Specimen_ID"]]
        row = {"Experiment_ID": eid, "Carbon_pct": c, "Hardness_HV": round(hv, 1), "Load_N": load,
               "Freq_Hz": f, "Stroke_mm": STROKE_MM, "Duration_s": DURATION_S, "Ball_diam_mm": 6.0,
               "Specimen_ID": d["Specimen_ID"], "Replicate_of": d["Replicate_of"],
               "Role": d["Role"], "Run_order": d["Run_order"],
               "Ambient_T_C": round(float(rng.normal(22.0, 1.0)), 1),
               "RH_pct": round(float(rng.normal(45.0, 8.0)), 1), "Notes": "synthetic"}

        cof_true = float(true_cof_ss(c, load, f))
        tau_cof = 60.0 * (6 / load) ** 0.5 * (6 / f) ** 0.3 * rng.lognormal(0, 0.15)
        run = _simulate_cof_signal(rng, cof_true + rng.normal(0, NOISE_COF_SS),
                                   0.15 * rng.lognormal(0, 0.2), tau_cof, load)
        cof_res, ts, _ = extract_cof(run, xcfg, load_N=load)
        row.update(cof_res)
        T = cof_res["sliding_time_s"]
        row["Speed_mps"] = physics.mean_speed_mps(f, STROKE_MM, sdef)
        row["N_cycles"] = f * T
        S = physics.sliding_distance_m(f, STROKE_MM, T, sdef)
        row["Sliding_distance_m"] = S
        hz = physics.hertz_contact(load, 6.0, tc["E_ball_GPa"], tc["nu_ball"], tc["E_flat_GPa"],
                                   tc["nu_flat"])
        row["Hertz_width_um"], row["Hertz_p_mean_MPa"] = hz["hertz_width_um"], hz["p_mean_MPa"]

        # ---- eq.-1 wear trajectory, in-situ scans ---------------------------------
        lk_true = float(true_log10_k(c, hv, load, f))
        v_end = 10 ** (lk_true + rng.normal(0, NOISE_LOG10_K)) * load * S
        frac = float(np.clip(0.25 * (load / 6) ** 0.3 * rng.lognormal(0, 0.2), 0.05, 0.6))
        tau = float(250.0 * (6 / load) ** 0.5 * rng.lognormal(0, 0.2))
        dv = frac * v_end
        vdot = (v_end - dv * (1 - np.exp(-DURATION_S / tau))) / DURATION_S
        lod = LOD_V_MM3 * rng.lognormal(0, 0.25)
        v = physics.two_stage_wear(t_cp, vdot, dv, tau) * (1 + rng.normal(0, 0.02, len(t_cp)))
        det = np.r_[False, v[1:] > lod]                 # t = 0 baseline: no scar
        vol = np.where(det, v, np.nan)
        for tt, vv, dd in zip(t_cp, vol, det):
            scans.append({"Experiment_ID": eid, "Time_s": tt, "Wear_Volume_mm3": vv,
                          "Wear_Volume_LOD_mm3": lod, "Wear_detected": bool(dd),
                          "Wear_method": "single_scan" if tt == 0 else "baseline_difference"})
        fit = physics.fit_wear_trajectory(t_cp, np.where(t_cp == 0, 0.0, vol))
        row.update(fit)
        if "Vdot_linear_mm3_per_s" in fit:
            row["Specific_Wear_Rate_slope_mm3_per_Nm"] = fit["Vdot_linear_mm3_per_s"] / (
                load * row["Speed_mps"])
        detected = bool(det[-1])
        row.update({"Wear_method": "baseline_difference", "Wear_detected": detected,
                    "Wear_Volume_mm3": vol[-1] if detected else np.nan,
                    "Wear_Volume_LOD_mm3": lod,
                    "Specific_Wear_Rate_mm3_per_Nm":
                        physics.specific_wear_rate(vol[-1], load, S) if detected else np.nan,
                    "Specific_Wear_Rate_LOD_mm3_per_Nm": physics.specific_wear_rate(lod, load, S),
                    "Initial_Sa_um": 0.3 * rng.lognormal(0, 0.1),
                    "wear_flags": "" if detected else "no_wear_track_detected", "issues": ""})
        row["status"] = "ok" if detected else "partial"
        rows.append(row)
        truth.append({"Experiment_ID": eid, "log10_k_true": lk_true, "COF_ss_true": cof_true,
                      "Vdot_steady_true": vdot, "dV_run_true": dv, "tau_trans_true": tau})
        ts = ts.copy()
        ts.insert(0, "Experiment_ID", eid)
        for col in FEATURE_COLUMNS:
            if col in row:
                ts[col] = row[col]
        series.append(ts)

    rows.append({"Experiment_ID": "TRIAL_2026-09-22", "status": "partial", "Load_N": 2.0,
                 "Stroke_mm": STROKE_MM, "sliding_time_s": 3.17, "COF_ss_mean": 0.107,
                 "Wear_detected": False, "issues": "missing_Freq_Hz"})
    runs = pd.DataFrame(rows).sort_values("Experiment_ID")
    lead = [c for c in LEAD_COLUMNS if c in runs.columns]
    trail = [c for c in TRAIL_COLUMNS if c in runs.columns]
    runs = runs[lead + [c for c in runs.columns if c not in lead + trail] + trail]

    sheet = design.assign(Stroke_mm=STROKE_MM, Duration_s=int(DURATION_S), Ball_diam_mm=6)
    for c in RUN_SHEET_COLUMNS:
        if c not in sheet:
            sheet[c] = np.nan
    paths = {"runs": out / "runs_targets.csv", "timeseries": out / "cof_timeseries.csv",
             "wear_trajectory": out / "wear_trajectory.csv", "run_sheet": out / "run_sheet.csv",
             "truth": out / "synthetic_truth.csv", "planted": out / "planted_effects.json"}
    runs.to_csv(paths["runs"], index=False)
    pd.concat(series, ignore_index=True).to_csv(paths["timeseries"], index=False)
    pd.DataFrame(scans).sort_values(["Experiment_ID", "Time_s"]).to_csv(paths["wear_trajectory"],
                                                                        index=False)
    sheet[RUN_SHEET_COLUMNS].to_csv(paths["run_sheet"], index=False)
    pd.DataFrame(truth).to_csv(paths["truth"], index=False)
    n_cens = int((~runs["Wear_detected"].astype(bool)).sum()) - 1          # minus TRIAL
    with open(paths["planted"], "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "planted": PLANTED, "noise_log10_k": NOISE_LOG10_K,
                   "noise_cof_ss": NOISE_COF_SS, "n_below_lod": n_cens}, fh, indent=2)
    if verbose:
        print(f"wrote {paths['runs']} (40 runs + TRIAL; {n_cens} below LOD; "
              f"{int(runs['Vdot_steady_mm3_per_s'].notna().sum())} eq.-1 fits)")
        print(f"wrote {paths['timeseries']}, {paths['wear_trajectory']}, {paths['run_sheet']}")
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: data/synthetic)")
    ap.add_argument("--seed", type=int, default=323)
    a = ap.parse_args(argv)
    generate(a.out, a.seed)


if __name__ == "__main__":
    main()
