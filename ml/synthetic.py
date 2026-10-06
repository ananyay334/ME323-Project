"""Synthetic stand-ins for ``data/processed/*.csv``, in the exact pipeline schema.

    python -m ml.synthetic [--out data/synthetic] [--seed 323]

Writes ``runs_targets.csv`` and ``cof_timeseries.csv`` (plus ``wear_checkpoints.csv``
and the planted ground truth) into ``data/synthetic/``. It never writes into
``data/processed/``.

How it is built
---------------
* Design: the 4 x 5 x 5 grid R001-R100 (same ordering as ``data/run_sheet.csv``),
  10 replicates R101-R110 (fresh specimen, ``Replicate_of`` = original), 6 off-grid
  unseen points U01-U06, one ``status == "error"`` run and a TRIAL row (both must be
  dropped by the loader).
* COF: a 20 Hz signal ``COF_ss + A exp(-t/tau) + AR(1) + white noise`` behind a 5 s
  approach step is passed through :func:`tribo_extract.cof.extract_cof`, so
  ``COF_ss_mean``, ``t_runin_s``, ``Phase`` and the 1 Hz series follow the pipeline's
  own definitions exactly.
* Wear: Archard-like, ``log10 k = f(carbon, hardness, load, freq)`` with a convex load
  term, a carbon x load interaction and ~15 % log-normal noise; ``V = k F S``. Runs whose
  volume falls below a per-run detection limit are censored (``Wear_detected = False``).
* Trajectory: 30 runs get checkpoint volumes at 0/60/150/300/450/600 s from the
  two-stage model (DOE eq. 4) and are fitted with
  :func:`tribo_extract.physics.fit_wear_trajectory`, as the pipeline does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from tribo_extract import physics
from tribo_extract.config import load_config as load_extraction_config
from tribo_extract.cof import extract_cof
from tribo_extract.pipeline import FEATURE_COLUMNS, LEAD_COLUMNS, TRAIL_COLUMNS
from tribo_extract.tribometer_csv import TriboRun

from .config import PROJECT_ROOT, resolve_path

DEFAULT_OUT = "data/synthetic"
CARBON = [0.0, 0.5, 1.0, 2.0]
LOADS = [2.0, 4.0, 6.0, 8.0, 10.0]
FREQS = [2.0, 4.0, 6.0, 8.0, 10.0]
UNSEEN = [(0.5, 3.0, 5.0), (1.0, 7.0, 3.0), (2.0, 5.0, 9.0),
          (0.0, 9.0, 7.0), (1.0, 5.0, 5.0), (2.0, 3.0, 7.0)]
CHECKPOINTS_S = [0, 60, 150, 300, 450, 600]
DURATION_S = 600.0
SAMPLE_HZ = 20.0

# Planted effects (documented for the report and asserted by ml/tests).
PLANTED = {
    "log10_k": "-4.80 - 1.5 log10(HV/600) - 0.20 C + 0.05 C^2 + 0.22 xL + 0.10 xL^2 "
               "- 0.08 C xL + 0.04 xF  (+ N(0, 0.065) ~ 15 % log-normal noise)",
    "COF_ss": "0.62 - 0.05 C + 0.012 C^2 + 0.035 xF - 0.015 xF^2 - 0.012 xL  (+ N(0, 0.012))",
    "running_in": "COF(t) = COF_ss + A exp(-t/tau), A ~ 0.15 LN(0, 0.2), "
                  "tau = 45 (6/L)^0.5 (6/F)^0.3 LN(0, 0.15) s; AR(1) noise (8 s, sd 0.012)",
    "hardness": "HV = 520 + 95 C - 18 C^2 + N(0, 6) per specimen",
    "scaling": "xL = (Load_N - 6)/4, xF = (Freq_Hz - 6)/4, C = Carbon_pct",
    "signs": {"load_on_log10_k": "+ (convex)", "carbon_on_log10_k": "-",
              "freq_on_COF_ss": "+", "carbon_on_COF_ss": "-"},
}
NOISE_LOG10_K = 0.065            # log10(1.16): ~15 % multiplicative scatter
NOISE_COF_SS = 0.012


def hardness_mean(carbon):
    """Mean specimen hardness (HV) for a carbon level (planted)."""
    c = np.asarray(carbon, float)
    return 520.0 + 95.0 * c - 18.0 * c ** 2


def true_log10_k(carbon, hardness, load, freq):
    """Planted noise-free log10 specific wear rate (mm^3/Nm)."""
    c, hv = np.asarray(carbon, float), np.asarray(hardness, float)
    xl, xf = (np.asarray(load, float) - 6) / 4, (np.asarray(freq, float) - 6) / 4
    return (-4.80 - 1.5 * np.log10(hv / 600.0) - 0.20 * c + 0.05 * c ** 2
            + 0.22 * xl + 0.10 * xl ** 2 - 0.08 * c * xl + 0.04 * xf)


def true_cof_ss(carbon, load, freq):
    """Planted noise-free steady-state COF."""
    c = np.asarray(carbon, float)
    xl, xf = (np.asarray(load, float) - 6) / 4, (np.asarray(freq, float) - 6) / 4
    return 0.62 - 0.05 * c + 0.012 * c ** 2 + 0.035 * xf - 0.015 * xf ** 2 - 0.012 * xl


# ---------------------------------------------------------------------------
def _design(rng: np.random.Generator) -> list[dict]:
    """Grid + replicates + unseen points, with specimen IDs."""
    runs, i = [], 1
    for c in CARBON:
        for li, load in enumerate(LOADS):
            for f in FREQS:          # one specimen per (carbon, load) block of 5 tracks
                runs.append({"Experiment_ID": f"R{i:03d}", "Carbon_pct": c, "Load_N": load,
                             "Freq_Hz": f, "Specimen_ID": f"S-C{c:g}-{li + 1}", "kind": "grid"})
                i += 1
    originals = rng.choice(100, size=10, replace=False)
    for j, k in enumerate(sorted(originals)):
        o = runs[k]
        runs.append({**o, "Experiment_ID": f"R{101 + j:03d}", "Replicate_of": o["Experiment_ID"],
                     "Specimen_ID": f"S-rep-{j + 1}", "kind": "replicate"})
    for j, (c, load, f) in enumerate(UNSEEN):
        runs.append({"Experiment_ID": f"U{j + 1:02d}", "Carbon_pct": c, "Load_N": load,
                     "Freq_Hz": f, "Specimen_ID": f"S-unseen-{j + 1}", "kind": "unseen"})
    return runs


def _simulate_cof_signal(rng, cof_ss, amp, tau, load):
    """20 Hz tribometer channels: a 5 s approach step, then 600 s of sliding."""
    n_app, n = int(5 * SAMPLE_HZ), int(DURATION_S * SAMPLE_HZ)
    t = np.arange(n) / SAMPLE_HZ
    phi = np.exp(-1.0 / (SAMPLE_HZ * 8.0))                      # 8 s correlation time
    eta = rng.normal(0, 0.012 * np.sqrt(1 - phi ** 2), n)
    eta[0] = rng.normal(0, 0.012)
    ar = lfilter([1.0], [1.0, -phi], eta)
    mu = cof_ss + amp * np.exp(-t / tau) + ar + rng.normal(0, 0.02, n)
    app = 0.01 + rng.normal(0, 0.002, n_app)
    fz = np.r_[np.linspace(0.2, 1.0, n_app) * load, load * (1 + rng.normal(0, 0.01, n))]
    z = 420.0 + np.cumsum(rng.normal(0, 0.02, n_app + n))       # servo Z: diagnostic only
    df = pd.DataFrame({
        "t_step_s": np.r_[np.arange(n_app) / SAMPLE_HZ, t],
        "recipe_step": np.r_[np.ones(n_app), 2 * np.ones(n)],
        "cof": np.r_[app, mu], "fz_N": fz, "z_depth_um": z})
    df.insert(0, "t_s", np.arange(n_app + n) / SAMPLE_HZ)
    return TriboRun(path=Path("synthetic.csv"), data=df, sample_rate_hz=SAMPLE_HZ)


def _trajectory(rng, v_final, load, lod_v):
    """Checkpoint volumes from the two-stage model (eq. 4) -> pipeline-style fit."""
    frac = float(np.clip(0.25 * (load / 6) ** 0.3 * rng.lognormal(0, 0.2), 0.05, 0.6))
    tau = float(40.0 * (6 / load) ** 0.5 * rng.lognormal(0, 0.2))
    dv = frac * v_final
    vdot = (v_final - dv * (1 - np.exp(-DURATION_S / tau))) / DURATION_S
    t = np.array(CHECKPOINTS_S, float)
    # 2 % scatter: the baseline-difference accuracy validated for the extraction (README)
    v = physics.two_stage_wear(t, vdot, dv, tau) * (1 + rng.normal(0, 0.02, len(t)))
    v[0] = 0.0                                   # baseline scan: no scar (pipeline convention)
    v[1:][v[1:] < lod_v] = np.nan                # undetected checkpoints carry no volume
    fit = physics.fit_wear_trajectory(t, v)
    truth = {"Vdot_steady_true": vdot, "dV_run_true": dv, "tau_trans_true": tau}
    return t, v, fit, truth


def generate(out_dir: str | Path = DEFAULT_OUT, seed: int = 323, n_trajectory: int = 30,
             verbose: bool = True) -> dict[str, Path]:
    """Generate the synthetic tables and return their paths."""
    out = resolve_path(out_dir)
    if out.resolve() == (PROJECT_ROOT / "data" / "processed").resolve():
        raise ValueError("synthetic data must never be written into data/processed/")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    xcfg = load_extraction_config()
    tc = xcfg["test"]
    stroke, sdef = float(tc["stroke_mm"]), tc["stroke_definition"]

    design = _design(rng)
    specimen_hv: dict[str, float] = {}
    for d in design:
        if d["Specimen_ID"] not in specimen_hv:
            specimen_hv[d["Specimen_ID"]] = float(hardness_mean(d["Carbon_pct"]) + rng.normal(0, 6))
    grid_ids = [d["Experiment_ID"] for d in design if d["kind"] == "grid"]
    traj_pool = rng.permutation(grid_ids)
    hv_missing = set(rng.choice(grid_ids, size=2, replace=False))      # unmeasured specimens
    no_scan = str(rng.choice([g for g in grid_ids if g not in hv_missing]))

    rows, series, checkpoints, truth = [], [], [], []
    n_traj = 0
    for d in design:
        eid, c, load, f = d["Experiment_ID"], d["Carbon_pct"], d["Load_N"], d["Freq_Hz"]
        hv = specimen_hv[d["Specimen_ID"]]
        row = {"Experiment_ID": eid, "Carbon_pct": c,
               "Hardness_HV": np.nan if eid in hv_missing else round(hv, 1),
               "Load_N": load, "Freq_Hz": f, "Stroke_mm": stroke, "Duration_s": DURATION_S,
               "Ball_diam_mm": tc["ball_diameter_mm"], "Specimen_ID": d["Specimen_ID"],
               "Replicate_of": d.get("Replicate_of", np.nan),
               "Ambient_T_C": round(float(rng.normal(22.0, 1.0)), 1),
               "RH_pct": round(float(rng.normal(45.0, 8.0)), 1),
               "Notes": f"synthetic ({d['kind']})"}
        issues = []

        # ---- COF via the extraction's own code -------------------------------
        cof_true = float(true_cof_ss(c, load, f))
        cof_ss = cof_true + rng.normal(0, NOISE_COF_SS)
        amp = 0.15 * rng.lognormal(0, 0.2)
        tau = 45.0 * (6 / load) ** 0.5 * (6 / f) ** 0.3 * rng.lognormal(0, 0.15)
        run = _simulate_cof_signal(rng, cof_ss, amp, tau, load)
        cof_res, ts, _ = extract_cof(run, xcfg, load_N=load)
        row.update(cof_res)
        T = cof_res["sliding_time_s"]
        row["Speed_mps"] = physics.mean_speed_mps(f, stroke, sdef)
        row["N_cycles"] = f * T
        S = physics.sliding_distance_m(f, stroke, T, sdef)
        row["Sliding_distance_m"] = S
        hz = physics.hertz_contact(load, tc["ball_diameter_mm"], tc["E_ball_GPa"], tc["nu_ball"],
                                   tc["E_flat_GPa"], tc["nu_flat"])
        row["Hertz_width_um"], row["Hertz_p_mean_MPa"] = hz["hertz_width_um"], hz["p_mean_MPa"]

        # ---- wear ---------------------------------------------------------------
        lk_true = float(true_log10_k(c, hv, load, f))
        k = 10 ** (lk_true + rng.normal(0, NOISE_LOG10_K))
        v = k * load * S
        lod_v = 5.5e-4 * rng.lognormal(0, 0.25)
        truth.append({"Experiment_ID": eid, "log10_k_true": lk_true, "COF_ss_true": cof_true,
                      "tau_runin_true": tau, "Hardness_true": hv})
        if eid == no_scan:
            issues.append("no_wli_scan")
        else:
            detected = bool(v > lod_v)
            row.update({"Wear_method": "single_scan", "Wear_detected": detected,
                        "Wear_Volume_LOD_mm3": lod_v, "Final_Sa_um": 0.3 * rng.lognormal(0, 0.1),
                        "Specific_Wear_Rate_LOD_mm3_per_Nm":
                            physics.specific_wear_rate(lod_v, load, S),
                        "wear_flags": "" if detected else "no_wear_track_detected"})
            if detected and d["kind"] == "grid" and eid in traj_pool[:60] and n_traj < n_trajectory:
                t_cp, v_cp, fit, ttruth = _trajectory(rng, v, load, lod_v)
                if "Vdot_steady_mm3_per_s" in fit:
                    n_traj += 1
                    v = float(v_cp[-1]) if np.isfinite(v_cp[-1]) else v
                    sa0 = 0.3 * rng.lognormal(0, 0.1)
                    row.update(fit)
                    row.update({"Wear_method": "baseline_difference", "Initial_Sa_um": sa0,
                                "Initial_Ra_um": sa0,
                                "Specific_Wear_Rate_slope_mm3_per_Nm":
                                    fit["Vdot_linear_mm3_per_s"] / (load * row["Speed_mps"])})
                    truth[-1].update(ttruth)
                    checkpoints += [{"Experiment_ID": eid, "t_s": tt, "Wear_Volume_mm3": vv}
                                    for tt, vv in zip(t_cp, v_cp)]
            row["Wear_Volume_mm3"] = v if detected else np.nan
            row["Specific_Wear_Rate_mm3_per_Nm"] = (physics.specific_wear_rate(v, load, S)
                                                    if detected else np.nan)
        row["issues"] = ";".join(issues)
        row["status"] = "ok" if np.isfinite(row.get("Specific_Wear_Rate_mm3_per_Nm", np.nan)) \
            else "partial"
        rows.append(row)

        ts = ts.copy()
        ts.insert(0, "Experiment_ID", eid)
        for col in FEATURE_COLUMNS:
            if col in row:
                ts[col] = row[col]
        series.append(ts)

    # rows the loader must drop: a failed run and the practice run
    rows.append({"Experiment_ID": "R111", "status": "error",
                 "issues": "ValueError('synthetic: expected 1 csv, found 0')"})
    trial = {"Experiment_ID": "TRIAL_2026-09-22", "status": "partial", "Load_N": 2.0,
             "Stroke_mm": stroke, "sliding_time_s": 3.17, "COF_ss_mean": 0.107,
             "Wear_detected": False, "issues": "missing_Freq_Hz"}
    rows.append(trial)
    series.append(pd.DataFrame({"Experiment_ID": trial["Experiment_ID"], "Time_s": [1.0, 2.0, 3.0],
                                "COF": [0.16, 0.13, 0.11], "COF_std": 0.01, "Fz_N": 2.0,
                                "Zdepth_um": 420.0, "Phase": ["running_in", "steady", "steady"],
                                "Load_N": 2.0, "Stroke_mm": stroke}))

    runs = pd.DataFrame(rows).sort_values("Experiment_ID")
    for col in LEAD_COLUMNS + ["Replicate_of", "Wear_detected"]:
        if col not in runs:
            runs[col] = np.nan
    lead = [c for c in LEAD_COLUMNS if c in runs.columns]
    trail = [c for c in TRAIL_COLUMNS if c in runs.columns]
    runs = runs[lead + [c for c in runs.columns if c not in lead + trail] + trail]
    cts = pd.concat(series, ignore_index=True).sort_values(["Experiment_ID", "Time_s"])

    paths = {"runs": out / "runs_targets.csv", "timeseries": out / "cof_timeseries.csv",
             "checkpoints": out / "wear_checkpoints.csv", "truth": out / "synthetic_truth.csv",
             "planted": out / "planted_effects.json"}
    runs.to_csv(paths["runs"], index=False)
    cts.to_csv(paths["timeseries"], index=False)
    pd.DataFrame(checkpoints).to_csv(paths["checkpoints"], index=False)
    pd.DataFrame(truth).to_csv(paths["truth"], index=False)
    with open(paths["planted"], "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "planted": PLANTED, "noise_log10_k": NOISE_LOG10_K,
                   "noise_cof_ss": NOISE_COF_SS, "unseen": [f"U{j + 1:02d}" for j in range(len(UNSEEN))],
                   "hardness_missing": sorted(hv_missing), "no_wli_scan": no_scan,
                   "n_trajectory": n_traj}, fh, indent=2)
    if verbose:
        n_cens = int((runs["Wear_detected"].astype("string") == "False").sum()) - 1   # minus TRIAL
        print(f"wrote {paths['runs']} ({len(runs)} rows: 100 grid + 10 replicates + "
              f"{len(UNSEEN)} unseen + 1 error + 1 TRIAL; {n_cens} below LOD; "
              f"{n_traj} with trajectories)")
        print(f"wrote {paths['timeseries']} ({len(cts)} rows)")
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: data/synthetic)")
    ap.add_argument("--seed", type=int, default=323)
    ap.add_argument("--n-trajectory", type=int, default=30)
    a = ap.parse_args(argv)
    generate(a.out, a.seed, a.n_trajectory)


if __name__ == "__main__":
    main()
