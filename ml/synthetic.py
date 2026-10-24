"""Synthetic stand-in for ``runs_checkpoints.csv`` (exact pipeline schema), 304 SS DOE.

    python -m ml.synthetic [--out data/synthetic] [--seed 323]

Writes ``runs_checkpoints.csv``, ``cof_cycles.csv`` and ``planted_effects.json`` into
``data/synthetic/`` (never into ``data/processed/``).

* Design: the fixed 30-run matrix R00-R29 (24 training runs incl. the centre replicates
  R04/R14/R20, 6 test runs R24-R29), two checkpoints per run at t_run/2 and t_run.
* log10 k = a + 0.3 ln L + 0.2 ln f - 0.25 ln t + N(0, 0.08)
* COF    = 0.12 + 0.02 ln f - 0.01 ln L + N(0, 0.005)
* S = 0.02 f t m, groove volume V = k L S, pile-up 10 % of V, LOD 3e-4 mm³; the two
  lowest-wear non-replicate training runs are below the LOD at both checkpoints.
* R04/R14/R20 share their conditions and differ only by noise.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tribo_extract.pipeline import CHECKPOINT_COLUMNS, CYCLE_COLUMNS

from .config import resolve_path

DEFAULT_OUT = "data/synthetic"
A_LOG10K = -4.3
EFFECTS = {"log10_k": {"intercept": A_LOG10K, "ln_L": 0.3, "ln_f": 0.2, "ln_t": -0.25, "noise_sd": 0.08},
           "COF_steady": {"intercept": 0.12, "ln_L": -0.01, "ln_f": 0.02, "ln_t": 0.0, "noise_sd": 0.005}}
LOD_MM3 = 3e-4
N_CENSORED_RUNS = 2
DESIGN = """Experiment_ID,Session,Face,X_mm,Y_mm,Load_N,Freq_Hz,Run_time_s,Role,Replicate_of
R00,0,,,,10,10,120,completed,
R01,0,,,,10,6,120,completed,
R02,1,F1,21,3,15,4,120,corner,
R03,1,F2,21,21,100,15,60,corner,
R04,1,F1,21,24,32,7.5,84,center_replicate,
R05,1,F1,34,18,10,15,60,corner,
R06,1,F2,8,15,24.5,4.5,116,interior,
R07,2,F2,34,12,10.5,12.5,68,interior,
R08,2,F1,21,27,54,12,72,interior,
R09,2,F1,34,27,20,11,62,interior,
R10,2,F2,34,3,40.5,6.5,72,interior,
R11,2,F1,8,39,100,4,60,corner,
R12,3,F2,21,39,100,7.5,120,corner,
R13,3,F2,8,6,16,5.5,82,interior,
R14,3,F1,8,30,32,7.5,84,center_replicate,R04
R15,3,F1,21,33,33,15,60,interior,
R16,4,F2,34,39,75,10,120,corner,
R17,4,F1,34,39,15,6.5,100,interior,
R18,4,F2,8,24,35,10,96,interior,
R19,4,F1,34,24,89,7.5,68,interior,
R20,5,F2,21,30,32,7.5,84,center_replicate,R04
R21,5,F1,8,6,47.5,7,108,interior,
R22,5,F2,8,18,80,4,70,interior,
R23,5,F2,21,9,59,9.5,90,interior,
R24,6,F2,34,6,23,5,64,test,
R25,6,F2,21,3,12,9,86,test,
R26,6,F1,8,24,73,8,102,test,
R27,7,F1,34,33,28.5,14,72,test,
R28,7,F2,21,6,64,6,64,test,
R29,7,F1,8,27,13.5,5,108,test,
"""


def design() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(DESIGN), dtype={"Experiment_ID": str, "Face": str, "Replicate_of": str})


def generate(out_dir: str | Path = DEFAULT_OUT, seed: int = 323, verbose: bool = True) -> Path:
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d = design()
    rows = []
    for r in d.itertuples(index=False):
        for j in (1, 2):
            t = r.Run_time_s * j / 2
            x = {"ln_L": np.log(r.Load_N), "ln_f": np.log(r.Freq_Hz), "ln_t": np.log(t)}
            y = {k: e["intercept"] + sum(e[c] * x[c] for c in x) + rng.normal(0, e["noise_sd"])
                 for k, e in EFFECTS.items()}
            N = r.Freq_Hz * t
            S = 4 * 5.0 * N / 1000.0
            k = 10 ** y["log10_k"]
            V = k * r.Load_N * S
            pile = 0.10 * V * (1 + 0.1 * rng.normal())
            rows.append({**r._asdict(), "Checkpoint": j, "Time_s": t, "Cycles_cum": N, "Distance_m": S,
                         "Freq_achieved_Hz": r.Freq_Hz, "Stroke_amp_mm": 5.0,
                         "Fz_mean_N": r.Load_N * (1 + 0.005 * rng.normal()),
                         "COF_steady": y["COF_steady"], "COF_std": abs(0.004 + 0.001 * rng.normal()),
                         "Groove_volume_mm3": V, "Groove_volume_se_mm3": 0.05 * V,
                         "Pileup_volume_mm3": pile, "Net_volume_mm3": V - pile,
                         "k_mm3_per_Nm": k, "k_net_mm3_per_Nm": (V - pile) / (r.Load_N * S),
                         "Wear_detected": True, "Wear_Volume_LOD_mm3": LOD_MM3,
                         "V_width_mm3": V * (1 + 0.2 * rng.normal()), "V_ball_mm3": np.nan,
                         "n_locations": 4, "status": "ok", "issues": "", "flags": ""})
    df = pd.DataFrame(rows)
    # the lowest-wear non-replicate training runs end below the detection limit
    cand = df[(df["Checkpoint"] == 2) & df["Role"].isin(["completed", "corner", "interior"])]
    for eid in cand.nsmallest(N_CENSORED_RUNS, "Groove_volume_mm3")["Experiment_ID"]:
        m = df["Experiment_ID"] == eid
        lod = 2.0 * df.loc[m, "Groove_volume_mm3"].max()
        df.loc[m, ["Groove_volume_mm3", "Groove_volume_se_mm3", "Pileup_volume_mm3", "Net_volume_mm3",
                   "k_mm3_per_Nm", "k_net_mm3_per_Nm", "V_width_mm3"]] = np.nan
        df.loc[m, ["Wear_detected", "Wear_Volume_LOD_mm3", "status", "flags"]] = [False, lod, "partial", "below_LOD"]
    df = df.reindex(columns=CHECKPOINT_COLUMNS)
    df.to_csv(out / "runs_checkpoints.csv", index=False)

    # per-cycle COF (replicate overlay figure): running-in transient in step 1, then steady
    cyc = []
    for r in d.itertuples(index=False):
        t_slide = 0.0
        for j in (1, 2):
            cof = df.loc[(df["Experiment_ID"] == r.Experiment_ID) & (df["Checkpoint"] == j), "COF_steady"].iloc[0]
            n = int(round(r.Freq_Hz * r.Run_time_s / 2))
            ts = t_slide + (np.arange(n) + 0.5) / r.Freq_Hz
            c = cof + (0.03 * np.exp(-ts / 8.0) if j == 1 else 0.0) + rng.normal(0, 0.002, n)
            cyc.append(pd.DataFrame({"Experiment_ID": r.Experiment_ID, "Step": j, "Cycle": np.arange(1, n + 1),
                                     "t_s": ts, "COF_c": c, "Fz_N": r.Load_N, "Halfrange_N": c * r.Load_N,
                                     "Amp_mm": 5.0}))
            t_slide += r.Run_time_s / 2
    pd.concat(cyc, ignore_index=True)[CYCLE_COLUMNS].to_csv(out / "cof_cycles.csv", index=False)
    (out / "planted_effects.json").write_text(json.dumps(
        {"effects": EFFECTS, "lod_mm3": LOD_MM3, "censored_runs": sorted(set(df.loc[df["Wear_detected"] == False,  # noqa: E712
                                                                                  "Experiment_ID"])),
         "seed": seed}, indent=2))
    if verbose:
        print(f"synthetic data -> {out}: {len(df)} rows ({df['Experiment_ID'].nunique()} runs), "
              f"{int((df['Wear_detected'] == False).sum())} censored rows")             # noqa: E712
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="synthetic runs_checkpoints.csv (304 SS DOE)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=323)
    a = ap.parse_args(argv)
    generate(a.out, a.seed)


if __name__ == "__main__":
    main()
