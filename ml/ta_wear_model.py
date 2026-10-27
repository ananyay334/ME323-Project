"""TA-method pipeline: does the TA's wear / COF calculation give a better model target than ours?

Three ways of getting the per-run wear rate and COF are compared, each modelled the same way:

  TA original   external/ta_analyze_wear/analyze_wear.py, run unchanged (deepest scan; |COF| column mean)
  TA fixed      tribo_extract.ta_wear (same method, bugs fixed; most-worn valid scan; half-range COF)
  ours          ml.maxwear_model (most-worn valid location, band method) and COF_steady (per-cycle median)

Models (no test-day factor): constant, COF = mu0 + c L^-1/3, power law in L, f, t, GPR, ANN ensemble;
leave-one-group-out over our run groups (R04/R14/R20 together). Also: replicate spread (R04/R14/R20),
the frequency bias of each COF, and agreement between the wear-rate methods.

    python -m ml.ta_wear_model            # ~5 min
    python -m ml.ta_wear_model --quick
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from tribo_extract.config import load_config as load_extract_config
from tribo_extract import ta_wear

from . import campaign_model as cm
from .maxwear_model import peak_wear_table

ROOT = Path(__file__).resolve().parents[1]
TA_FILE = ROOT / "external" / "ta_analyze_wear" / "analyze_wear.py"
log = logging.getLogger("ml.ta_wear_model")
WEAR = {"k_ta_orig": "TA original", "k_ta_fixed": "TA fixed", "k_ours": "ours (band, most-worn location)"}
COFS = {"cof_ta_orig": "TA original", "cof_ta_fixed": "TA fixed", "COF": "ours (per-cycle half-range)"}
SETS_K = {"constant": [], "ln L": ["ln_L"], "power law: ln L, ln f, ln t": ["ln_L", "ln_f", "ln_t"]}
SETS_C = {"constant": [], "mu0 + c*L^-1/3": ["L_m13"], "power law: ln L, ln f, ln t": ["ln_L", "ln_f", "ln_t"]}
MODELS = ["linear", "gpr", "ann"]
INK, INK2, GRID, SURFACE = cm.INK, cm.INK2, cm.GRID, cm.SURFACE
METHOD_COL = {"TA original": "#e87ba4", "TA fixed": cm.COLORS[1], "ours": cm.COLORS[0]}


# ----------------------------------------------------------------------------- the three methods
def run_ta_original(raw: Path, runs: pd.DataFrame) -> pd.DataFrame:
    """The TA script's functions, unchanged, as its run_pipeline() uses them (no files written)."""
    spec = importlib.util.spec_from_file_location("ta_analyze_wear", TA_FILE)
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in runs.iterrows():
            d = raw / r["Experiment_ID"]
            files = sorted(glob.glob(str(d / "*.bcrf")),
                           key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)])
            res = []
            for p in files:
                z, px, _ = ta.read_bcrf(p)
                transposed = np.nanstd(np.nanmean(z, axis=0)) > np.nanstd(np.nanmean(z, axis=1))
                zl, mask = ta.robust_level(ta.inpaint_voids(ta.orient_across_track(z)), order=1,
                                           shoulder_fraction=0.15)
                m, _, _ = ta.compute_groove_metrics(zl, px, mask, track_len_mm=10.0)
                m.update({"file": os.path.basename(p), "transposed": transposed})
                res.append(m)
            sc = pd.DataFrame(res)
            sel = sc.loc[sc["depth_max_um"].idxmax()]
            S = 4.0 * 0.005 * r["Freq_Hz"] * r["Run_time_s"]
            fm, _ = ta.compute_friction_features(ta.read_friction_csv(str(d / f"{r['Experiment_ID']}.csv")))
            k = sel["vol_track_mm3"] / (r["Load_N"] * S)
            rows.append({"Experiment_ID": r["Experiment_ID"], "ta_orig_selected": sel["file"],
                         "ta_orig_depth_um": sel["depth_max_um"], "ta_orig_area_um2": sel["area_groove_um2"],
                         "k_ta_orig": k if k > 0 else np.nan, "ta_orig_transposed_scans": int(sc["transposed"].sum()),
                         "cof_ta_orig": fm["cof_steady_state"]})
    return pd.DataFrame(rows)


def run_ta_fixed(raw: Path, ids: list[str]) -> pd.DataFrame:
    cfg = load_extract_config()
    rows = []
    for rid in ids:
        r, _ = ta_wear.analyse_run(raw / rid, cfg)
        rows.append({"Experiment_ID": rid, "ta_fixed_selected": r["selected"], "ta_fixed_status": r["status"],
                     "k_ta_fixed": r["k"], "cof_ta_fixed": r["cof_steady_state"]})
    return pd.DataFrame(rows)


def build_table(table: Path, locations: Path, raw: Path) -> pd.DataFrame:
    d = peak_wear_table(cm.load_runs(table, locations), pd.read_csv(locations))
    d["k_ours"] = d["k_max"]
    d = d.merge(run_ta_original(raw, d), on="Experiment_ID").merge(run_ta_fixed(raw, list(d["Experiment_ID"])),
                                                                   on="Experiment_ID")
    for c in WEAR:
        d[f"log10_{c}"] = np.log10(d[c])
    return d


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    return plt


def fig_methods(d, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), facecolor=SURFACE)
    ax = axes[0]
    cm._style(ax)
    for (c, lab), mk in zip(WEAR.items(), ("^", "s", "o")):
        key = lab.split(" (")[0]
        ax.scatter(d["Load_N"], d[c], s=32, marker=mk, color=METHOD_COL[key], edgecolor=SURFACE, lw=0.8, zorder=3,
                   label=f"{key} (n={int(d[c].notna().sum())})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3])
    ax.set_yticklabels(["1e-6", "1e-5", "1e-4", "1e-3"])
    ax.set_xlabel("Load (N)", color=INK2)
    ax.set_ylabel("k (mm³ / N·m)", color=INK2)
    ax.set_title("Wear rate per run, by method", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    ax = axes[1]
    cm._style(ax)
    for c, mk in zip(("cof_ta_orig", "cof_ta_fixed"), ("^", "s")):
        key = COFS[c]
        ax.scatter(d["Freq_Hz"], d[c] / d["COF"], s=32, marker=mk, color=METHOD_COL[key], edgecolor=SURFACE,
                   lw=0.8, zorder=3, label=key)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Frequency (Hz)", color=INK2)
    ax.set_ylabel("COF / our COF", color=INK2)
    ax.set_title("COF relative to ours vs frequency", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(table: Path, locations: Path, raw: Path, out: Path, quick=False, n_jobs=-1) -> dict:
    t0 = time.time()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d = build_table(table, locations, raw)
    keep = ["Experiment_ID", "Load_N", "Freq_Hz", "Run_time_s", "group", "k_ta_orig", "k_ta_fixed", "k_ours",
            "cof_ta_orig", "cof_ta_fixed", "COF", "ta_orig_selected", "ta_orig_depth_um", "ta_orig_transposed_scans",
            "ta_fixed_selected", "ta_fixed_status", "max_location", "wear_status_max"]
    d[keep].to_csv(out / "run_table.csv", index=False)

    jobs = []
    for c in WEAR:
        t = f"log10_{c}"
        dt = d.dropna(subset=[t]).reset_index(drop=True)
        for s, f in SETS_K.items():
            for m in (["mean"] if not f else MODELS):
                jobs.append(("wear", c, t, s, f, m, dt))
    for c in COFS:
        dt = d.dropna(subset=[c]).reset_index(drop=True)
        for s, f in SETS_C.items():
            for m in (["mean"] if not f else (["linear"] if s.startswith("mu0") else MODELS)):
                jobs.append(("COF", c, c, s, f, m, dt))
    preds = Parallel(n_jobs=n_jobs)(delayed(cm.logo_predict)(dt, t, f, m, quick) for _, _, t, _, f, m, dt in jobs)
    rows = []
    for (kind, c, t, s, f, m, dt), p in zip(jobs, preds):
        y = dt[t].to_numpy(float)
        rows.append({"kind": kind, "method": (WEAR if kind == "wear" else COFS)[c], "inputs": s, "model": m,
                     "n_runs": len(dt), **cm.metrics(y, p)})
    cv = pd.DataFrame(rows).sort_values(["kind", "method", "RMSE"]).reset_index(drop=True)
    cv.to_csv(out / "cv_metrics.csv", index=False)

    # data quality: replicate spread vs spread over runs, frequency bias, agreement
    rep = d[d["group"] == "R04"]
    q = []
    for c, lab in {**{f"log10_{k}": v for k, v in WEAR.items()}, **COFS}.items():
        v = d[c].dropna()
        rv = rep[c].dropna()
        q.append({"quantity": ("log10 k: " if c.startswith("log10") else "COF: ") + lab, "runs": len(v),
                  "replicates with a value": len(rv), "replicate SD": rv.std(ddof=1) if len(rv) > 1 else np.nan,
                  "SD over runs": v.std(ddof=1),
                  "corr with frequency": stats.pearsonr(d.loc[v.index, "Freq_Hz"], v)[0]})
    quality = pd.DataFrame(q)
    quality["SD over runs / replicate SD"] = quality["SD over runs"] / quality["replicate SD"]
    quality.to_csv(out / "data_quality.csv", index=False)
    agree = []
    for a, b in (("k_ta_orig", "k_ours"), ("k_ta_fixed", "k_ours"), ("k_ta_orig", "k_ta_fixed")):
        ok = d[a].notna() & d[b].notna()
        agree.append({"pair": f"{WEAR[a].split(' (')[0]} vs {WEAR[b].split(' (')[0]}", "runs with both": int(ok.sum()),
                      "Spearman ρ": stats.spearmanr(d.loc[ok, a], d.loc[ok, b])[0] if ok.sum() > 2 else np.nan,
                      "median ratio": float(np.median(d.loc[ok, a] / d.loc[ok, b])) if ok.any() else np.nan})
    agree = pd.DataFrame(agree)
    agree.to_csv(out / "method_agreement.csv", index=False)
    fig_methods(d, out / "figures" / "methods.png")
    ctx = {"d": d, "cv": cv, "quality": quality, "agree": agree, "quick": quick, "minutes": (time.time() - t0) / 60}
    write_report(out, ctx)
    log.info("done in %.1f min -> %s", ctx["minutes"], out / "report.md")
    return ctx


def write_report(out: Path, ctx: dict) -> None:
    d, cv, q, ag = ctx["d"], ctx["cv"], ctx["quality"], ctx["agree"]
    L = ["# TA wear-rate method vs ours: model check", "",
         f"`python -m ml.ta_wear_model{' --quick' if ctx['quick'] else ''}` · {ctx['minutes']:.1f} min · "
         "leave-one-group-out over our runs (R04/R14/R20 together), no day factor", "",
         "## Bugs found in analyze_wear.py (fixed in tribo_extract/ta_wear.py)", "",
         f"1. Orientation chosen on the unlevelled map: {int(d['ta_orig_transposed_scans'].sum())} of "
         f"{4 * len(d)} scans transposed, so the profile runs along the track.",
         "2. The deepest point can be picked inside the baseline (shoulder) rows.",
         "3. Points shallower than the cutoff are zeroed one by one: shallow grooves vanish (area 0) and the rest "
         "are biased low.",
         "4. Pile-up sums everything above the baseline over the whole image outside the groove.",
         "5. No check that the shoulders are unworn: the groove can extend into the reference rows.",
         "6. The scan is selected by deepest point (texture valleys on rough plates), invalid scans included.",
         "7. COF = mean of the instrument |Fx|/Fz column per second: reversal samples bias it low, more at high "
         "frequency; the deceleration second is not trimmed.",
         "8. Load, frequency and time are hard-coded R01 defaults; the friction log is the first *.csv in the folder "
         "(after one run, possibly the script's own output); outputs are written into the data folder.", "",
         "Not fixable without changing the method: the groove is taken as the deepest region of the mean profile. "
         "On polished samples that is the track; on our plates (Sa 3–9 µm, waviness) it is often waviness, "
         "because nothing locates the track itself.", "",
         "## Data quality of each target", "", cm._tab(q), "",
         "## Agreement between wear-rate methods", "", cm._tab(ag), ""]
    for kind in ("wear", "COF"):
        sub = cv[cv["kind"] == kind][["method", "inputs", "model", "n_runs", "R2", "RMSE", "MAE"]]
        sub = sub.rename(columns={"n_runs": "runs", "R2": "R²"})
        L += [f"## Model scores: {'log10 wear rate' if kind == 'wear' else 'COF'}", "", cm._tab(sub), ""]
    L += ["![methods](figures/methods.png)", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TA wear method vs ours")
    ap.add_argument("--table", default=str(ROOT / "data" / "processed" / "runs_checkpoints.csv"))
    ap.add_argument("--locations", default=str(ROOT / "data" / "processed" / "wear_locations.csv"))
    ap.add_argument("--raw", default=str(ROOT / "testdata"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "results" / "ta_wear_model"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run(Path(a.table), Path(a.locations), Path(a.raw), Path(a.out), quick=a.quick, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
