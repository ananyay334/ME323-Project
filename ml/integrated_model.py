"""Integrated pipeline: the useful ideas from the TA's analyze_wear.py added to our method, one at a time.

What is taken from the TA's script and how it is tested here (all on all runs, leave-one-group-out,
R04/R14/R20 as one group, no test-day factor):

Wear (our band method, most-worn valid location as in ``ml.maxwear_model``) with
  W1  robust levelling: 3-pass 2-sigma clipping of the reference plane, and of each row before the
      across-track mean (the TA's robust_level, extended to the profile so pits/debris drop out);
  W2  Savitzky-Golay smoothing of the mean profile (window 51 px, cubic; the TA's compute_groove_metrics);
  W3  both;
  W4  the TA's detection rule: a location counts if its maximum depth exceeds max(2.5 x baseline SD, 0.25 um)
      (instead of our area-vs-uncertainty LOD);
  W5  the TA's depth metric as the wear target: maximum wear depth of the most-worn location.
  Also reported: the TA's plasticity ratio (pile-up / groove) per run.
COF (from our per-cycle, reversal-free COF) with the TA's three windows
  C1  first 10 s (cof_initial), C2 last 10 s (cof_steady_state), C3 whole run (cof_overall_mean),
  compared with our COF_steady (median over the second half).
Models: constant, COF = mu0 + c L^-1/3, power law in L, f, t (OLS), the same with Huber regression (the TA's
script imports sklearn's HuberRegressor), GPR, ANN ensemble.

    python -m ml.integrated_model             # ~5-8 min
    python -m ml.integrated_model --quick
"""
from __future__ import annotations

import argparse
import copy
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tribo_extract.config import load_config as load_extract_config
from tribo_extract.height_io import read_bcrf
from tribo_extract.layout import classify_run
from tribo_extract.wear import measure_track_band

from . import campaign_model as cm

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("ml.integrated_model")
TRACK_MM = 10.0
VARIANTS = {"ours (current)": {}, "+ robust levelling": {"clip_sigma": 2.0},
            "+ Savitzky-Golay profile": {"profile_savgol_px": 51},
            "+ robust + Savitzky-Golay": {"clip_sigma": 2.0, "profile_savgol_px": 51}}
INK, INK2, GRID, SURFACE = cm.INK, cm.INK2, cm.GRID, cm.SURFACE


# ----------------------------------------------------------------------------- wear measurement
def _measure(path: str, band_opts: dict, cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["wear"]["band"] = {**c["wear"].get("band", {}), **band_opts}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, _ = measure_track_band(read_bcrf(path), c)
    return res


def measure_all(raw: Path, ids: list[str], n_jobs: int) -> pd.DataFrame:
    cfg = load_extract_config()
    scans = []
    for rid in ids:
        _, sc, _ = classify_run(raw / rid, cfg["layout"], None)
        for _, s in sc.iterrows():
            scans.append((rid, s["location"], str(s["height"])))
    jobs = [(v, opts, rid, loc, p) for v, opts in VARIANTS.items() for rid, loc, p in scans]
    res = Parallel(n_jobs=n_jobs)(delayed(_measure)(p, opts, cfg) for _, opts, _, _, p in jobs)
    rows = []
    for (v, _, rid, loc, p), r in zip(jobs, res):
        flags = str(r.get("wear_flags", ""))
        found = "no_track_found" not in flags
        rows.append({"variant": v, "Experiment_ID": rid, "Location": loc, "file": Path(p).name,
                     "valid": found and bool(r.get("reference_ok", False)),
                     "groove_um2": r.get("worn_area_um2", np.nan) if found else np.nan,
                     "pileup_um2": r.get("pileup_area_um2", np.nan), "area_LOD_um2": r.get("area_LOD_um2", np.nan),
                     "depth_um": r.get("track_max_depth_um", np.nan), "ta_cutoff_um": r.get("ta_depth_cutoff_um", np.nan),
                     "width_um": r.get("track_width_um", np.nan)})
    return pd.DataFrame(rows)


def per_run(loc: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per run and variant: k_max with our LOD rule and with the TA depth rule, max depth, plasticity."""
    out = []
    FS = runs.set_index("Experiment_ID")["Load_N"] * runs.set_index("Experiment_ID")["Distance_m"]
    for (v, rid), g in loc.groupby(["variant", "Experiment_ID"]):
        ok = g[g["valid"]]
        row = {"variant": v, "Experiment_ID": rid, "n_valid": len(ok)}
        for rule, det in (("lod", ok["groove_um2"] > ok["area_LOD_um2"]),
                          ("ta", ok["depth_um"] >= ok["ta_cutoff_um"])):
            d = ok[det]
            if len(d):
                top = d.loc[d["groove_um2"].idxmax()]
                row[f"k_{rule}"] = top["groove_um2"] * 1e-6 * TRACK_MM / FS[rid]
                row[f"loc_{rule}"] = top["Location"]
            else:
                row[f"k_{rule}"], row[f"loc_{rule}"] = np.nan, ""
        row["depth_max_um"] = ok["depth_um"].max() if len(ok) else np.nan
        tot_g, tot_p = ok["groove_um2"].sum(), ok["pileup_um2"].sum()
        row["plasticity_ratio"] = tot_p / tot_g if tot_g > 0 else np.nan
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- COF windows
def cof_windows(cycles: Path) -> pd.DataFrame:
    c = pd.read_csv(cycles).dropna(subset=["COF_c"])
    rows = []
    for rid, g in c.groupby("Experiment_ID"):
        t = g["t_s"].to_numpy()
        rows.append({"Experiment_ID": rid, "cof_initial": g.loc[t <= t.min() + 10, "COF_c"].mean(),
                     "cof_final10": g.loc[t >= t.max() - 10, "COF_c"].mean(), "cof_overall": g["COF_c"].mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- models
def huber():
    return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000))


def logo(d: pd.DataFrame, target: str, feats: list[str], model: str, quick: bool) -> np.ndarray:
    if model != "huber":
        return cm.logo_predict(d, target, feats, model, quick)
    y, g = d[target].to_numpy(float), d["group"].to_numpy(str)
    X = d[feats].to_numpy(float)
    p = np.empty(len(d))
    for grp in np.unique(g):
        te = g == grp
        p[te] = huber().fit(X[~te], y[~te]).predict(X[te])
    return p


# ----------------------------------------------------------------------------- figure
def fig_summary(summary: pd.DataFrame, out: Path):
    """Wear: best R² of any model (n runs in the label). COF: R² of COF = mu0 + c L^-1/3 (the chosen model)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), facecolor=SURFACE)
    for ax, kind, col, xl in ((axes[0], "wear", "best_R2", "best leave-one-run-out R², any model"),
                              (axes[1], "COF", "R² of reference model", "R² of COF = μ₀ + c·L^(−1/3)")):
        cm._style(ax)
        s = summary[summary["kind"] == kind].iloc[::-1]
        base = s[s["is_baseline"]][col].iloc[0]
        cols = [cm.COLORS[0] if b else cm.COLORS[1] for b in s["is_baseline"]]
        ax.scatter(s[col], range(len(s)), s=46, color=cols, edgecolor=SURFACE, zorder=3)
        for i, (_, r) in enumerate(s.iterrows()):
            ax.plot([base, r[col]], [i, i], color=GRID, lw=1.2, zorder=1)
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(s["target"], s["runs"])], fontsize=7.5)
        ax.set_xlabel(xl, color=INK2, fontsize=8.5)
        ax.set_title("Wear rate" if kind == "wear" else "COF", loc="left", color=INK)
        ax.grid(False, axis="y")
    fig.text(0.01, 0.01, "blue = current pipeline; orange = with a TA idea; dashed = no better than predicting the mean",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(table: Path, locations: Path, cycles: Path, raw: Path, out: Path, quick=False, n_jobs=-1) -> dict:
    t0 = time.time()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = cm.load_runs(table, locations)
    loc = measure_all(raw, list(runs["Experiment_ID"]), n_jobs)
    loc.to_csv(out / "locations_by_variant.csv", index=False)
    pr = per_run(loc, runs)
    pr.to_csv(out / "runs_by_variant.csv", index=False)
    cw = cof_windows(cycles)
    d = runs.merge(cw, on="Experiment_ID")
    targets = {}                                   # name -> (kind, column)
    for v in VARIANTS:
        p = pr[pr["variant"] == v].set_index("Experiment_ID")
        d[f"log10_k_lod|{v}"] = np.log10(d["Experiment_ID"].map(p["k_lod"]))
        targets[f"k (LOD rule), {v}"] = ("wear", f"log10_k_lod|{v}")
    p0 = pr[pr["variant"] == "ours (current)"].set_index("Experiment_ID")
    d["log10_k_ta_rule"] = np.log10(d["Experiment_ID"].map(p0["k_ta"]))
    d["log10_depth"] = np.log10(d["Experiment_ID"].map(p0["depth_max_um"]))
    d["plasticity_ratio"] = d["Experiment_ID"].map(p0["plasticity_ratio"])
    targets["k, TA depth rule (W4)"] = ("wear", "log10_k_ta_rule")
    targets["max wear depth (W5)"] = ("wear", "log10_depth")
    for name, col in (("COF_steady (current: 2nd-half median)", "COF"), ("COF first 10 s (C1)", "cof_initial"),
                      ("COF last 10 s (C2)", "cof_final10"), ("COF whole run (C3)", "cof_overall")):
        targets[name] = ("COF", col)
    sets = {"wear": {"constant": [], "ln L": ["ln_L"], "power law L,f,t": ["ln_L", "ln_f", "ln_t"]},
            "COF": {"constant": [], "mu0 + c*L^-1/3": ["L_m13"], "power law L,f,t": ["ln_L", "ln_f", "ln_t"]}}
    jobs = []
    for name, (kind, col) in targets.items():
        dt = d[np.isfinite(d[col])].reset_index(drop=True)
        for s, f in sets[kind].items():
            models = ["mean"] if not f else (["linear", "huber"] if len(f) == 1 else ["linear", "huber", "gpr", "ann"])
            for m in models:
                jobs.append((name, kind, col, s, f, m, dt))
    preds = Parallel(n_jobs=n_jobs)(delayed(logo)(dt, col, f, m, quick) for _, _, col, _, f, m, dt in jobs)
    rows = []
    for (name, kind, col, s, f, m, dt), p in zip(jobs, preds):
        rows.append({"target": name, "kind": kind, "inputs": s, "model": m, "n_runs": len(dt),
                     **cm.metrics(dt[col].to_numpy(float), p)})
    cv = pd.DataFrame(rows).sort_values(["kind", "target", "RMSE"]).reset_index(drop=True)
    cv.to_csv(out / "cv_metrics.csv", index=False)
    rep = d[d["group"] == "R04"]
    summ = []
    for name, (kind, col) in targets.items():
        v, rv = d[col].dropna(), rep[col].dropna()
        sub = cv[cv["target"] == name]
        best = sub.sort_values("R2", ascending=False).iloc[0]
        named = sub[(sub["inputs"] == ("mu0 + c*L^-1/3" if kind == "COF" else "constant")) &
                    (sub["model"] == ("linear" if kind == "COF" else "mean"))].iloc[0]
        summ.append({"kind": kind, "target": name, "runs": len(v), "replicate SD": rv.std(ddof=1) if len(rv) > 1 else np.nan,
                     "replicates": len(rv), "SD over runs": v.std(ddof=1),
                     "R² of reference model": named["R2"], "best_R2": best["R2"],
                     "best model": f"{best['model']} on {best['inputs']}",
                     "is_baseline": name.endswith("ours (current)") or name.startswith("COF_steady")})
    summary = pd.DataFrame(summ)
    checks = robustness_checks(d, targets)
    checks.to_csv(out / "robustness_checks.csv", index=False)
    fig_cof_initial(d, out / "figures" / "cof_first10s_vs_load.png")
    summary.to_csv(out / "summary.csv", index=False)
    fig_summary(summary, out / "figures" / "summary.png")
    ctx = {"d": d, "cv": cv, "summary": summary, "pr": pr, "loc": loc, "checks": checks, "quick": quick,
           "minutes": (time.time() - t0) / 60}
    write_report(out, ctx)
    log.info("done in %.1f min -> %s", ctx["minutes"], out / "report.md")
    return ctx


def _logo_r2_linear(dd: pd.DataFrame, y: np.ndarray, feats: list[str]) -> float:
    X = np.c_[np.ones(len(dd)), dd[feats].to_numpy(float)]
    e = np.empty(len(dd))
    g = dd["group"].to_numpy(str)
    for grp in np.unique(g):
        te = g == grp
        c = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        e[te] = X[te] @ c - y[te]
    return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()


def robustness_checks(d: pd.DataFrame, targets: dict, n_perm: int = 2000) -> pd.DataFrame:
    """Permutation test of the linear fits (is the CV R² better than with shuffled targets?) and, for wear,
    the power-law exponents of the wear VOLUME (Archard: volume ~ L * distance, exponents 1)."""
    rng = np.random.default_rng(0)
    rows = []
    for name, (kind, col) in targets.items():
        dd = d[np.isfinite(d[col])].reset_index(drop=True)
        feats = ["L_m13"] if kind == "COF" else ["ln_L", "ln_f", "ln_t"]
        y = dd[col].to_numpy(float)
        r2 = _logo_r2_linear(dd, y, feats)
        perm = np.array([_logo_r2_linear(dd, rng.permutation(y), feats) for _ in range(n_perm)])
        row = {"target": name, "runs": len(dd), "linear model": "mu0 + c*L^-1/3" if kind == "COF" else "power law L,f,t",
               "CV R²": r2, "permutation p": float(np.mean(perm >= r2))}
        if kind == "wear" and col.startswith("log10_k"):
            v = y + np.log10(dd["Load_N"] * dd["Distance_m"]).to_numpy()        # log10 wear volume
            X = np.c_[np.ones(len(dd)), dd[["ln_L", "ln_f", "ln_t"]].to_numpy(float)]
            c = np.linalg.lstsq(X, v, rcond=None)[0] * np.log(10)
            row.update({"volume exponent L": c[1], "volume exponent f": c[2], "volume exponent t": c[3]})
        rows.append(row)
    return pd.DataFrame(rows)


def fig_cof_initial(d: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), facecolor=SURFACE, sharey=True)
    grid = np.geomspace(9.5, 105, 100)
    for ax, (col, ttl) in zip(axes, (("cof_initial", "COF, first 10 s of sliding"),
                                      ("COF", "COF_steady (second half)"))):
        cm._style(ax)
        dd = d.dropna(subset=[col]).reset_index(drop=True)
        c = cm.ols(dd[["L_m13"]].to_numpy(float), dd[col].to_numpy(float))
        _, bp = cm.group_bootstrap(dd, ["L_m13"], col, 1000, 3, (grid ** (-1 / 3))[:, None])
        lo, hi = np.percentile(bp, [2.5, 97.5], axis=0)
        ax.fill_between(grid, lo, hi, color=cm.COLORS[0], alpha=0.14, lw=0)
        ax.plot(grid, c[0] + c[1] * grid ** (-1 / 3), color=cm.COLORS[0], lw=2)
        ax.scatter(dd["Load_N"], dd[col], s=34, color=cm.COLORS[0], edgecolor=SURFACE, lw=1.0, zorder=3)
        r2 = _logo_r2_linear(dd, dd[col].to_numpy(float), ["L_m13"])
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100])
        ax.set_xticklabels(["10", "20", "50", "100"])
        ax.set_xlabel("Load (N)", color=INK2)
        ax.set_title(f"{ttl}\n{c[0]:.3f} + {c[1]:.3f}·L^(−1/3), CV R² = {r2:+.2f}", loc="left", color=INK, fontsize=9.5)
    axes[0].set_ylabel("COF", color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def write_report(out: Path, ctx: dict) -> None:
    s, cv, d = ctx["summary"].copy(), ctx["cv"], ctx["d"]
    s = s.drop(columns=["is_baseline"]).rename(columns={"best_R2": "best R²"})
    L = ["# Integrated pipeline: TA ideas added to our method", "",
         f"`python -m ml.integrated_model{' --quick' if ctx['quick'] else ''}` · {ctx['minutes']:.1f} min · "
         "all runs, leave-one-group-out (R04/R14/R20 together), no day factor", "",
         "Reference model: constant for wear, COF = μ₀ + c·L^(−1/3) for COF. Wear targets in log10; replicate SD "
         "is over R04/R14/R20 where they have a value.", "", cm._tab(s), "",
         f"Median plasticity ratio (pile-up / groove over valid locations, current method): "
         f"{np.nanmedian(d['plasticity_ratio']):.2f}; runs with ratio > 1: "
         f"{int((d['plasticity_ratio'] > 1).sum())} of {int(d['plasticity_ratio'].notna().sum())}.", "",
         "## Robustness checks", "", cm._tab(ctx["checks"]), "",
         "Permutation p = share of 2000 shuffles of the target that score at least as well. Volume exponents near 0 "
         "mean the wear volume does not grow with load or sliding distance, so a k trend then only reflects the "
         "division by F·S.", "",
         "![summary](figures/summary.png)", "", "![COF first 10 s](figures/cof_first10s_vs_load.png)", "",
         "## All scores", "",
         cm._tab(cv[["target", "inputs", "model", "n_runs", "R2", "RMSE"]].rename(columns={"R2": "R²"})), ""]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TA ideas integrated into our method: does the model improve?")
    ap.add_argument("--table", default=str(ROOT / "data" / "processed" / "runs_checkpoints.csv"))
    ap.add_argument("--locations", default=str(ROOT / "data" / "processed" / "wear_locations.csv"))
    ap.add_argument("--cycles", default=str(ROOT / "data" / "processed" / "cof_cycles.csv"))
    ap.add_argument("--raw", default=str(ROOT / "testdata"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "results" / "integrated_model"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run(Path(a.table), Path(a.locations), Path(a.cycles), Path(a.raw), Path(a.out), quick=a.quick, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
