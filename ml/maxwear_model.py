"""Peak-wear pipeline: wear rate from the most-worn scan location, COF = mu0 + c * L^(-1/3), no day factor.

Differences from ``ml.campaign_model``:

* **Wear rate from the maximum profile.** For each run the groove area of the most-worn of its 4 scan
  locations is used instead of the mean over locations:
  k_max = (max groove area * 10 mm track length) / (load * sliding distance).
  Only locations with unworn surface on both sides of the track count (``reference_one_sided`` scans are
  extrapolated and are skipped). R12 and R19 have no such location and stay "not measurable". If even the
  maximum is below that location's detection limit (2 x the reference uncertainty), the run is censored
  ("< LOD"): it is left out of the fit (main analysis) or imputed at LOD/2 (sensitivity analysis).
* **No test-day factor** anywhere.
* **COF model** = mu0 + c * L^(-1/3) (Hertz-adhesion form), compared with the DOE inputs and with a constant.

All runs (training + test), one row per run, leave-one-group-out CV (R04/R14/R20 = one group).
Candidates: input sets x {linear, ridge, GPR, ANN ensemble}. Outputs in ``ml/results/maxwear_model/``.

    python -m ml.maxwear_model            # ~5-10 min (ANN ensembles dominate)
    python -m ml.maxwear_model --quick
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import campaign_model as cm

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("ml.maxwear_model")
TRACK_MM = 10.0
TARGETS = {
    "COF": "COF_steady (second half of the run)",
    "log10_kmax": "log10 k from the most-worn location [mm³/(N·m)]",
    "log10_kmax_imp": "log10 k_max, censored runs imputed at LOD/2 (sensitivity)",
    "log10_kavg": "log10 k from the mean over locations (previous target, for reference)",
}
INPUT_SETS = {
    "COF": {"constant": [], "L^-1/3 (physics)": ["L_m13"], "DOE: ln L, ln f, ln t": ["ln_L", "ln_f", "ln_t"]},
    "k": {"constant (Archard: k fixed)": [], "ln L": ["ln_L"], "DOE: ln L, ln f, ln t": ["ln_L", "ln_f", "ln_t"]},
}
MODELS = ["linear", "ridge", "gpr", "ann"]
INK, INK2, GRID, SURFACE = cm.INK, cm.INK2, cm.GRID, cm.SURFACE
BLUE, ORANGE = cm.COLORS[0], cm.COLORS[1]


# ----------------------------------------------------------------------------- data
def peak_wear_table(runs: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Per run: most-worn valid location, its groove area and LOD, status and k_max."""
    w = locations[locations["Checkpoint"] == 2].copy()
    flags = w["wear_flags"].fillna("").astype(str)
    w["valid"] = w["Groove_area_um2"].notna() & ~flags.str.contains("reference_one_sided")
    rows = []
    for _, r in runs.iterrows():
        g = w[(w["Experiment_ID"] == r["Experiment_ID"]) & w["valid"]]
        base = {"Experiment_ID": r["Experiment_ID"], "n_valid_locations": len(g)}
        if g.empty:
            rows.append({**base, "max_location": "", "max_groove_area_um2": np.nan, "max_area_LOD_um2": np.nan,
                         "wear_status_max": "not_measurable"})
            continue
        top = g.loc[g["Groove_area_um2"].idxmax()]
        det = top["Groove_area_um2"] > top["Area_LOD_um2"]
        rows.append({**base, "max_location": top["Location"], "max_groove_area_um2": top["Groove_area_um2"],
                     "max_area_LOD_um2": top["Area_LOD_um2"], "wear_status_max": "detected" if det else "below_LOD"})
    t = pd.DataFrame(rows)
    out = runs.merge(t, on="Experiment_ID", how="left")
    FS = out["Load_N"] * out["Distance_m"]
    um2_to_mm3 = 1e-6 * TRACK_MM
    out["V_max_mm3"] = out["max_groove_area_um2"] * um2_to_mm3
    out["k_max"] = np.where(out["wear_status_max"] == "detected", out["V_max_mm3"] / FS, np.nan)
    out["k_max_LOD"] = out["max_area_LOD_um2"] * um2_to_mm3 / FS
    out["log10_kmax"] = np.log10(out["k_max"])
    imp = np.where(out["wear_status_max"] == "below_LOD", out["k_max_LOD"] / 2, out["k_max"])
    out["log10_kmax_imp"] = np.log10(imp)
    kavg = pd.to_numeric(out["k_mm3_per_Nm"], errors="coerce")
    out["log10_kavg"] = np.log10(kavg.where(kavg > 0))
    return out


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    return plt


def fig_cof(d, c, lo, hi, grid_L, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.9), facecolor=SURFACE)
    cm._style(ax)
    ax.fill_between(grid_L, lo, hi, color=BLUE, alpha=0.14, lw=0)
    ax.plot(grid_L, c[0] + c[1] * grid_L ** (-1 / 3), color=BLUE, lw=2)
    ax.scatter(d["Load_N"], d["COF"], s=40, color=BLUE, edgecolor=SURFACE, lw=1.1, zorder=3, label="runs")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_xlabel("Load (N)", color=INK2)
    ax.set_ylabel("COF (second half of run)", color=INK2)
    ax.set_title("COF = μ₀ + c·L^(−1/3), no day factor (95 % band)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_k(d, out):
    """k from the most-worn location vs from the mean, per run, against load."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.9), facecolor=SURFACE)
    cm._style(ax)
    det = d["wear_status_max"] == "detected"
    cen = d["wear_status_max"] == "below_LOD"
    ax.scatter(d.loc[det, "Load_N"], d.loc[det, "k_max"], s=44, marker="o", color=BLUE, edgecolor=SURFACE,
               lw=1.1, zorder=3, label="k_max (most-worn location)")
    ax.scatter(d.loc[cen, "Load_N"], d.loc[cen, "k_max_LOD"], s=44, marker="v", facecolor="none", edgecolor=BLUE,
               lw=1.2, zorder=3, label="k_max < LOD (upper bound)")
    kav = pd.to_numeric(d["k_mm3_per_Nm"], errors="coerce")
    ok = kav > 0
    ax.scatter(d.loc[ok, "Load_N"], kav[ok], s=30, marker="s", facecolor="none", edgecolor=ORANGE, lw=1.2,
               zorder=2, label="k from mean of locations (previous)")
    for _, r in d[det & ok].iterrows():
        ax.plot([r["Load_N"]] * 2, [float(r["k_mm3_per_Nm"]), r["k_max"]], color=GRID, lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_yticks([3e-6, 5e-6, 1e-5, 2e-5, 5e-5])
    ax.set_yticklabels(["3e-6", "5e-6", "1e-5", "2e-5", "5e-5"])
    ax.minorticks_off()
    ax.set_xlabel("Load (N)", color=INK2)
    ax.set_ylabel("k (mm³ / N·m)", color=INK2)
    ax.set_title("Specific wear rate: most-worn location vs mean", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_parity(panels, out):
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), facecolor=SURFACE)
    for ax, (title, y, p) in zip(np.atleast_1d(axes), panels):
        cm._style(ax)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        pad = 0.06 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, color=GRID, lw=1.2)
        ax.scatter(y, p, s=34, color=BLUE, edgecolor=SURFACE, lw=1.0, zorder=3)
        m = cm.metrics(y, p)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("measured", color=INK2)
        ax.set_ylabel("predicted (run left out)", color=INK2)
        ax.set_title(f"{title}\nR² = {m['R2']:+.2f}, RMSE = {m['RMSE']:.3g}", loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(table: Path, locations: Path, out: Path, quick=False, n_boot=2000, n_jobs=-1) -> dict:
    t0 = time.time()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d = peak_wear_table(cm.load_runs(table, locations), pd.read_csv(locations))
    cols = ["Experiment_ID", "Role", "Load_N", "Freq_Hz", "Run_time_s", "Distance_m", "COF", "n_valid_locations",
            "max_location", "max_groove_area_um2", "max_area_LOD_um2", "wear_status_max", "V_max_mm3", "k_max",
            "k_max_LOD", "k_mm3_per_Nm", "Wear_detected"]
    d[cols].rename(columns={"k_mm3_per_Nm": "k_mean_locations", "Wear_detected": "detected_mean_locations"}).to_csv(
        out / "run_table.csv", index=False)

    jobs = []
    for t in TARGETS:
        sets = INPUT_SETS["COF" if t == "COF" else "k"]
        dt = d.dropna(subset=[t]).reset_index(drop=True)
        for s, feats in sets.items():
            for m in (["mean"] if not feats else MODELS):
                jobs.append((t, s, feats, m, dt))
    log.info("%d runs; %d CV candidates", len(d), len(jobs))
    preds = Parallel(n_jobs=n_jobs)(delayed(cm.logo_predict)(dt, t, f, m, quick) for t, s, f, m, dt in jobs)
    rows, P = [], {}
    for (t, s, f, m, dt), p in zip(jobs, preds):
        y = dt[t].to_numpy(float)
        rows.append({"target": t, "inputs": s, "model": m, "n_runs": len(dt), "groups": dt["group"].nunique(),
                     **cm.metrics(y, p)})
        P[(t, s, m)] = (dt, p)
    cv = pd.DataFrame(rows).sort_values(["target", "RMSE"]).reset_index(drop=True)
    # tie with the best (1-SE rule on per-run squared error)
    cv["tied_with_best"] = ""
    for t in TARGETS:
        sub = cv[(cv["target"] == t) & (cv["model"] != "mean")]
        b = sub.iloc[0]
        dtb, pb = P[(t, b["inputs"], b["model"])]
        eb = (pb - dtb[t].to_numpy(float)) ** 2
        for i, r in cv[cv["target"] == t].iterrows():
            dti, pi = P[(t, r["inputs"], r["model"])]
            diff = (pi - dti[t].to_numpy(float)) ** 2 - eb
            cv.loc[i, "tied_with_best"] = "yes" if diff.mean() <= diff.std(ddof=1) / np.sqrt(len(diff)) else "no"
    cv.to_csv(out / "cv_metrics.csv", index=False)

    # replicate noise per target
    rep = d[d["group"] == "R04"]
    noise = {t: (float(rep[t].std(ddof=1)) if rep[t].notna().sum() > 1 else np.nan, int(rep[t].notna().sum()))
             for t in TARGETS}

    # final fits with run-bootstrap intervals
    coef_rows = []
    grid_L = np.geomspace(d["Load_N"].min() * 0.95, d["Load_N"].max() * 1.05, 120)
    dc = d.dropna(subset=["COF"])
    c = cm.ols(dc[["L_m13"]].to_numpy(float), dc["COF"].to_numpy(float))
    bc, bp = cm.group_bootstrap(dc, ["L_m13"], "COF", n_boot, 11, (grid_L ** (-1 / 3))[:, None])
    dmu = lambda cc: cc[1] * (100 ** (-1 / 3) - 10 ** (-1 / 3))  # noqa: E731
    for name, fn in [("mu0", lambda cc: cc[0]), ("c (L^-1/3 term)", lambda cc: cc[1]),
                     ("COF change 10 N -> 100 N", dmu)]:
        s = np.array([fn(r) for r in bc])
        coef_rows.append({"model": "COF = mu0 + c*L^(-1/3)", "term": name, "estimate": fn(c),
                          "ci_low": np.percentile(s, 2.5), "ci_high": np.percentile(s, 97.5)})
    lo, hi = np.percentile(bp, [2.5, 97.5], axis=0)
    fig_cof(dc, c, lo, hi, grid_L, out / "figures" / "cof_vs_load.png")
    for t in ("log10_kmax", "log10_kmax_imp", "log10_kavg"):
        dk = d.dropna(subset=[t])
        feats = ["ln_L", "ln_f", "ln_t"]
        ck = cm.ols(dk[feats].to_numpy(float), dk[t].to_numpy(float))
        bk, _ = cm.group_bootstrap(dk, feats, t, n_boot, 12)
        lab = {"log10_kmax": "k_max", "log10_kmax_imp": "k_max (LOD/2 imputed)", "log10_kavg": "k mean"}[t]
        for j, name in enumerate(["exponent of L", "exponent of f", "exponent of t"], start=1):
            s = bk[:, j] * cm.LN10                     # log10 k per ln x  ->  power-law exponent
            coef_rows.append({"model": f"{lab} ∝ L^a f^b t^c (n = {len(dk)})", "term": name,
                              "estimate": ck[j] * cm.LN10, "ci_low": np.percentile(s, 2.5),
                              "ci_high": np.percentile(s, 97.5)})
        kk = 10 ** dk[t]
        coef_rows.append({"model": f"{lab} ∝ L^a f^b t^c (n = {len(dk)})", "term": "geometric mean k",
                          "estimate": float(10 ** dk[t].mean()), "ci_low": float(kk.min()), "ci_high": float(kk.max())})
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(out / "coefficients.csv", index=False)
    fig_k(d, out / "figures" / "k_max_vs_mean.png")
    panels = []
    for t, s in (("COF", "L^-1/3 (physics)"), ("log10_kmax", "constant (Archard: k fixed)"),
                 ("log10_kmax", "DOE: ln L, ln f, ln t")):
        m = "mean" if s.startswith("constant") else "linear"
        dt, p = P[(t, s, m)]
        panels.append((f"{'COF' if t == 'COF' else 'log10 k_max'}: {s.split(' (')[0]}", dt[t].to_numpy(float), p))
    fig_parity(panels, out / "figures" / "parity.png")

    ctx = {"d": d, "cv": cv, "coefs": coefs, "noise": noise, "quick": quick, "minutes": (time.time() - t0) / 60}
    write_report(out, ctx)
    log.info("done in %.1f min -> %s", ctx["minutes"], out / "report.md")
    return ctx


def write_report(out: Path, ctx: dict) -> None:
    d, cv, coefs, noise = ctx["d"], ctx["cv"], ctx["coefs"], ctx["noise"]
    st = d["wear_status_max"].value_counts().to_dict()
    L = ["# Peak-wear model: k from the most-worn location, COF = μ₀ + c·L^(−1/3), no day factor", "",
         f"`python -m ml.maxwear_model{' --quick' if ctx['quick'] else ''}` · {ctx['minutes']:.1f} min · "
         f"{len(d)} runs, leave-one-group-out CV (R04/R14/R20 = one group)", "",
         "## Wear rate from the most-worn location", "",
         f"- detected: {st.get('detected', 0)} runs; below LOD even at the most-worn location: "
         f"{st.get('below_LOD', 0)}; not measurable (no two-sided location): {st.get('not_measurable', 0)}.",
         f"- detected with the mean over locations (previous target): "
         f"{int(d['log10_kavg'].notna().sum())} runs.", "",
         "| run | L (N) | most-worn location | groove area (µm²) | LOD (µm²) | status | k_max | k mean (previous) |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        km = f"{r['k_max']:.2e}" if np.isfinite(r["k_max"]) else (
            f"< {r['k_max_LOD']:.1e}" if r["wear_status_max"] == "below_LOD" else "–")
        kv = pd.to_numeric(r["k_mm3_per_Nm"], errors="coerce")
        L.append(f"| {r['Experiment_ID']} | {r['Load_N']:g} | {r['max_location'] or '–'} | "
                 f"{r['max_groove_area_um2']:.0f} | {r['max_area_LOD_um2']:.0f} | {r['wear_status_max']} | {km} | "
                 f"{kv:.2e} |".replace("| nan |", "| – |"))
    L += ["", "Replicate spread (R04/R14/R20, SD over the replicates that have a value):", "",
          "| target | SD | n |", "|---|---|---|"]
    L += [f"| {t} | {noise[t][0]:.4g} | {noise[t][1]} |" for t in TARGETS]
    L += [""]
    for t, label in TARGETS.items():
        sub = cv[cv["target"] == t][["inputs", "model", "n_runs", "R2", "RMSE", "MAE", "tied_with_best"]].copy()
        sub.columns = ["inputs", "model", "runs", "R²", "RMSE", "MAE", "tied"]
        L += [f"## {label}", "", cm._tab(sub), ""]
    L += ["## Fitted relations (all runs, 95 % run-bootstrap intervals)", "",
          cm._tab(coefs[["model", "term", "estimate", "ci_low", "ci_high"]]), "",
          "For the geometric-mean rows, ci_low / ci_high are the smallest and largest run values.", "",
          "![COF](figures/cof_vs_load.png)", "", "![k](figures/k_max_vs_mean.png)", "",
          "![parity](figures/parity.png)", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Peak-wear model (most-worn location), COF = mu0 + c L^-1/3")
    ap.add_argument("--table", default=str(ROOT / "data" / "processed" / "runs_checkpoints.csv"))
    ap.add_argument("--locations", default=str(ROOT / "data" / "processed" / "wear_locations.csv"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "results" / "maxwear_model"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run(Path(a.table), Path(a.locations), Path(a.out), quick=a.quick, n_boot=a.n_boot, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
