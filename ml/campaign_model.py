"""Campaign model: the best-supported model for the October 2026 runs (no further experiments).

What it does differently from the DOE pipeline (``ml.train``), and why:

* **All 23 runs, one row per run.** No more runs can be made, so the 5 test runs join the
  training data. They were scored once, sealed, by ``ml.train --final``; that pre-registered result
  stays in ``ml/results/final*``. Performance here comes from leave-one-group-out CV over all runs
  (R04/R14/R20 form one group).
* **Test day as a blocking factor.** The runs were made on two days (CSV ``TestDate``), and the
  second day's COF is ~0.018 lower at every load. The day is read from the files, like the other
  inputs. Plate roughness (unworn Sa from the scans) is offered as a physical alternative: it
  separates the two days perfectly, so the data cannot tell the two explanations apart.
* **Small, physics-shaped candidates next to the ML models.** COF = mu0 + c * L^(-1/3) + day offset
  (Hertz contact area ~ L^(2/3) gives the adhesion term L^(-1/3)); track width w ~ L^b.

Candidates = input sets x models (linear, ridge, GPR, ANN ensemble). Outputs go to
``ml/results/campaign_model/``: report.md, cv_metrics.csv, cv_predictions.csv, coefficients.csv, figures/.

    python -m ml.campaign_model            # ~10-20 min on a laptop (ANN ensembles dominate)
    python -m ml.campaign_model --quick    # smaller ANN ensembles, ~1-2 min
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models.ann import ANNEnsemble

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("ml.campaign_model")

TARGETS = {"COF": "COF_steady (second half of the run)", "log10_w": "log10 track width [µm]"}
SHORT = {"COF": "COF", "log10_w": "Track width (log10 µm)"}
INPUT_SETS = {
    "COF": {
        "none (mean)": [],
        "DOE: ln L, ln f, ln t": ["ln_L", "ln_f", "ln_t"],
        "test day": ["day"],
        "ln L + day": ["ln_L", "day"],
        "L^-1/3 + day (physics)": ["L_m13", "day"],
        "L^-1/3 + plate roughness": ["L_m13", "ln_Sa"],
        "ln L, ln f, ln t + day": ["ln_L", "ln_f", "ln_t", "day"],
    },
    "log10_w": {
        "none (mean)": [],
        "DOE: ln L, ln f, ln t": ["ln_L", "ln_f", "ln_t"],
        "ln L only": ["ln_L"],
        "ln L + day (physics)": ["ln_L", "day"],
        "ln L, ln f, ln t + day": ["ln_L", "ln_f", "ln_t", "day"],
    },
}
PHYSICS = {"COF": "L^-1/3 + day (physics)", "log10_w": "ln L + day (physics)"}
LN10 = np.log(10.0)
PHYS_SPEC = {
    "COF": {
        "label": "COF = mu0 + c*L^(-1/3) + day offset",
        "x": lambda L: L ** (-1.0 / 3.0),
        "terms": [("mu0 (13 Oct intercept)", lambda c: c[0]),
                  ("c (L^-1/3 term)", lambda c: c[1]),
                  ("16 Oct offset", lambda c: c[2]),
                  ("COF change 10 N -> 100 N", lambda c: c[1] * (100 ** (-1 / 3) - 10 ** (-1 / 3)))],
        "ylabel": "COF (second half of run)", "ytrans": lambda v: v, "logy": False,
        "title": lambda c: "COF = μ₀ + c·L^(−1/3) + day offset (95 % band)",
        "figure": "cof_vs_load_by_day.png",
    },
    "log10_w": {
        "label": "w = A * L^b * (day factor)",
        "x": np.log,
        "terms": [("A: width at 1 N on 13 Oct (um)", lambda c: 10 ** c[0]),
                  ("b: load exponent", lambda c: c[1] * LN10),
                  ("16 Oct width factor", lambda c: 10 ** c[2])],
        "ylabel": "Track width (µm)", "ytrans": lambda v: 10 ** v, "logy": True, "yticks": [150, 200, 300, 500, 700],
        "title": lambda c: f"Track width ∝ L^{c[1] * LN10:.2f} with a day factor (95 % band)",
        "figure": "width_vs_load_by_day.png",
    },
}
MODELS = ["linear", "ridge", "gpr", "ann"]
COLORS = {0: "#2a78d6", 1: "#eb6834"}            # categorical slots 1-2 (dataviz reference palette)
MARKERS = {0: "o", 1: "s"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------- data
def load_runs(table: Path, locations: Path | None = None) -> pd.DataFrame:
    """One row per run (checkpoint 2) with inputs, targets, test day and replicate group."""
    r = pd.read_csv(table)
    b = r[(r["Checkpoint"] == 2) & (r["status"].astype(str) != "error")].copy()
    if "Test_date" not in b or b["Test_date"].isna().all():
        raise SystemExit(f"{table} has no Test_date: re-run `python -m tribo_extract extract`.")
    day = pd.to_datetime(b["Test_date"], errors="coerce").dt.normalize()
    b["day"] = (day != day.min()).astype(float)
    b["day_label"] = day.dt.strftime("%d %b")
    b["ln_L"], b["ln_f"], b["ln_t"] = np.log(b["Load_N"]), np.log(b["Freq_Hz"]), np.log(b["Run_time_s"])
    b["L_m13"] = b["Load_N"] ** (-1.0 / 3.0)
    b["COF"] = pd.to_numeric(b["COF_steady"], errors="coerce")
    b["log10_w"] = np.log10(pd.to_numeric(b["Track_width_um"], errors="coerce"))
    b["ln_Sa"] = np.nan
    if locations is not None and Path(locations).exists():
        w = pd.read_csv(locations)
        sa = w[w["Checkpoint"] == 2].groupby("Experiment_ID")["Sa_unworn_um"].median()
        b["ln_Sa"] = np.log(b["Experiment_ID"].map(sa))
    rep = b["Replicate_of"].fillna("").astype(str).str.strip() if "Replicate_of" in b else pd.Series("", index=b.index)
    b["group"] = np.where(rep.isin(["", "nan"]), b["Experiment_ID"].astype(str), rep)
    return b.sort_values("Experiment_ID").reset_index(drop=True)


# ----------------------------------------------------------------------------- models
class AnnCV:
    """ANN ensemble (inputs -> 3 tanh -> 1), weight decay chosen by inner GroupKFold."""

    def __init__(self, n_members=20, decays=(1e-3, 1e-2, 1e-1), seed=0, max_epochs=3000):
        self.n_members, self.decays, self.seed, self.max_epochs = n_members, decays, seed, max_epochs

    def _make(self, wd):
        return ANNEnsemble(hidden=3, weight_decay=wd, n_members=self.n_members, random_state=self.seed,
                           max_epochs=self.max_epochs)

    def fit(self, X, y, groups):
        wd = self.decays[0]
        if len(self.decays) > 1 and len(np.unique(groups)) >= 5:
            scores = []
            for d in self.decays:
                e = []
                for a, t in GroupKFold(5).split(X, y, groups):
                    e.append(self._make(d).fit(X[a], y[a], groups[a]).predict(X[t]) - y[t])
                scores.append(np.sqrt(np.mean(np.concatenate(e) ** 2)))
            wd = self.decays[int(np.argmin(scores))]
        self.weight_decay_ = wd
        self.est_ = self._make(wd).fit(X, y, groups)
        return self

    def predict(self, X):
        return self.est_.predict(X)


def make_model(name: str, n_inputs: int, quick: bool = False):
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if name == "gpr":
        k = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_inputs), (1e-2, 1e3), nu=2.5)
             + WhiteKernel(0.1, (1e-5, 10.0)))
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3,
                                                                        random_state=0))
    if name == "ann":
        return AnnCV(n_members=5 if quick else 20, decays=(1e-2,) if quick else (1e-3, 1e-2, 1e-1))
    raise ValueError(name)


def _fit(model, X, y, groups):
    if isinstance(model, AnnCV):
        return model.fit(X, y, groups)
    return model.fit(X, y)


def logo_predict(d: pd.DataFrame, target: str, feats: list[str], model_name: str, quick=False) -> np.ndarray:
    """Leave-one-group-out predictions for every run."""
    y = d[target].to_numpy(float)
    groups = d["group"].to_numpy(str)
    pred = np.empty(len(d))
    if not feats:
        for g in np.unique(groups):
            te = groups == g
            pred[te] = y[~te].mean()
        return pred
    X = d[feats].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in np.unique(groups):
            te = groups == g
            m = _fit(make_model(model_name, len(feats), quick), X[~te], y[~te], groups[~te])
            pred[te] = m.predict(X[te])
    return pred


def metrics(y, p) -> dict:
    e = p - y
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}


# ----------------------------------------------------------------------------- final fits
def ols(X, y):
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def group_bootstrap(d, feats, target, n_boot=2000, seed=0, grid=None):
    """Coefficients (and optional grid predictions) refitted on run-group bootstrap resamples."""
    rng = np.random.default_rng(seed)
    groups = d["group"].to_numpy(str)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    X, y = d[feats].to_numpy(float), d[target].to_numpy(float)
    coefs, preds = [], []
    for _ in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        A = np.c_[np.ones(len(take)), X[take]]
        if np.linalg.matrix_rank(A) < A.shape[1]:
            continue                                   # e.g. a resample with runs from one day only
        c = ols(X[take], y[take])
        coefs.append(c)
        if grid is not None:
            preds.append(np.c_[np.ones(len(grid)), grid] @ c)
    return np.array(coefs), (np.array(preds) if grid is not None else None)


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figure_by_day(d, ycol, curves, grid_L, ylabel, title, out, ytrans=lambda v: v, logy=False, yticks=None):
    """Points coloured by test day, one fitted curve + 95 % band per day."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    for k in (0, 1):
        s = d[d["day"] == k]
        fit, lo, hi = curves[k]
        ax.fill_between(grid_L, ytrans(lo), ytrans(hi), color=COLORS[k], alpha=0.14, lw=0)
        ax.plot(grid_L, ytrans(fit), color=COLORS[k], lw=2)
        ax.scatter(s["Load_N"], ytrans(s[ycol]), s=42, marker=MARKERS[k], color=COLORS[k], edgecolor=SURFACE,
                   lw=1.2, zorder=3, label=f"{s['day_label'].iloc[0]} ({len(s)} runs)")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    if logy:
        ax.set_yscale("log")
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks])
        ax.minorticks_off()
    ax.set_xlabel("Load (N)", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_parity(d, preds: dict, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(preds), figsize=(6.4 * len(preds) / 1.6, 3.8), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (title, (target, p, m)) in zip(axes, preds.items()):
        _style(ax)
        y = d[target].to_numpy(float)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, color=GRID, lw=1.2, zorder=1)
        for k in (0, 1):
            s = (d["day"] == k).to_numpy()
            ax.scatter(y[s], p[s], s=36, marker=MARKERS[k], color=COLORS[k], edgecolor=SURFACE, lw=1.0, zorder=3,
                       label=d.loc[s, "day_label"].iloc[0])
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("measured", color=INK2, fontsize=9)
        ax.set_ylabel("predicted (run left out)", color=INK2, fontsize=9)
        ax.set_title(f"{title}\nR² = {m['R2']:.2f}, RMSE = {m['RMSE']:.3g}", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(table: Path, locations: Path, out: Path, quick: bool = False, n_boot: int = 2000, n_jobs: int = -1) -> dict:
    t0 = time.time()
    d = load_runs(table, locations)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    jobs = []
    for target, sets in INPUT_SETS.items():
        for set_name, feats in sets.items():
            if any(d[f].isna().any() for f in feats):
                log.warning("skipping %s / %s: missing inputs", target, set_name)
                continue
            for m in (["mean"] if not feats else MODELS):
                jobs.append((target, set_name, feats, m))
    log.info("%d runs, %d groups; %d CV candidates", len(d), d["group"].nunique(), len(jobs))
    preds = Parallel(n_jobs=n_jobs)(delayed(logo_predict)(d, t, f, m, quick) for t, _, f, m in jobs)
    rows, prow = [], []
    for (t, s, f, m), p in zip(jobs, preds):
        y = d[t].to_numpy(float)
        res = d["day"].to_numpy()
        rows.append({"target": t, "inputs": s, "model": m, **metrics(y, p),
                     "bias_day1": float(np.mean((p - y)[res == 0])), "bias_day2": float(np.mean((p - y)[res == 1]))})
        prow.append(pd.DataFrame({"Experiment_ID": d["Experiment_ID"], "group": d["group"], "day": d["day_label"],
                                  "target": t, "inputs": s, "model": m, "y_true": y, "y_pred": p}))
    cv = pd.DataFrame(rows).sort_values(["target", "RMSE"]).reset_index(drop=True)
    cv.to_csv(out / "cv_metrics.csv", index=False)
    pd.concat(prow).to_csv(out / "cv_predictions.csv", index=False)

    # noise floors
    rep = d[d["group"] == "R04"] if (d["group"] == "R04").sum() > 1 else d.iloc[:0]
    noise = {}
    for t in TARGETS:
        same_day = rep.groupby("day")[t].std(ddof=1).dropna()
        noise[t] = {"all_replicates": float(rep[t].std(ddof=1)) if len(rep) > 1 else np.nan,
                    "same_day_replicates": float(same_day.mean()) if len(same_day) else np.nan}

    # which candidates are statistically tied with the lowest-error one (1-SE rule on per-run squared error)
    sq = {(t, si, m): (p - d[t].to_numpy(float)) ** 2 for (t, si, f, m), p in zip(jobs, preds)}
    cv["tied_with_best"] = ""
    for t in TARGETS:
        sub = cv[(cv["target"] == t) & (cv["model"] != "mean")]
        b = sub.iloc[0]
        e_best = sq[(t, b["inputs"], b["model"])]
        for i, r in sub.iterrows():
            diff = sq[(t, r["inputs"], r["model"])] - e_best
            cv.loc[i, "tied_with_best"] = "yes" if diff.mean() <= diff.std(ddof=1) / np.sqrt(len(diff)) else "no"
    cv.to_csv(out / "cv_metrics.csv", index=False)

    # physics-shaped models refitted on all runs, run-bootstrap intervals
    grid_L = np.geomspace(d["Load_N"].min() * 0.95, d["Load_N"].max() * 1.05, 120)
    coef_rows, figs, phys = [], {}, {}
    for t, spec in PHYS_SPEC.items():
        feats = INPUT_SETS[t][PHYSICS[t]]
        X, y = d[feats].to_numpy(float), d[t].to_numpy(float)
        c = ols(X, y)
        gx = spec["x"](grid_L)
        grid = np.r_[np.c_[gx, np.zeros_like(gx)], np.c_[gx, np.ones_like(gx)]]
        bc, bp = group_bootstrap(d, feats, t, n_boot, 7, grid)
        n = len(grid_L)
        fit = np.c_[np.ones(len(grid)), grid] @ c
        curves = {k: (fit[k * n:(k + 1) * n], *np.percentile(bp[:, k * n:(k + 1) * n], [2.5, 97.5], axis=0))
                  for k in (0, 1)}
        for (name, fn) in spec["terms"]:
            est, samples = fn(c), np.array([fn(row) for row in bc])
            lo, hi = np.percentile(samples, [2.5, 97.5])
            coef_rows.append({"target": t, "model": spec["label"], "term": name, "estimate": est, "ci_low": lo,
                              "ci_high": hi})
        phys[t] = {"coef": c, "n_boot_used": len(bc)}
        figs[t] = out / "figures" / spec["figure"]
        figure_by_day(d, t, curves, grid_L, spec["ylabel"], spec["title"](c), figs[t], spec["ytrans"],
                      spec["logy"], spec.get("yticks"))
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(out / "coefficients.csv", index=False)

    best = cv[cv["model"] != "mean"].groupby("target").head(1).set_index("target")
    rec = {}
    for t in TARGETS:
        r = cv[(cv["target"] == t) & (cv["inputs"] == PHYSICS[t]) & (cv["model"] == "linear")].iloc[0]
        rec[t] = r
    par = {}
    for t in TARGETS:
        p = next(p for (tt, s_, f, m), p in zip(jobs, preds) if tt == t and s_ == PHYSICS[t] and m == "linear")
        par[f"{SHORT[t]}: {PHYSICS[t]}, linear"] = (t, p, metrics(d[t].to_numpy(float), p))
    figs["parity"] = out / "figures" / "parity_physics_loo.png"
    figure_parity(d, par, figs["parity"])

    ctx = {"d": d, "cv": cv, "best": best, "rec": rec, "coefs": coefs, "noise": noise, "figs": figs, "quick": quick,
           "minutes": (time.time() - t0) / 60}
    write_report(out, ctx)
    log.info("done in %.1f min -> %s", ctx["minutes"], out / "report.md")
    return ctx


def _fmt(col: str, v) -> str:
    if not isinstance(v, (float, np.floating)):
        return str(v)
    if not np.isfinite(v):
        return "–"
    if col == "R²":
        return f"{v:+.2f}"
    if col.startswith("bias"):
        return f"{v:+.4f}".replace("-0.0000", "+0.0000")
    return f"{v:.4g}"


def _tab(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(c, v) for c, v in zip(cols, r)) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def write_report(out: Path, ctx: dict) -> None:
    d, cv, best, rec, coefs, noise = ctx["d"], ctx["cv"], ctx["best"], ctx["rec"], ctx["coefs"], ctx["noise"]
    days = d.groupby("day_label")["Experiment_ID"].apply(lambda s: ", ".join(s)).to_dict()
    L = ["# Campaign model: all 23 runs, test day as a blocking factor", "",
         f"`python -m ml.campaign_model{' --quick' if ctx['quick'] else ''}` · {ctx['minutes']:.1f} min · "
         "leave-one-group-out CV (R04/R14/R20 = one group)", "",
         "## Data", "",
         f"- {len(d)} runs in {d['group'].nunique()} groups, one row per run.",
         "- The 5 test runs are included because no further runs can be made. Their single sealed score is the "
         "pre-registered result in `ml/results/final/` and `ml/results/final_width/`.",
         *[f"- {k}: {v}" for k, v in days.items()],
         "- COF target: COF_steady over the second half of each run. Wear target: log10 track width "
         "(the groove volume is below its detection limit in 9 runs and not measurable in 2).",
         "- `tied` = within one standard error (per-run squared error) of the lowest-error candidate. Among tied "
         "candidates the physics-shaped linear model is recommended, because it has the fewest parameters and a "
         "physical meaning.", ""]
    for t, label in TARGETS.items():
        sub = cv[cv["target"] == t][["inputs", "model", "R2", "RMSE", "MAE", "bias_day1", "bias_day2",
                                     "tied_with_best"]].copy()
        sub.columns = ["inputs", "model", "R²", "RMSE", "MAE", "bias 13 Oct", "bias 16 Oct", "tied"]
        b, r, nf = best.loc[t], rec[t], noise[t]
        L += [f"## {label}", "",
              f"- Lowest CV error: {b['model']} on {b['inputs']} (R² {b['R2']:+.2f}, RMSE {b['RMSE']:.4g}).",
              f"- Recommended: linear on {PHYSICS[t]} (R² {r['R2']:+.2f}, RMSE {r['RMSE']:.4g}; tied with the "
              f"lowest: {r['tied_with_best']}).",
              f"- Replicate noise: {nf['all_replicates']:.4g} across R04/R14/R20 (two days), "
              f"{nf['same_day_replicates']:.4g} between the same-day replicates R14 and R20.",
              "- Bias = mean (predicted − measured) per day.", "", _tab(sub), ""]
    L += ["## Physics-shaped models (all runs, 95 % run-bootstrap intervals)", ""]
    for t in TARGETS:
        c = coefs[coefs["target"] == t]
        L += [f"**{c['model'].iloc[0]}**", "", _tab(c[["term", "estimate", "ci_low", "ci_high"]]), ""]
    L += ["A pure adhesion law μ ∝ L^(−1/3) would halve COF from 10 N to 100 N. Elastic (Hertz) contact predicts a "
          "track-width exponent b = 1/3.", "",
          "![COF vs load](figures/cof_vs_load_by_day.png)", "",
          "![width vs load](figures/width_vs_load_by_day.png)", "",
          "![parity](figures/parity_physics_loo.png)", "",
          "## How to read this", "",
          "- The day offset is a nuisance correction. It cannot predict a run on a new day; for that, the "
          "day-to-day difference is part of the uncertainty.",
          "- Plate roughness (`L^-1/3 + plate roughness`) separates the two days perfectly, so it is an equally "
          "good but not independent explanation.",
          "- For track width, part of the 16 Oct factor is probably measurement: on the rougher 16 Oct plates the "
          "width finder can under-read faint tracks.",
          "- Choosing the best of many candidates by the same CV makes its score slightly optimistic.",
          "- Frequency and time are left out of the physics models because adding them makes the CV worse.", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Campaign model: all runs, test day as blocking factor")
    ap.add_argument("--table", default=str(ROOT / "data" / "processed" / "runs_checkpoints.csv"))
    ap.add_argument("--locations", default=str(ROOT / "data" / "processed" / "wear_locations.csv"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "results" / "campaign_model"))
    ap.add_argument("--quick", action="store_true", help="5-member ANN ensembles, fixed weight decay")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run(Path(a.table), Path(a.locations), Path(a.out), quick=a.quick, n_boot=a.n_boot, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
