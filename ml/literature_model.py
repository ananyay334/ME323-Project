"""Literature pipeline: does adding Ponnuswamy & Manoharan (2024) improve the model of OUR runs?

Paper: SS304 pin-on-disk, dry, 10/15/20 N, 1/2/3 m/s, 60/120 mm track, 200 s (Table 5, 18 runs; file
``data/external/ponnuswamy2024_table5.csv``). Its COF is ~30x below ours and its wear volumes are
implausibly large, so units may differ; every strategy except "naive pooling" works in log space with a
study offset, which makes a constant unit factor irrelevant.

Common inputs: ln L (load, N), ln v (mean sliding speed, m/s; ours v = 4*A*f with A = 5 mm), ln t (s).
Paper-only covariate: ln(track diameter), centred, 0 for our runs. No test-day factor anywhere.
Targets: COF, and log10 of the specific wear rate (ours: k_max from the most-worn scan location, as in
``ml.maxwear_model``; paper: wear volume / (load * speed * time)).

Evaluation is only on our runs: leave-one-group-out over our run groups (R04/R14/R20 together). The 18
paper rows are always in the training data. Strategies:

  ours only      constant | COF = mu0 + c L^-1/3 | power law in L, v, t | GPR | ANN ensemble
  naive pooling  the same power law / GPR / ANN on ours + paper, no study term
  study offset   power law with a separate intercept per study (slopes shared)
  paper prior    our power law with its L and v slopes shrunk toward the paper's slopes; strength tuned by
                 inner CV on our runs (no shrinkage ... fixed at the paper's slopes)
  paper slopes   our power law with the paper's L and v slopes imposed (only intercept and t fitted), or
                 with only the paper's load slope imposed (load is the factor whose ranges overlap)
  study flag     GPR / ANN ensemble on ours + paper with a 0/1 study input

    python -m ml.literature_model            # ~5-10 min
    python -m ml.literature_model --quick
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
from sklearn.model_selection import GroupKFold

from . import campaign_model as cm
from .maxwear_model import peak_wear_table

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "data" / "external" / "ponnuswamy2024_table5.csv"
log = logging.getLogger("ml.literature_model")
AMP_M = 0.005                                    # stroke amplitude A (m): mean speed = 4 A f
FEATS = ["lnL", "lnv", "lnt"]
LAMBDAS = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
TARGETS = {"COF": "COF", "k": "log10 specific wear rate (ours: most-worn location)"}
INK, INK2, GRID, SURFACE = cm.INK, cm.INK2, cm.GRID, cm.SURFACE
SRC_COL = {"ours": cm.COLORS[0], "paper": cm.COLORS[1]}


# ----------------------------------------------------------------------------- data
def load_paper(path: Path = PAPER) -> pd.DataFrame:
    p = pd.read_csv(path)
    p["source"] = "paper"
    p["Experiment_ID"] = p["Run"]
    p["Distance_m"] = p["Speed_m_s"] * p["Time_s"]
    p["k"] = p["Wear_volume_mm3"] / (p["Load_N"] * p["Distance_m"])
    p["group"] = "P_L" + p["Load_N"].astype(str) + "_v" + p["Speed_m_s"].astype(str)   # 2 diameters = 1 group
    p["lnD"] = np.log(p["Track_diameter_mm"] / np.sqrt(60 * 120))
    p["Run_time_s"] = p["Time_s"]
    return p


def load_ours(table: Path, locations: Path) -> pd.DataFrame:
    o = peak_wear_table(cm.load_runs(table, locations), pd.read_csv(locations))
    o["source"] = "ours"
    o["Speed_m_s"] = 4 * AMP_M * o["Freq_Hz"]
    o["k"] = o["k_max"]
    o["lnD"] = 0.0
    return o


def combine(ours: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    cols = ["Experiment_ID", "source", "group", "Load_N", "Speed_m_s", "Run_time_s", "Distance_m", "COF", "k", "lnD"]
    d = pd.concat([ours[cols], paper[cols]], ignore_index=True)
    d["lnL"], d["lnv"], d["lnt"] = np.log(d["Load_N"]), np.log(d["Speed_m_s"]), np.log(d["Run_time_s"])
    d["src"] = (d["source"] == "paper").astype(float)
    d["L_m13"] = d["Load_N"] ** (-1 / 3)
    d["y_COF"] = np.log(d["COF"])                 # modelled in log space, scored on the COF scale
    d["y_k"] = np.log10(d["k"])
    return d


# ----------------------------------------------------------------------------- models
def _ols(X, y):
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _pred(c, X):
    """Intercept c[0] plus slopes c[1:]."""
    return c[0] + np.asarray(X, float) @ c[1:]


def paper_slopes(tr: pd.DataFrame, ycol: str) -> np.ndarray:
    """Paper-only OLS: y ~ lnL + lnv + lnD -> slopes for (lnL, lnv)."""
    p = tr[tr["source"] == "paper"]
    c = _ols(p[["lnL", "lnv", "lnD"]].to_numpy(float), p[ycol].to_numpy(float))
    return c[1:3]


def prior_fit(o: pd.DataFrame, ycol: str, b0: np.ndarray, lam: float) -> np.ndarray:
    """Our power law (intercept, lnL, lnv, lnt) with (lnL, lnv) slopes shrunk toward b0 by lam (per run)."""
    X = np.c_[np.ones(len(o)), o[FEATS].to_numpy(float)]
    y = o[ycol].to_numpy(float)
    if np.isinf(lam):                              # slopes fixed at the paper's; fit intercept and t only
        r = y - X[:, 1:3] @ b0
        c = np.linalg.lstsq(X[:, [0, 3]], r, rcond=None)[0]
        return np.array([c[0], b0[0], b0[1], c[1]])
    P = np.diag([0.0, 1.0, 1.0, 0.0]) * lam * len(o)
    prior = np.r_[0.0, b0, 0.0]
    return np.linalg.solve(X.T @ X + P, X.T @ y + P @ prior)


def prior_tuned(o_tr: pd.DataFrame, ycol: str, b0: np.ndarray) -> tuple[np.ndarray, float]:
    groups = o_tr["group"].to_numpy(str)
    best, best_s = 0.0, np.inf
    if len(np.unique(groups)) >= 5:
        for lam in LAMBDAS:
            e = []
            for a, b in GroupKFold(5).split(o_tr, groups=groups):
                c = prior_fit(o_tr.iloc[a], ycol, b0, lam)
                e.append(_pred(c, o_tr.iloc[b][FEATS].to_numpy(float)) - o_tr.iloc[b][ycol].to_numpy(float))
            s = np.sqrt(np.mean(np.concatenate(e) ** 2))
            if s < best_s - 1e-12:
                best, best_s = lam, s
    return prior_fit(o_tr, ycol, b0, best), best


def fit_predict(strategy: str, tr: pd.DataFrame, te: pd.DataFrame, target: str, quick: bool):
    """Prediction on the target's scoring scale (COF, or log10 k) for the held-out rows of our data."""
    ycol = f"y_{target}"
    o_tr = tr[tr["source"] == "ours"]
    back = np.exp if target == "COF" else (lambda v: v)
    info = ""
    if strategy == "ours: constant":
        return np.full(len(te), o_tr[target if target == "COF" else ycol].mean()), info
    if strategy == "ours: mu0 + c*L^-1/3":
        if target != "COF":
            return None, info
        c = _ols(o_tr[["L_m13"]].to_numpy(float), o_tr["COF"].to_numpy(float))
        return _pred(c, te[["L_m13"]].to_numpy(float)), info
    if strategy == "ours: power law":
        c = _ols(o_tr[FEATS].to_numpy(float), o_tr[ycol].to_numpy(float))
        return back(_pred(c, te[FEATS].to_numpy(float))), info
    if strategy == "naive pooling: power law":
        c = _ols(tr[FEATS].to_numpy(float), tr[ycol].to_numpy(float))
        return back(_pred(c, te[FEATS].to_numpy(float))), info
    if strategy == "study offset: power law":
        f = FEATS + ["src", "lnD"]
        c = _ols(tr[f].to_numpy(float), tr[ycol].to_numpy(float))
        return back(_pred(c, te[f].to_numpy(float))), info
    if strategy == "paper slopes fixed: power law":
        b0 = paper_slopes(tr, ycol)
        c = prior_fit(o_tr, ycol, b0, np.inf)
        return back(_pred(c, te[FEATS].to_numpy(float))), ""
    if strategy == "paper load slope fixed: power law":
        bL = paper_slopes(tr, ycol)[0]
        r = o_tr[ycol].to_numpy(float) - bL * o_tr["lnL"].to_numpy(float)
        c = _ols(o_tr[["lnv", "lnt"]].to_numpy(float), r)
        p = c[0] + bL * te["lnL"].to_numpy(float) + te[["lnv", "lnt"]].to_numpy(float) @ c[1:]
        return back(p), ""
    if strategy == "paper prior: power law":
        b0 = paper_slopes(tr, ycol)
        c, lam = prior_tuned(o_tr.reset_index(drop=True), ycol, b0)
        return back(_pred(c, te[FEATS].to_numpy(float))), f"lambda={lam:g}"
    model, rows, feats = None, None, None
    if strategy.startswith("ours: "):
        rows, feats = o_tr, FEATS
    elif strategy.startswith("naive pooling: "):
        rows, feats = tr, FEATS
    elif strategy.startswith("study flag: "):
        rows, feats = tr, FEATS + ["src", "lnD"]
    name = strategy.split(": ")[1].lower()
    model = cm.make_model({"gpr": "gpr", "ann": "ann"}[name], len(feats), quick)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm._fit(model, rows[feats].to_numpy(float), rows[ycol].to_numpy(float), rows["group"].to_numpy(str))
        p = model.predict(te[feats].to_numpy(float))
    return back(np.ravel(p)), info


STRATEGIES = ["ours: constant", "ours: mu0 + c*L^-1/3", "ours: power law", "ours: GPR", "ours: ANN",
              "naive pooling: power law", "naive pooling: GPR", "naive pooling: ANN",
              "study offset: power law", "paper prior: power law", "paper slopes fixed: power law",
              "paper load slope fixed: power law", "study flag: GPR", "study flag: ANN"]


def logo(d: pd.DataFrame, strategy: str, target: str, quick: bool):
    """Leave-one-group-out over OUR runs that have the target; paper rows always train."""
    ycol = f"y_{target}"
    dd = d[np.isfinite(d[ycol])].reset_index(drop=True)
    ours = dd[dd["source"] == "ours"]
    preds, infos = pd.Series(np.nan, index=ours.index), []
    for g in ours["group"].unique():
        te_idx = ours.index[ours["group"] == g]
        tr = dd.drop(index=te_idx)
        p, info = fit_predict(strategy, tr, dd.loc[te_idx], target, quick)
        if p is None:
            return None
        preds.loc[te_idx] = p
        infos.append(info)
    y = ours[target if target == "COF" else ycol].to_numpy(float)
    return {"strategy": strategy, "target": target, "ids": ours["Experiment_ID"].to_numpy(), "y": y,
            "p": preds.to_numpy(float), "info": ";".join(sorted(set(i for i in infos if i)))}


# ----------------------------------------------------------------------------- trends
def trend_table(d: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for target, ycol in (("COF", "y_COF"), ("k", "y_k")):
        for src, feats in (("paper", ["lnL", "lnv", "lnD"]), ("ours", FEATS)):
            s = d[(d["source"] == src) & np.isfinite(d[ycol])].reset_index(drop=True)
            c = _ols(s[feats].to_numpy(float), s[ycol].to_numpy(float))
            bs = cm.group_bootstrap(s, feats, ycol, n_boot, 5)[0]
            scale = 1.0 if target == "COF" else cm.LN10          # log10 k per ln x -> power-law exponent
            for j, f in enumerate(feats, start=1):
                if f == "lnD":
                    continue
                v = {"lnL": "load", "lnv": "sliding speed", "lnt": "time"}[f]
                rows.append({"target": target, "source": src, "n": len(s), "exponent of": v,
                             "estimate": c[j] * scale, "ci_low": np.percentile(bs[:, j] * scale, 2.5),
                             "ci_high": np.percentile(bs[:, j] * scale, 97.5)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    return plt


def fig_trends(d, out):
    """Each study's values relative to its own geometric mean, vs load and vs speed (log-log)."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), facecolor=SURFACE)
    for i, (target, ycol, lab) in enumerate((("COF", "y_COF", "COF / study mean"),
                                             ("k", "y_k", "wear rate / study mean"))):
        for j, (x, xl) in enumerate((("Load_N", "Load (N)"), ("Speed_m_s", "Mean sliding speed (m/s)"))):
            ax = axes[i, j]
            cm._style(ax)
            for src in ("ours", "paper"):
                s = d[(d["source"] == src) & np.isfinite(d[ycol])]
                yy = s[ycol] - s[ycol].mean()
                rel = np.exp(yy) if target == "COF" else 10 ** yy
                ax.scatter(s[x], rel, s=30, marker="o" if src == "ours" else "s", color=SRC_COL[src],
                           edgecolor=SURFACE, lw=0.9, zorder=3,
                           label=f"{'our runs' if src == 'ours' else 'Ponnuswamy 2024'} (n={len(s)})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.minorticks_off()
            if x == "Load_N":
                ax.set_xticks([10, 20, 50, 100])
                ax.set_xticklabels(["10", "20", "50", "100"])
            else:
                ax.set_xticks([0.1, 0.3, 1, 3])
                ax.set_xticklabels(["0.1", "0.3", "1", "3"])
            yt = [0.5, 0.7, 1, 1.5, 2] if target == "COF" else [0.2, 0.5, 1, 2, 5]
            ax.set_yticks(yt)
            ax.set_yticklabels([f"{v:g}" for v in yt])
            ax.set_xlabel(xl, color=INK2)
            ax.set_ylabel(lab, color=INK2)
            ax.set_title(f"{'COF' if target == 'COF' else 'Specific wear rate'} vs {xl.split(' (')[0].lower()}",
                         loc="left", color=INK)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_scores(cv, out):
    """Dot plot of the RMSE on our runs per strategy; dashed line = constant (mean) prediction."""
    plt = _plt()
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4), facecolor=SURFACE)
    for ax, target in zip(axes, TARGETS):
        cm._style(ax)
        s = cv[cv["target"] == target].sort_values("RMSE", ascending=False)
        base = s[s["strategy"] == "ours: constant"]["RMSE"].iloc[0]
        for i, (_, r) in enumerate(s.iterrows()):
            src = "ours" if r["strategy"].startswith("ours") else "paper"
            ax.plot([base, r["RMSE"]], [i, i], color=GRID, lw=1.2, zorder=1)
            ax.scatter(r["RMSE"], i, s=36, color=SRC_COL[src], edgecolor=SURFACE, lw=0.8, zorder=3,
                       marker="o" if src == "ours" else "s")
        ax.axvline(base, color=INK2, lw=1, ls="--")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s["strategy"], fontsize=7.5)
        if target == "COF":                         # COF errors span a decade; wear-rate errors do not
            ax.set_xscale("log")
            ax.set_xticks([0.01, 0.02, 0.05, 0.1])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel("RMSE of COF on our runs (log axis)" if target == "COF" else "RMSE of log10 k on our runs",
                      color=INK2)
        ax.set_title("COF" if target == "COF" else "Specific wear rate", loc="left", color=INK)
        ax.grid(False, axis="y")
    fig.text(0.01, 0.01, "circles = our data only, squares = with the paper's data; dashed = predicting the mean",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(table: Path, locations: Path, out: Path, quick=False, n_jobs=-1, paper_path: Path = PAPER) -> dict:
    t0 = time.time()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d = combine(load_ours(table, locations), load_paper(paper_path))
    d.to_csv(out / "combined_data.csv", index=False)
    jobs = [(s, t) for t in TARGETS for s in STRATEGIES]
    res = Parallel(n_jobs=n_jobs)(delayed(logo)(d, s, t, quick) for s, t in jobs)
    res = [r for r in res if r is not None]
    rows, preds = [], []
    for r in res:
        m = cm.metrics(r["y"], r["p"])
        rows.append({"target": r["target"], "strategy": r["strategy"], "n_our_runs": len(r["y"]), **m,
                     "notes": r["info"]})
        preds.append(pd.DataFrame({"Experiment_ID": r["ids"], "target": r["target"], "strategy": r["strategy"],
                                   "y_true": r["y"], "y_pred": r["p"]}))
    cv = pd.DataFrame(rows).sort_values(["target", "RMSE"]).reset_index(drop=True)
    # paired comparison with the best "ours only" strategy (1-SE rule on per-run squared error)
    P = {(r["target"], r["strategy"]): r for r in res}
    cv["vs best ours-only"] = ""
    for t in TARGETS:
        ours_best = cv[(cv["target"] == t) & cv["strategy"].str.startswith("ours")].iloc[0]
        rb = P[(t, ours_best["strategy"])]
        eb = (rb["p"] - rb["y"]) ** 2
        for i, r in cv[cv["target"] == t].iterrows():
            if r["strategy"] == ours_best["strategy"]:
                cv.loc[i, "vs best ours-only"] = "reference"
                continue
            rr = P[(t, r["strategy"])]
            diff = (rr["p"] - rr["y"]) ** 2 - eb
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            cv.loc[i, "vs best ours-only"] = ("better" if diff.mean() < -se else "worse" if diff.mean() > se
                                              else "tied")
    cv.to_csv(out / "cv_metrics.csv", index=False)
    pd.concat(preds).to_csv(out / "cv_predictions.csv", index=False)
    trends = trend_table(d)
    trends.to_csv(out / "trend_comparison.csv", index=False)
    fig_trends(d, out / "figures" / "trends_ours_vs_paper.png")
    fig_scores(cv, out / "figures" / "cv_rmse.png")
    ctx = {"d": d, "cv": cv, "trends": trends, "quick": quick, "minutes": (time.time() - t0) / 60}
    write_report(out, ctx)
    log.info("done in %.1f min -> %s", ctx["minutes"], out / "report.md")
    return ctx


def write_report(out: Path, ctx: dict) -> None:
    d, cv, tr = ctx["d"], ctx["cv"], ctx["trends"]
    o, p = d[d["source"] == "ours"], d[d["source"] == "paper"]
    L = ["# Literature pipeline: our runs + Ponnuswamy & Manoharan (2024)", "",
         f"`python -m ml.literature_model{' --quick' if ctx['quick'] else ''}` · {ctx['minutes']:.1f} min · "
         "scored on our runs only, leave-one-group-out; the 18 paper rows are always in training; no day factor", "",
         "## The two data sets", "",
         "| | our runs | paper |", "|---|---|---|",
         "| test | reciprocating ball-on-flat, 52100 ball on SS304 | unidirectional pin-on-disk, SS304 on SS304 |",
         f"| runs | {len(o)} (wear rate: {int(np.isfinite(o['y_k']).sum())}) | {len(p)} |",
         f"| load (N) | {o['Load_N'].min():g}–{o['Load_N'].max():g} | {p['Load_N'].min():g}–{p['Load_N'].max():g} |",
         f"| mean sliding speed (m/s) | {o['Speed_m_s'].min():.2f}–{o['Speed_m_s'].max():.2f} | "
         f"{p['Speed_m_s'].min():g}–{p['Speed_m_s'].max():g} |",
         f"| sliding distance (m) | {o['Distance_m'].min():.1f}–{o['Distance_m'].max():.1f} | "
         f"{p['Distance_m'].min():g}–{p['Distance_m'].max():g} |",
         f"| COF | {o['COF'].min():.3f}–{o['COF'].max():.3f} | {p['COF'].min():.4f}–{p['COF'].max():.4f} |",
         f"| wear rate k (mm³/N·m) | {np.nanmin(o['k']):.2g}–{np.nanmax(o['k']):.2g} | "
         f"{p['k'].min():.2g}–{p['k'].max():.2g} (as published) |", "",
         "## Do the trends agree? (power-law exponents, 95 % bootstrap intervals)", "",
         cm._tab(tr[["target", "source", "n", "exponent of", "estimate", "ci_low", "ci_high"]]), "",
         "COF exponents are d ln COF / d ln x; wear-rate exponents are d ln k / d ln x. The paper's fit also has a "
         "track-diameter term (not shown).", ""]
    for t, lab in TARGETS.items():
        sub = cv[cv["target"] == t][["strategy", "n_our_runs", "R2", "RMSE", "MAE", "vs best ours-only", "notes"]]
        sub = sub.rename(columns={"n_our_runs": "our runs", "R2": "R²"})
        L += [f"## {lab}: scores on our runs", "", cm._tab(sub), ""]
    L += ["`vs best ours-only` compares per-run squared errors with the best strategy that uses only our data "
          "(better / worse = more than one standard error apart).", "",
          "![trends](figures/trends_ours_vs_paper.png)", "", "![scores](figures/cv_rmse.png)", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Our runs + Ponnuswamy & Manoharan (2024): does it help?")
    ap.add_argument("--table", default=str(ROOT / "data" / "processed" / "runs_checkpoints.csv"))
    ap.add_argument("--locations", default=str(ROOT / "data" / "processed" / "wear_locations.csv"))
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--out", default=str(ROOT / "ml" / "results" / "literature_model"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run(Path(a.table), Path(a.locations), Path(a.out), quick=a.quick, n_jobs=a.jobs, paper_path=Path(a.paper))


if __name__ == "__main__":
    main()
