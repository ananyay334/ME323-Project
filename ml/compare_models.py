"""Side-by-side metrics: DOE pipeline model (old) vs campaign model (new) -> PDF.

Old = what ``ml.train`` selected: inputs ln L, ln f, ln t; COF -> ANN ensemble (default config),
track width -> XGBoost (width config); tuned on the same grids by inner GroupKFold.
New = ``ml.campaign_model`` recommendation: COF -> linear on L^-1/3 + test day;
track width -> linear on ln L + test day.

Protocols:
  A  as reported by each pipeline (old: 5x5 grouped CV on 18 runs + sealed test; new: LOGO on 23 runs);
  B  head-to-head, identical protocol: leave-one-group-out over all 23 runs, one row per run;
  C  head-to-head on the 5 sealed test runs (old: its sealed predictions; new: fitted on the 18
     training runs, scored after the test labels had been seen).

    python -m ml.compare_models      # -> ml/results/model_comparison/model_comparison.pdf
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

from . import campaign_model as cm
from .config import load_config
from .models.registry import get_spec
from .tuning import tune_and_fit

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "ml" / "results"
OUT = RES / "model_comparison"
INK, INK2, GRID, SURFACE = cm.INK, cm.INK2, cm.GRID, cm.SURFACE
DAYC = cm.COLORS
OLD_SPEC = {"COF": ("ann", ROOT / "ml" / "configs" / "default.yaml", "final", "COF_steady"),
            "log10_w": ("xgb", ROOT / "ml" / "configs" / "width.yaml", "final_width", "log10_w")}
NEW_FEATS = {t: cm.INPUT_SETS[t][cm.PHYSICS[t]] for t in cm.TARGETS}
OLD_FEATS = ["ln_L", "ln_f", "ln_t"]
LABEL = {"COF": "COF (second half of run)", "log10_w": "Track width (log10 µm)"}


# ----------------------------------------------------------------------------- predictions
def old_logo(d: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    """LOGO predictions of the old model (same spec and grid as ml.train, inner GroupKFold(4))."""
    name, cfg_path, _, _ = OLD_SPEC[target]
    cfg = load_config(cfg_path)
    spec, mcfg = get_spec(name), cfg["models"][name]
    X = d[OLD_FEATS].rename(columns={"ln_L": "ln_L", "ln_f": "ln_f", "ln_t": "ln_t"}).reset_index(drop=True)
    y = d[target].to_numpy(float)
    groups = d["group"].to_numpy(str)
    mu, sd = np.empty(len(d)), np.full(len(d), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in np.unique(groups):
            te = groups == g
            tr = np.flatnonzero(~te)
            inner = [(tr_[a], tr_[b]) for tr_ in [np.arange(len(tr))]
                     for a, b in GroupKFold(4).split(tr, groups=groups[tr])]
            est, _, _ = tune_and_fit(spec, mcfg, X.iloc[tr].reset_index(drop=True), y[tr], inner, 323, {},
                                     groups=groups[tr])
            mu[te] = np.ravel(est.predict(X[te]))
            if hasattr(est, "predict_std"):
                sd[te] = np.ravel(est.predict_std(X[te]))
    return mu, sd


def ols_fit_predict(Xtr, ytr, Xte):
    """OLS mean and 95 % prediction interval half-width."""
    A = np.c_[np.ones(len(Xtr)), Xtr]
    B = np.c_[np.ones(len(Xte)), Xte]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    dof = len(ytr) - A.shape[1]
    s2 = np.sum((ytr - A @ coef) ** 2) / dof
    cov = np.linalg.pinv(A.T @ A)
    se = np.sqrt(s2 * (1 + np.einsum("ij,jk,ik->i", B, cov, B)))
    return B @ coef, stats.t.ppf(0.975, dof) * se


def new_logo(d: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    X, y, groups = d[NEW_FEATS[target]].to_numpy(float), d[target].to_numpy(float), d["group"].to_numpy(str)
    mu, half = np.empty(len(d)), np.empty(len(d))
    for g in np.unique(groups):
        te = groups == g
        mu[te], half[te] = ols_fit_predict(X[~te], y[~te], X[te])
    return mu, half


# ----------------------------------------------------------------------------- metrics
def full_metrics(y, p, day, tol, half=None, cuts=None) -> dict:
    e = p - y
    m = {"n runs": len(y), "R²": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
         "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "max |error|": np.max(np.abs(e)),
         "bias (mean error)": e.mean(), "bias 13 Oct": e[day == 0].mean() if (day == 0).any() else np.nan,
         "bias 16 Oct": e[day == 1].mean() if (day == 1).any() else np.nan,
         "Pearson r": stats.pearsonr(y, p)[0] if np.std(p) > 0 else np.nan,
         "Spearman ρ": stats.spearmanr(y, p)[0] if np.std(p) > 0 else np.nan,
         "calibration slope": np.polyfit(p, y, 1)[0] if np.std(p) > 0 else np.nan,
         "within ±1 noise SD": np.mean(np.abs(e) <= tol), "within ±2 noise SD": np.mean(np.abs(e) <= 2 * tol)}
    if half is not None and np.isfinite(half).all():
        m["95 % interval coverage"] = np.mean(np.abs(e) <= half)
        m["95 % interval mean width"] = np.mean(2 * half)
    else:
        m["95 % interval coverage"] = np.nan
        m["95 % interval mean width"] = np.nan
    if cuts is not None:
        a, b = np.digitize(y, cuts), np.digitize(p, cuts)
        m["tercile accuracy"] = np.mean(a == b)
        m["tercile κ (Cohen)"] = cohen_kappa_score(a, b)
        m["_cm"] = pd.crosstab(pd.Categorical(a, [0, 1, 2]), pd.Categorical(b, [0, 1, 2]), dropna=False).to_numpy()
    return m


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    return plt


def fig_parity(d, P, out):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), facecolor=SURFACE)
    for i, t in enumerate(cm.TARGETS):
        y = d[t].to_numpy(float)
        lim = [min(y.min(), P[t]["old"].min(), P[t]["new"].min()), max(y.max(), P[t]["old"].max(), P[t]["new"].max())]
        pad = 0.06 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        for j, (k, name) in enumerate([("old", "Old (DOE pipeline)"), ("new", "New (campaign model)")]):
            ax = axes[i, j]
            cm._style(ax)
            ax.plot(lim, lim, color=GRID, lw=1.2)
            for dy in (0, 1):
                s = (d["day"] == dy).to_numpy()
                ax.scatter(y[s], P[t][k][s], s=34, marker=cm.MARKERS[dy], color=DAYC[dy], edgecolor=SURFACE,
                           lw=1.0, zorder=3, label=d.loc[s, "day_label"].iloc[0])
            e = P[t][k] - y
            r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
            short = "COF" if t == "COF" else "Track width (log10 µm)"
            ax.set_title(f"{name.split(' (')[0]} — {short}\nR² = {r2:+.2f}, RMSE = {np.sqrt(np.mean(e ** 2)):.3g}",
                         loc="left", color=INK)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("measured", color=INK2)
            ax.set_ylabel("predicted (run left out)", color=INK2)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_residuals(d, P, out):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for i, t in enumerate(cm.TARGETS):
        y = d[t].to_numpy(float)
        ax = axes[i, 0]
        cm._style(ax)
        ax.axhline(0, color=INK2, lw=0.8)
        for j, k in enumerate(["old", "new"]):
            for dy in (0, 1):
                s = (d["day"] == dy).to_numpy()
                e = (P[t][k] - y)[s]
                x = j * 2.6 + dy + rng.uniform(-0.15, 0.15, s.sum())
                if k == "new":
                    ax.scatter(x, e, s=26, marker=cm.MARKERS[dy], facecolor=DAYC[dy], edgecolor=SURFACE, lw=0.8,
                               zorder=3)
                else:
                    ax.scatter(x, e, s=26, marker=cm.MARKERS[dy], facecolor="none", edgecolor=DAYC[dy], lw=1.1,
                               zorder=3)
                ax.plot([j * 2.6 + dy - 0.3, j * 2.6 + dy + 0.3], [e.mean()] * 2, color=INK, lw=2)
        ax.set_xticks([0, 1, 2.6, 3.6])
        ax.set_xticklabels(["old\n13 Oct", "old\n16 Oct", "new\n13 Oct", "new\n16 Oct"])
        ax.set_ylabel("predicted − measured", color=INK2)
        short = "COF" if t == "COF" else "Width (log10 µm)"
        ax.set_title(f"{short}: residual by day (bar = mean)", loc="left", color=INK)
        ax = axes[i, 1]
        cm._style(ax)
        ax.axhline(0, color=INK2, lw=0.8)
        for k, mk in (("old", "o"), ("new", "s")):
            e = P[t][k] - y
            if k == "old":
                ax.scatter(d["Load_N"], e, s=26, marker=mk, facecolor="none", edgecolor=INK2, lw=1.1, zorder=3,
                           label="old")
            else:
                ax.scatter(d["Load_N"], e, s=24, marker=mk, facecolor=INK, edgecolor=SURFACE, lw=0.6, zorder=3,
                           label="new")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100])
        ax.set_xticklabels(["10", "20", "50", "100"])
        ax.set_xlabel("Load (N)", color=INK2)
        ax.set_ylabel("predicted − measured", color=INK2)
        ax.set_title(f"{short}: residual vs load", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_confusion(CM, out):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.8), facecolor=SURFACE)
    labs = ["low", "mid", "high"]
    for i, t in enumerate(cm.TARGETS):
        for j, k in enumerate(["old", "new"]):
            ax = axes[i, j]
            m = CM[t][k]
            ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1, CM[t]["max"]))
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, str(m[a, b]), ha="center", va="center", fontsize=11,
                            color="white" if m[a, b] > 0.6 * CM[t]["max"] else INK)
            ax.set_xticks(range(3))
            ax.set_xticklabels(labs)
            ax.set_yticks(range(3))
            ax.set_yticklabels(labs)
            ax.set_xlabel("predicted tercile", color=INK2)
            ax.set_ylabel("measured tercile", color=INK2)
            acc = np.trace(m) / m.sum()
            ax.set_title(f"{'Old' if k == 'old' else 'New'} — {LABEL[t].split(' (')[0]}  (accuracy {acc:.0%})",
                         loc="left", color=INK, fontsize=9)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_test(Ct, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), facecolor=SURFACE)
    for ax, t in zip(axes, cm.TARGETS):
        cm._style(ax)
        c = Ct[t]
        x = np.arange(len(c))
        ax.scatter(x, c["measured"], marker="x", s=46, color=cm.COLORS[1], lw=1.6, zorder=4, label="measured")
        ax.scatter(x - 0.12, c["old"], s=34, facecolor="none", edgecolor=INK2, lw=1.2, zorder=3, label="old")
        ax.scatter(x + 0.12, c["new"], s=30, marker="s", facecolor=INK, edgecolor=SURFACE, zorder=3, label="new")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n{l:g} N" for r, l in zip(c.index, c["Load_N"])])
        ax.set_title(f"{LABEL[t]}: 5 test runs", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = cm.load_runs(ROOT / "data" / "processed" / "runs_checkpoints.csv",
                     ROOT / "data" / "processed" / "wear_locations.csv")
    day = d["day"].to_numpy()
    rep = d[d["group"] == "R04"]
    tol = {t: float(rep[t].std(ddof=1)) for t in cm.TARGETS}

    # ---- B: head-to-head LOGO, all 23 runs
    P, M_B, CM = {}, {}, {}
    for t in cm.TARGETS:
        y = d[t].to_numpy(float)
        o_mu, o_sd = old_logo(d, t)
        n_mu, n_half = new_logo(d, t)
        P[t] = {"old": o_mu, "new": n_mu}
        cuts = np.quantile(y, [1 / 3, 2 / 3])
        mo = full_metrics(y, o_mu, day, tol[t], 1.96 * o_sd if np.isfinite(o_sd).all() else None, cuts)
        mn = full_metrics(y, n_mu, day, tol[t], n_half, cuts)
        CM[t] = {"old": mo.pop("_cm"), "new": mn.pop("_cm")}
        CM[t]["max"] = max(CM[t]["old"].max(), CM[t]["new"].max())
        M_B[t] = pd.DataFrame({"old": mo, "new": mn})
    pd.concat(M_B, names=["target", "metric"]).to_csv(OUT / "B_logo_all23.csv")

    # ---- C: the 5 sealed test runs
    tr, te = d[d["Role"] != "test"], d[d["Role"] == "test"]
    M_C, Ct = {}, {}
    for t, (name, _, fdir, col) in OLD_SPEC.items():
        tp = pd.read_csv(RES / fdir / "test_predictions.csv")
        tp = tp[(tp["target"] == col) & (tp["Checkpoint"] == 2)].set_index("Experiment_ID")
        y = te[t].to_numpy(float)
        o_mu = tp.loc[te["Experiment_ID"], "y_pred"].to_numpy(float)
        o_sd = tp.loc[te["Experiment_ID"], "y_std"].to_numpy(float)
        n_mu, n_half = ols_fit_predict(tr[NEW_FEATS[t]].to_numpy(float), tr[t].to_numpy(float),
                                       te[NEW_FEATS[t]].to_numpy(float))
        dte = te["day"].to_numpy()
        mo = full_metrics(y, o_mu, dte, tol[t], 1.96 * o_sd if np.isfinite(o_sd).all() else None)
        mn = full_metrics(y, n_mu, dte, tol[t], n_half)
        M_C[t] = pd.DataFrame({"old": mo, "new": mn})
        Ct[t] = pd.DataFrame({"Load_N": te["Load_N"].to_numpy(), "measured": y, "old": o_mu, "new": n_mu},
                             index=te["Experiment_ID"].to_numpy())
    pd.concat(M_C, names=["target", "metric"]).to_csv(OUT / "C_sealed_test5.csv")

    # ---- A: as reported by each pipeline
    rows = []
    from .evaluate import pooled_metrics
    for t, (name, _, fdir, col) in OLD_SPEC.items():
        pm = pooled_metrics(pd.read_csv(RES / fdir / "predictions.csv")).set_index(["target", "model"])
        tm = pd.read_csv(RES / fdir / "test_metrics.csv").set_index("target")
        rows.append({"target": LABEL[t], "model": f"old: {name.upper()} on ln L, ln f, ln t",
                     "validation": "5×5 grouped CV, 18 runs" + (" × 2 checkpoints" if t == "COF" else ""),
                     "R²": pm.loc[(col, name), "R2_pooled"], "RMSE": pm.loc[(col, name), "RMSE_pooled"]})
        rows.append({"target": LABEL[t], "model": f"old: {name.upper()} on ln L, ln f, ln t",
                     "validation": f"sealed test, 5 runs ({int(tm.loc[col, 'n'])} rows)",
                     "R²": tm.loc[col, "R2"], "RMSE": tm.loc[col, "RMSE"]})
        cvn = pd.read_csv(RES / "campaign_model" / "cv_metrics.csv")
        r = cvn[(cvn["target"] == t) & (cvn["inputs"] == cm.PHYSICS[t]) & (cvn["model"] == "linear")].iloc[0]
        rows.append({"target": LABEL[t], "model": f"new: linear on {cm.PHYSICS[t].replace(' (physics)', '')}",
                     "validation": "leave-one-group-out, 23 runs", "R²": r["R2"], "RMSE": r["RMSE"]})
    pmk = pooled_metrics(pd.read_csv(RES / "final" / "predictions.csv")).set_index(["target", "model"])
    tmk = pd.read_csv(RES / "final" / "test_metrics.csv").set_index("target")
    rows += [{"target": "log10 k (wear rate)", "model": "old: GPR on ln L, ln f, ln t",
              "validation": "5×5 grouped CV, 9 detected runs", "R²": pmk.loc[("log10_k", "gpr"), "R2_pooled"],
              "RMSE": pmk.loc[("log10_k", "gpr"), "RMSE_pooled"]},
             {"target": "log10 k (wear rate)", "model": "old: GPR on ln L, ln f, ln t",
              "validation": "sealed test, 3 detected runs", "R²": tmk.loc["log10_k", "R2"],
              "RMSE": tmk.loc["log10_k", "RMSE"]},
             {"target": "log10 k (wear rate)", "model": "new: not modelled", "validation": "–", "R²": np.nan,
              "RMSE": np.nan}]
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "A_as_reported.csv", index=False)

    figs = {"parity": OUT / "parity.png", "resid": OUT / "residuals.png", "cm": OUT / "confusion.png",
            "test": OUT / "test_runs.png"}
    fig_parity(d, P, figs["parity"])
    fig_residuals(d, P, figs["resid"])
    fig_confusion(CM, figs["cm"])
    fig_test(Ct, figs["test"])
    build_pdf(A, M_B, M_C, tol, figs, OUT / "model_comparison.pdf")
    print("wrote", OUT / "model_comparison.pdf")


# ----------------------------------------------------------------------------- PDF
def _num(metric: str, v, target: str) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if metric == "n runs":
        return f"{int(v)}"
    if metric in ("R²", "Pearson r", "Spearman ρ", "calibration slope", "tercile κ (Cohen)"):
        return f"{v:+.2f}"
    if metric.startswith("within") or metric.startswith("tercile acc") or metric.endswith("coverage"):
        return f"{v:.0%}"
    if metric.startswith("bias"):
        return f"{v:+.4f}" if target == "COF" else f"{v:+.3f}"
    return f"{v:.4f}" if target == "COF" else f"{v:.3f}"


def bold_better(M: dict, order: list) -> list:
    """Table styles that bold the better of old/new per metric (closer to 0 / 1 / 95 % where that is ideal)."""
    out = []
    for i, m in enumerate(order, start=1):
        for j, tg in enumerate(cm.TARGETS):
            o, n = M[tg].loc[m, "old"], M[tg].loc[m, "new"]
            if not (np.isfinite(o) and np.isfinite(n)) or o == n or m == "n runs":
                continue
            if m in ("RMSE", "MAE", "max |error|") or m.startswith("bias") or m == "95 % interval mean width":
                win = "new" if abs(n) < abs(o) else "old"
            elif m == "calibration slope":
                win = "new" if abs(n - 1) < abs(o - 1) else "old"
            elif m == "95 % interval coverage":
                win = "new" if abs(n - 0.95) < abs(o - 0.95) else "old"
            else:
                win = "new" if n > o else "old"
            col = 1 + 2 * j + (1 if win == "new" else 0)
            out.append(("FONT", (col, i), (col, i), "DVB", 7.5))
    return out


def build_pdf(A, M_B, M_C, tol, figs, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fdir = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DV", str(fdir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DVB", str(fdir / "DejaVuSans-Bold.ttf")))
    addMapping("DV", 0, 0, "DV")
    addMapping("DV", 1, 0, "DVB")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="DVB", fontSize=14, leading=18, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DVB", fontSize=10.5, leading=13, spaceBefore=8,
                        spaceAfter=4)
    note = ParagraphStyle("note", parent=ss["Normal"], fontName="DV", fontSize=7, leading=9,
                          textColor=colors.HexColor(INK2))
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontName="DV", fontSize=7.5, leading=9.5)
    W = A4[0] - 30 * mm
    blue, orange = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834")
    ts = [("FONT", (0, 0), (-1, -1), "DV", 7.5), ("FONT", (0, 0), (-1, 0), "DVB", 7.5),
          ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK2)),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f3ef")]),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]

    def P_(s, st=cell):
        return Paragraph(s, st)

    story = [Paragraph("Model comparison: DOE pipeline (old) vs campaign model (new)", h1)]
    defs = [["", "Old (DOE pipeline, ml.train)", "New (campaign model)"],
            ["COF model", P_("ANN ensemble (20 seeds) on ln L, ln f, ln t"),
             P_("linear: COF = μ<sub>0</sub> + c·L<super>−1/3</super> + day offset")],
            ["Width model", P_("XGBoost on ln L, ln f, ln t"), P_("linear: log w = a + b·ln L + day offset")],
            ["Wear-rate model", P_("GPR on ln L, ln f, ln t (9 detected runs)"), "not modelled"],
            ["Training data", "18 runs (test runs sealed)", "all 23 runs"],
            ["Parameters (COF / width)", "≤ 26 weights × 20 / boosted stumps", "3 / 3"]]
    t = Table(defs, colWidths=[0.22 * W, 0.39 * W, 0.39 * W])
    t.setStyle(TableStyle(ts))
    story += [t, Spacer(1, 4)]

    story.append(Paragraph("A. As reported by each pipeline", h2))
    rows = [["target", "model", "validation", "R²", "RMSE"]]
    for _, r in A.iterrows():
        tgt = "COF" if r["target"].startswith("COF") else "w"
        rows.append([P_(r["target"]), P_(r["model"]), P_(r["validation"]), _num("R²", r["R²"], tgt),
                     _num("RMSE", r["RMSE"], tgt)])
    t = Table(rows, colWidths=[0.2 * W, 0.3 * W, 0.3 * W, 0.1 * W, 0.1 * W])
    t.setStyle(TableStyle(ts + [("ALIGN", (3, 0), (-1, -1), "RIGHT")]))
    story += [t, Paragraph("R² on a sealed test set is relative to that set's own mean (5 runs). Width and k in "
                           "log10 units.", note)]

    story.append(Paragraph("B. Head-to-head: leave-one-group-out over all 23 runs (identical protocol)", h2))
    order = [m for m in M_B["COF"].index]
    rows = [["metric", "COF old", "COF new", "width old", "width new"]]
    for m in order:
        rows.append([m] + [_num(m, M_B[t].loc[m, k], t) for t in cm.TARGETS for k in ("old", "new")])
    t = Table(rows, colWidths=[0.32 * W] + [0.17 * W] * 4)
    better = bold_better(M_B, order)
    t.setStyle(TableStyle(ts + [("ALIGN", (1, 0), (-1, -1), "RIGHT")] + better))
    story += [t, Paragraph(
        f"Bold = better of the two. Width in log10 µm (RMSE 0.127 ≈ ±34 %). Noise SD = replicate SD of R04/R14/R20: "
        f"COF {tol['COF']:.4f}, width {tol['log10_w']:.3f}. Calibration slope = slope of measured on predicted (ideal 1). "
        "Intervals: old ANN = ensemble spread only; old XGBoost has none; new = OLS 95 % prediction interval. "
        "Old model refitted per fold with the same grid and inner GroupKFold as ml.train.", note)]

    story += [PageBreak(), Paragraph("B. Parity (each run predicted with its group left out)", h2),
              Image(str(figs["parity"]), width=W * 0.92, height=W * 0.92 * 6.6 / 7.2)]
    story += [PageBreak(), Paragraph("B. Residuals", h2),
              Image(str(figs["resid"]), width=W * 0.92, height=W * 0.92 * 6.2 / 7.2),
              Paragraph("Left: open markers = old, filled = new; blue circles 13 Oct, orange squares 16 Oct. "
                        "Right: open grey = old, filled black = new.", note)]
    story += [PageBreak(), Paragraph("B. Confusion matrices (terciles of the measured values)", h2),
              Image(str(figs["cm"]), width=W * 0.8, height=W * 0.8 * 5.8 / 6.4),
              Paragraph("The targets are continuous, so a confusion matrix needs classes: each run is binned as "
                        "low / mid / high by the terciles of the measured values, for both measured and predicted.",
                        note)]

    story += [PageBreak(), Paragraph("C. Head-to-head on the 5 sealed test runs (R24, R25, R26, R28, R29)", h2)]
    order = [m for m in M_C["COF"].index]
    rows = [["metric", "COF old", "COF new", "width old", "width new"]]
    for m in order:
        rows.append([m] + [_num(m, M_C[t].loc[m, k], t) for t in cm.TARGETS for k in ("old", "new")])
    t = Table(rows, colWidths=[0.32 * W] + [0.17 * W] * 4)
    t.setStyle(TableStyle(ts + [("ALIGN", (1, 0), (-1, -1), "RIGHT")] + bold_better(M_C, order)))
    story += [t, Spacer(1, 3), Image(str(figs["test"]), width=W, height=W * 3.0 / 7.2),
              Paragraph("Bold = better of the two. Old = its sealed predictions (trained on the 18 training runs, scored once). New = fitted on "
                        "the same 18 runs, scored after the test labels had been seen. All 5 test runs are from 16 Oct. "
                        "COF on checkpoint 2.", note)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"ME 323 · model comparison · page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                      bottomMargin=15 * mm, title="Model comparison",
                      author="ME 323").build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
