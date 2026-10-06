"""Interpretation of the fitted models: SHAP, partial dependence, KAN functions, GPR maps.

Everything here uses the *final* models (refitted on all CV data) or the out-of-fold
CV predictions, never training-set fits, for any accuracy statement.

Partial dependence is computed on the design variables (carbon, load, frequency)
and every other input is recomputed physically consistently
(:func:`ml.features.complete_inputs`): moving load also moves the Hertz pressure and
PV; moving carbon moves hardness to that carbon level's mean. A naive PDP on
``Load_N`` with ``Hertz_p_mean_MPa`` held fixed would describe impossible contacts.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting as P
from .config import has_module, stable_seed
from .data import COF_T_RUN, TARGETS, TriboData
from .evaluate import CVResult, FinalResult, make_design
from .features import DESIGN, build_X, complete_inputs
from .metrics import calibration_curve

log = logging.getLogger("ml.interpret")


class WearHead:
    """Expose a multi-task model's wear head as a plain ``predict(run_X)`` estimator."""

    def __init__(self, est):
        self.est = est

    def predict(self, X):
        return self.est.predict_wear(X)


# ---------------------------------------------------------------------------
# partial dependence
def partial_dependence(est, data: TriboData, columns: list[str], var: str, grid: np.ndarray,
                       carbon: float | None = None) -> np.ndarray:
    """Average prediction over the CV runs with ``var`` set to each grid value.

    With ``carbon`` given, every run is also moved to that carbon level (two-way PD).
    """
    ref = data.cv_runs
    prim0 = ref[DESIGN + (["Hardness_HV"] if "Hardness_HV" in ref else [])].reset_index(drop=True)
    out = []
    for v in grid:
        prim = prim0.copy()
        prim[var] = v
        if carbon is not None:
            prim["Carbon_pct"] = carbon
        if var == "Carbon_pct" or carbon is not None:
            prim = prim.drop(columns=["Hardness_HV"], errors="ignore")   # hardness follows carbon
        X = build_X(complete_inputs(prim, ref), columns)
        out.append(float(np.mean(est.predict(X))))
    return np.asarray(out)


def curvature(x: np.ndarray, y: np.ndarray) -> dict:
    """Linear vs curved: chord sagitta at mid-range relative to the total rise, and fit R²."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mid = np.interp(0.5 * (x[0] + x[-1]), x, y)
    rise = y[-1] - y[0]
    sag = mid - 0.5 * (y[0] + y[-1])
    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2)
    ss = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2_lin = 1 - float(((y - np.polyval(lin, x)) ** 2).sum()) / ss
    rel = sag / abs(rise) if abs(rise) > 1e-12 else np.nan
    curved = np.isfinite(rel) and abs(rel) > 0.1 and r2_lin < 0.98
    shape = ("flat" if not np.isfinite(rel) else "approximately linear" if not curved
             else "convex (accelerating)" if rel < 0 else "concave (saturating)")
    return {"slope_per_unit": float(lin[0]), "quad_coef": float(quad[0]), "R2_linear": r2_lin,
            "sagitta_rel": float(rel), "rise": float(rise), "shape": shape}


@P.styled
def pdp_figure(est, model: str, data: TriboData, target: str, columns: list[str], n: int,
               out: Path) -> tuple[Path, pd.DataFrame, dict]:
    """Load (one line per carbon level), carbon and frequency partial dependence."""
    spec = TARGETS[target]
    ref = data.cv_runs
    carbons = sorted(ref["Carbon_pct"].unique())[:4]
    grids = {v: np.linspace(ref[v].min(), ref[v].max(), n) for v in DESIGN}
    rows = []
    fig, ax = P.new_fig(1, 3, w=11.5, h=3.4)
    for k, c in enumerate(carbons):
        yv = partial_dependence(est, data, columns, "Load_N", grids["Load_N"], carbon=c)
        ax[0, 0].plot(grids["Load_N"], yv, color=P.ORDINAL_BLUE[k % 4], label=f"{c:g} % C")
        rows += [{"var": "Load_N", "carbon": c, "x": x, "pd": v} for x, v in zip(grids["Load_N"], yv)]
    ya = partial_dependence(est, data, columns, "Load_N", grids["Load_N"])
    rows += [{"var": "Load_N", "carbon": "all", "x": x, "pd": v} for x, v in zip(grids["Load_N"], ya)]
    shape = {"load": curvature(grids["Load_N"], ya)}
    ax[0, 0].set_xlabel("load [N]")
    ax[0, 0].set_ylabel(f"partial dependence: {spec.label}")
    ax[0, 0].set_title(f"load → {target} ({shape['load']['shape']})")
    ax[0, 0].legend(title="carbon", loc="best")
    for j, var, xl in ((1, "Carbon_pct", "carbon [wt %]"), (2, "Freq_Hz", "frequency [Hz]")):
        yv = partial_dependence(est, data, columns, var, grids[var])
        rows += [{"var": var, "carbon": "all", "x": x, "pd": v} for x, v in zip(grids[var], yv)]
        shape["carbon" if var == "Carbon_pct" else "freq"] = curvature(grids[var], yv)
        ax[0, j].plot(grids[var], yv, color=P.SLOTS[0])
        ax[0, j].set_xlabel(xl)
        ax[0, j].set_title(f"{xl.split(' ')[0]} → {target}")
    fig.suptitle(f"Partial dependence, {model} (derived inputs recomputed physically)",
                 x=0.01, ha="left", color=P.INK2, fontsize=9)
    path = P.save(fig, out)
    return path, pd.DataFrame(rows), shape


# ---------------------------------------------------------------------------
# SHAP
def shap_values(est, model: str, X: pd.DataFrame, seed: int, max_samples: int):
    """SHAP values on (a subsample of) the training inputs; trees exactly, others model-agnostic."""
    import shap
    rng = np.random.default_rng(seed)
    Xs = X.iloc[rng.permutation(len(X))[:max_samples]].reset_index(drop=True)
    if model == "xgb":
        return shap.TreeExplainer(est).shap_values(Xs), Xs
    if model == "rf":
        pre = est[:-1]
        Xt = pd.DataFrame(pre.transform(Xs), columns=X.columns)
        return shap.TreeExplainer(est[-1]).shap_values(Xt), Xs
    masker = shap.maskers.Independent(Xs, max_samples=min(len(Xs), 50))
    algo = "exact" if X.shape[1] <= 10 else "permutation"
    expl = shap.Explainer(lambda a: est.predict(pd.DataFrame(a, columns=X.columns)), masker,
                          algorithm=algo, seed=seed)
    return expl(Xs).values, Xs


@P.styled
def shap_figure(sv: np.ndarray, Xs: pd.DataFrame, title: str, out: Path) -> tuple[Path, pd.DataFrame]:
    """Mean |SHAP| bars + a strip of per-run SHAP values coloured by the input's value."""
    imp = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    order = imp["feature"].tolist()
    fig, ax = P.new_fig(1, 2, w=10, h=0.45 * len(order) + 1.4,
                        gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.45})
    ax[0, 0].barh(order, imp["mean_abs_shap"], color=P.SLOTS[0], height=0.6)
    ax[0, 0].invert_yaxis()
    ax[0, 0].grid(axis="y", visible=False)
    ax[0, 0].set_xlabel("mean |SHAP| (target units)")
    ax[0, 0].set_title(title)
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        j = list(Xs.columns).index(f)
        v = Xs[f].to_numpy(float)
        rngv = np.nanmax(v) - np.nanmin(v)
        c = (v - np.nanmin(v)) / rngv if rngv > 0 else np.full(len(v), 0.5)
        sc = ax[0, 1].scatter(sv[:, j], i + rng.uniform(-0.18, 0.18, len(v)), c=c, cmap=P.CMAP_MEAN,
                              vmin=0, vmax=1, s=16, edgecolors=P.SURFACE, linewidths=0.6)
    ax[0, 1].axvline(0, color=P.AXIS, lw=1)
    ax[0, 1].set_yticks(range(len(order)), order)
    ax[0, 1].invert_yaxis()
    ax[0, 1].grid(axis="y", visible=False)
    ax[0, 1].set_xlabel("SHAP value per run")
    cb = fig.colorbar(sc, ax=ax[0, 1], fraction=0.04, pad=0.02)
    cb.set_label("input value (low → high)", color=P.INK2)
    cb.outline.set_visible(False)
    return P.save(fig, out), imp


# ---------------------------------------------------------------------------
# KAN functions
@P.styled
def kan_figure(funcs: list[dict], title: str, out: Path) -> tuple[Path, pd.DataFrame]:
    inputs = list(dict.fromkeys(f["input"] for f in funcs))
    nodes = sorted({f["node"] for f in funcs})
    nc = min(4, len(inputs))
    nr = int(np.ceil(len(inputs) / nc))
    fig, ax = P.new_fig(nr, nc, w=3.1 * nc, h=2.6 * nr)
    imp_rows = []
    for k, name in enumerate(inputs):
        a = ax[k // nc, k % nc]
        fs = [f for f in funcs if f["input"] == name]
        tot = np.nansum([f["importance"] for f in fs])
        imp_rows.append({"input": name, "importance": tot})
        for f in fs:
            a.plot(f["x"], f["phi"], color=P.SLOTS[f["node"] % 8], lw=1.6,
                   label=f"node {f['node'] + 1}")
        a.set_title(f"{name}  (Σ|φ| = {tot:.2f})", fontsize=9)
        a.set_xlabel(name, fontsize=8.5)
        if k % nc == 0:
            a.set_ylabel("φ(x)")
    for k in range(len(inputs), nr * nc):
        ax[k // nc, k % nc].set_visible(False)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles[:len(nodes)], labels[:len(nodes)], loc="upper right", ncol=len(nodes),
               title="hidden node")
    fig.suptitle(title, x=0.01, ha="left", color=P.INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return P.save(fig, out), pd.DataFrame(imp_rows).sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# GPR maps
@P.styled
def gpr_maps(est, data: TriboData, target: str, columns: list[str], n: int, out: Path):
    """Predicted mean and std over the load x frequency plane, one column per carbon level."""
    ref = data.cv_runs
    carbons = sorted(ref["Carbon_pct"].unique())[:4]
    L = np.linspace(ref["Load_N"].min(), ref["Load_N"].max(), n)
    F = np.linspace(ref["Freq_Hz"].min(), ref["Freq_Hz"].max(), n)
    LL, FF = np.meshgrid(L, F)
    mus, sds, rows = [], [], []
    for c in carbons:
        prim = pd.DataFrame({"Carbon_pct": c, "Load_N": LL.ravel(), "Freq_Hz": FF.ravel()})
        X = build_X(complete_inputs(prim, ref), columns)
        mu, sd = est.predict(X, return_std=True)
        lat = est.predict_latent_std(X)
        mus.append(mu.reshape(LL.shape))
        sds.append(lat.reshape(LL.shape))
        rows.append(pd.DataFrame({"Carbon_pct": c, "Load_N": LL.ravel(), "Freq_Hz": FF.ravel(),
                                  "mean": mu, "std_predictive": sd, "std_latent": lat}))
    spec = TARGETS[target]
    fig, ax = P.new_fig(2, len(carbons), w=3.0 * len(carbons) + 0.8, h=5.8)
    noise_sd = float(np.sqrt(est.noise_var_))
    for row, (vals, cmap, lab) in enumerate(((mus, P.CMAP_MEAN, f"mean {spec.label}"),
                                             (sds, P.CMAP_SD, "model std (excl. noise)"))):
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        for k, c in enumerate(carbons):
            a = ax[row, k]
            cs = a.contourf(LL, FF, vals[k], levels=P.nice_levels(lo, hi), cmap=cmap)
            a.grid(False)
            tr = ref[np.isclose(ref["Carbon_pct"], c)]
            P.dots(a, tr["Load_N"], tr["Freq_Hz"], P.INK2, size=14)
            ho = data.holdout_runs[np.isclose(data.holdout_runs["Carbon_pct"], c)]
            if len(ho):
                P.dots(a, ho["Load_N"], ho["Freq_Hz"], P.SLOTS[1], size=46, marker="D",
                       label="unseen point")
            a.set_title(f"{c:g} % C" if row == 0 else "", fontsize=9.5)
            a.set_xlabel("load [N]" if row == 1 else "")
            a.set_ylabel("frequency [Hz]" if k == 0 else "")
        cb = fig.colorbar(cs, ax=list(ax[row, :]), fraction=0.025, pad=0.02)
        cb.set_label(lab, color=P.INK2)
        cb.outline.set_visible(False)
    fig.suptitle(f"GPR {target}: mean (top) and model uncertainty (bottom; fitted noise sd "
                 f"{noise_sd:.3g} adds on top). Dots = CV runs, diamonds = unseen points",
                 x=0.01, ha="left", color=P.INK, fontsize=10)
    return P.save(fig, out), pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# CV-based figures
@P.styled
def calibration_figure(cv: CVResult, target: str, out: Path) -> tuple[Path | None, pd.DataFrame]:
    src = cv.predictions_time if target == "COF_t" else cv.predictions
    if src.empty:
        return None, pd.DataFrame()
    s = src[(src["target"] == target) & np.isfinite(src["y_std"].astype(float))]
    if "censored" in s:
        s = s[~s["censored"].astype(bool)]
    if s.empty:
        return None, pd.DataFrame()
    rows = []
    fig, ax = P.new_fig(w=4.4, h=4.0)
    a = ax[0, 0]
    a.plot([0, 1], [0, 1], color=P.AXIS, lw=1)
    for model, sm in s.groupby("model"):
        se = (sm["y_true"] - sm["y_pred"]) ** 2
        best_fs = se.groupby(sm["feature_set"]).mean().idxmin()
        d = sm[sm["feature_set"] == best_fs]
        cc = calibration_curve(d["y_true"], d["y_pred"], d["y_std"])
        a.plot(cc["nominal"], cc["observed"], color=P.model_color(model), label=f"{model} ({best_fs})")
        rows.append(cc.assign(model=model, feature_set=best_fs))
    a.set_xlabel("nominal central-interval coverage")
    a.set_ylabel("observed coverage (outer folds)")
    a.set_title(f"{target}: calibration")
    a.set_aspect("equal")
    a.legend(loc="upper left")
    return P.save(fig, out), pd.concat(rows, ignore_index=True)


@P.styled
def parity_figure(cv: CVResult, summary: pd.DataFrame, targets: list[str], out: Path) -> Path | None:
    tg = [t for t in targets if t in set(summary["target"])]
    if not tg:
        return None
    fig, ax = P.new_fig(1, len(tg), w=3.6 * len(tg), h=3.8)
    for k, t in enumerate(tg):
        best = summary[summary["target"] == t].sort_values("RMSE_mean").iloc[0]
        p = cv.predictions[(cv.predictions["target"] == t) & (cv.predictions["model"] == best["model"])
                           & (cv.predictions["feature_set"] == best["feature_set"])
                           & (cv.predictions["repeat"] == 0)]
        if "censored" in p:
            p = p[~p["censored"].astype(bool)]
        P.parity(ax[0, k], p["y_true"], p["y_pred"], P.model_color(best["model"]))
        ax[0, k].set_title(f"{t}: {best['model']} ({best['feature_set']})", fontsize=9.5)
        ax[0, k].set_xlabel("measured")
        ax[0, k].set_ylabel("predicted (out-of-fold)" if k == 0 else "")
        ax[0, k].text(0.03, 0.97, f"R² = {best['R2_mean']:.2f} ± {best['R2_std']:.2f}",
                      transform=ax[0, k].transAxes, va="top", color=P.INK2, fontsize=8.5)
    return P.save(fig, out)


@P.styled
def cof_curves_figure(pred_t: pd.DataFrame, summary: pd.DataFrame, n: int, out: Path,
                      title: str) -> Path | None:
    """Measured vs predicted COF(t) for a spread of runs (best single-task + best joint model)."""
    if pred_t.empty:
        return None
    s = summary[summary["target"] == "COF_t"].sort_values("RMSE_mean") if len(summary) else summary
    models = []
    if len(s):
        single = s[~s["model"].str.startswith("mt_")]
        joint = s[s["model"].str.startswith("mt_")]
        for sub in (single, joint):
            if len(sub):
                models.append((sub.iloc[0]["model"], sub.iloc[0]["feature_set"]))
    else:
        models = list(pred_t.groupby(["model", "feature_set"]).groups)[:2]
    p = pred_t[pred_t["repeat"] == 0] if "repeat" in pred_t else pred_t
    ss = p.groupby("Experiment_ID")["y_true"].mean().sort_values()
    ids = list(ss.index[np.unique(np.linspace(0, len(ss) - 1, n).round().astype(int))])
    nc = min(3, len(ids))
    nr = int(np.ceil(len(ids) / nc))
    fig, ax = P.new_fig(nr, nc, w=3.6 * nc, h=2.7 * nr, sharey=True)
    for k, eid in enumerate(ids):
        a = ax[k // nc, k % nc]
        meas = p[p["Experiment_ID"] == eid].drop_duplicates("Time_s").sort_values("Time_s")
        a.plot(meas["Time_s"], meas["y_true"], color=P.AXIS, lw=1.0, label="measured")
        for m, fs in models:
            d = p[(p["Experiment_ID"] == eid) & (p["model"] == m) & (p["feature_set"] == fs)] \
                .sort_values("Time_s")
            if d.empty:
                continue
            col = P.model_color(m)
            if np.isfinite(d["y_std"].astype(float)).any():
                a.fill_between(d["Time_s"], d["y_pred"] - 1.96 * d["y_std"],
                               d["y_pred"] + 1.96 * d["y_std"], color=col, alpha=0.1, lw=0)
            a.plot(d["Time_s"], d["y_pred"], color=col, label=f"{m} ({fs})")
        a.set_title(eid, fontsize=9)
        a.set_xlabel("time [s]" if k // nc == nr - 1 else "")
        a.set_ylabel("COF" if k % nc == 0 else "")
    for k in range(len(ids), nr * nc):
        ax[k // nc, k % nc].set_visible(False)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=len(lab))
    fig.suptitle(title, x=0.01, ha="left", color=P.INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return P.save(fig, out)


@P.styled
def holdout_figure(final: FinalResult, target: str, out: Path) -> Path | None:
    p = final.predictions[final.predictions["target"] == target] if len(final.predictions) else None
    if p is None or p.empty:
        return None
    ids = sorted(p["Experiment_ID"].unique())
    models = list(p.groupby(["model", "feature_set"]).groups)
    fig, ax = P.new_fig(w=max(4.5, 0.9 * len(ids) + 2), h=3.6)
    a = ax[0, 0]
    x = np.arange(len(ids))
    meas = p.drop_duplicates("Experiment_ID").set_index("Experiment_ID").reindex(ids)["y_true"]
    a.scatter(x, meas, marker="_", s=260, color=P.INK, linewidths=2.2, label="measured", zorder=4)
    w = 0.6 / max(len(models), 1)
    for i, (m, fs) in enumerate(models):
        d = p[(p["model"] == m) & (p["feature_set"] == fs)].set_index("Experiment_ID").reindex(ids)
        xi = x + (i - (len(models) - 1) / 2) * w
        has_pi = "pi_lo" in d and np.isfinite(d["pi_lo"].astype(float)).any()
        if has_pi:
            a.vlines(xi, d["pi_lo"], d["pi_hi"], color=P.model_color(m), lw=1.2, alpha=0.7)
        P.dots(a, xi, d["y_pred"], P.model_color(m), size=34, label=f"{m} ({fs})")
    a.set_xticks(x, ids)
    a.grid(axis="x", visible=False)
    a.set_ylabel(TARGETS[target].label if target in TARGETS else target)
    a.set_title(f"Unseen operating points: {target} (bars = 95 % prediction interval)")
    a.legend(loc="best")
    return P.save(fig, out)


# ---------------------------------------------------------------------------
def archard_check(data: TriboData) -> pd.DataFrame:
    """OLS of log10 V on log10 F and log10 S (+ carbon-level offsets), detected CV runs only.

    Archard's law (V = k F S with k independent of F and S) predicts both exponents = 1.
    With a fixed test duration S is proportional to frequency, so the S exponent also
    absorbs any frequency effect on k.
    """
    r = data.cv_runs
    need = ["Wear_Volume_mm3", "Load_N", "Sliding_distance_m", "Carbon_pct"]
    if not set(need) <= set(r.columns):
        return pd.DataFrame()
    r = r[(r.get("Wear_detected") != False) & (r["Wear_Volume_mm3"] > 0)  # noqa: E712
          & (r["Sliding_distance_m"] > 0)].dropna(subset=need)
    if len(r) < 8:
        return pd.DataFrame()
    y = np.log10(r["Wear_Volume_mm3"].to_numpy(float))
    cols = {"log10_F": np.log10(r["Load_N"].to_numpy(float)),
            "log10_S": np.log10(r["Sliding_distance_m"].to_numpy(float))}
    levels = sorted(r["Carbon_pct"].unique())
    X = [np.ones(len(r)), cols["log10_F"], cols["log10_S"]] + \
        [np.isclose(r["Carbon_pct"], c).astype(float) for c in levels[1:]]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(res @ res) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    from scipy.stats import t as tdist
    tq = float(tdist.ppf(0.975, dof))
    out = []
    for k, name in ((1, "load exponent (log10 F)"), (2, "distance exponent (log10 S)")):
        lo, hi = beta[k] - tq * se[k], beta[k] + tq * se[k]
        out.append({"term": name, "estimate": beta[k], "ci95_lo": lo, "ci95_hi": hi,
                    "archard": 1.0, "consistent_with_archard": bool(lo <= 1.0 <= hi), "n": len(y)})
    return pd.DataFrame(out)


def symbolic_analysis(data: TriboData, cfg: dict, out_dir: Path) -> pd.DataFrame | None:
    """PySR on log10 k (base features, all detected CV runs) if installed and enabled."""
    if not (cfg["interpret"].get("symbolic") and (cfg["models"].get("pysr") or {}).get("enabled")):
        return None
    if not has_module("pysr"):
        log.warning("symbolic regression skipped: pysr not installed")
        return None
    from .models.symbolic import SymbolicRegressor
    d = make_design(data, "log10_k", "base", cfg)
    try:
        sr = SymbolicRegressor(random_state=cfg["seed"],
                               **(cfg["models"]["pysr"].get("params") or {})).fit(d.X, d.y)
        eq = sr.equations()
        eq.to_csv(out_dir / "symbolic_equations.csv", index=False)
        return eq
    except Exception as exc:                                         # noqa: BLE001
        log.warning("symbolic regression failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
def run_interpretation(data: TriboData, cfg: dict, cv: CVResult, summary: pd.DataFrame,
                       final: FinalResult, out_dir: Path) -> dict:
    """Produce every interpretation figure/table; returns paths and findings for the report."""
    icfg = cfg["interpret"]
    figs = Path(out_dir) / "figures"
    tables = Path(out_dir) / "interpretation"
    tables.mkdir(exist_ok=True)
    res: dict = {"figures": {}, "pdp_shapes": {}, "shap": {}, "kan": {}, "gpr_lengthscales": {},
                 "notes": []}

    def safe(name, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:                                     # noqa: BLE001
            log.warning("interpretation step %s failed: %s", name, exc)
            res["notes"].append(f"{name} failed: {type(exc).__name__}: {exc}")
            return None

    main = summary[summary["censor_mode"] == cfg["censoring"]["mode"]]
    for t in main["target"].unique():
        lab = TARGETS[t].label if t in TARGETS else "COF, steady phase"
        p = safe(f"bars {t}", P.bar_rmse, main, t, lab, figs / f"cv_rmse_{t.replace('@', '_at_')}.png")
        if p:
            res["figures"][f"cv_rmse_{t}"] = p
    p = safe("parity", parity_figure, cv, main, [t for t in ["log10_k", "log10_V", "COF_ss_mean",
                                                             COF_T_RUN] if t in set(main["target"])],
             figs / "parity_best_models.png")
    if p:
        res["figures"]["parity"] = p
    for t in [t for t in ["log10_k", "COF_ss_mean", "COF_t", "log10_V"] if t in set(main["target"])]:
        r = safe(f"calibration {t}", calibration_figure, cv, t, figs / f"calibration_{t}.png")
        if r and r[0]:
            res["figures"][f"calibration_{t}"] = r[0]
            r[1].to_csv(tables / f"calibration_{t}.csv", index=False)
    if not cv.predictions_time.empty:
        p = safe("cof curves", cof_curves_figure, cv.predictions_time, main,
                 int(icfg.get("cof_curves", 6)), figs / "cof_curves_cv.png",
                 "COF(t): measured vs out-of-fold prediction (band = 95 % PI)")
        if p:
            res["figures"]["cof_curves_cv"] = p

    # ---- final-model interpretation --------------------------------------
    targets = [t for t in icfg.get("targets", ["log10_k", "COF_ss_mean"])]
    items = []
    for (t, model, fs), est in final.estimators.items():
        if t == "joint":
            if "log10_k" in targets:            # the wear head of a multi-task model
                items.append((("log10_k", model, fs), WearHead(est)))
            continue
        items.append(((t, model, fs), est))
    for (t, model, fs), est in items:
        d = make_design(data, t, fs, cfg)
        if d is None:
            continue
        cols = list(d.X.columns)
        tag = f"{t}__{model}__{fs}".replace("@", "_at_")
        if t in targets and TARGETS[t].regime == "run":
            r = safe(f"pdp {tag}", pdp_figure, est, model, data, t, cols,
                     int(icfg.get("pdp_points", 25)), figs / f"pdp_{tag}.png")
            if r:
                res["figures"][f"pdp_{tag}"] = r[0]
                r[1].to_csv(tables / f"pdp_{tag}.csv", index=False)
                res["pdp_shapes"][(t, model, fs)] = r[2]
            if icfg.get("shap", True) and has_module("shap"):
                r = safe(f"shap {tag}", shap_values, est, model, d.X,
                         stable_seed(cfg["seed"], "shap", tag), int(icfg.get("shap_max_samples", 100)))
                if r:
                    p, imp = shap_figure(r[0], r[1], f"SHAP {t}: {model} ({fs})",
                                         figs / f"shap_{tag}.png")
                    res["figures"][f"shap_{tag}"] = p
                    imp.to_csv(tables / f"shap_{tag}.csv", index=False)
                    res["shap"][(t, model, fs)] = imp
            if model == "gpr" and TARGETS[t].regime == "run":
                r = safe(f"gpr map {tag}", gpr_maps, est, data, t, cols, int(icfg.get("map_points", 41)),
                         figs / f"gpr_map_{tag}.png")
                if r:
                    res["figures"][f"gpr_map_{tag}"] = r[0]
                    r[1].to_csv(tables / f"gpr_map_{tag}.csv", index=False)
                res["gpr_lengthscales"][(t, fs)] = getattr(est, "length_scales_", {})
        if model == "kan" and icfg.get("kan_plots", True):
            funcs = safe(f"kan {tag}", est.edge_functions, d.X)
            if funcs:
                p, imp = kan_figure(funcs, f"KAN learned functions, first layer: {t} ({fs})",
                                    figs / f"kan_functions_{tag}.png")
                res["figures"][f"kan_{tag}"] = p
                imp.to_csv(tables / f"kan_importance_{tag}.csv", index=False)
                res["kan"][(t, fs)] = imp
    for (t, model, fs), est in final.estimators.items():
        if t == "joint" and model == "mt_kan" and icfg.get("kan_plots", True):
            from .models.kan import layer_functions
            jd_cols = est.feature_names_
            layer = est.trunk_kan(0)
            funcs = safe("mt_kan trunk", layer_functions, layer, est.pre_run_, jd_cols, None, 101,
                         est.spline_order)
            if funcs:
                p, _ = kan_figure(funcs, f"Multi-task KAN: trunk first-layer functions ({fs})",
                                  figs / f"kan_functions_mt_kan_trunk__{fs}.png")
                res["figures"][f"kan_mt_trunk_{fs}"] = p

    # ---- unseen points ---------------------------------------------------
    for t in sorted(set(final.predictions["target"])) if len(final.predictions) else []:
        p = safe(f"holdout {t}", holdout_figure, final, t, figs / f"holdout_{t.replace('@', '_at_')}.png")
        if p:
            res["figures"][f"holdout_{t}"] = p
    if len(final.predictions_time):
        p = safe("holdout cof curves", cof_curves_figure,
                 final.predictions_time.assign(repeat=0), pd.DataFrame(), 6,
                 figs / "cof_curves_holdout.png", "COF(t) at the unseen operating points")
        if p:
            res["figures"]["cof_curves_holdout"] = p

    res["archard"] = safe("archard", archard_check, data)
    if res["archard"] is not None and len(res["archard"]):
        res["archard"].to_csv(tables / "archard_check.csv", index=False)
    res["symbolic"] = safe("symbolic", symbolic_analysis, data, cfg, tables)
    return res
