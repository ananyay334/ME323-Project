"""Figures for the report (B6): parity, GPR response surfaces, ARD length scales, LOAO
per-grade errors, trajectory reconstructions and GPR calibration.

Accuracy statements use out-of-fold predictions only. The GPR response surfaces and
length scales come from a GPR fitted on all usable runs of a target (description, not
evaluation). Surface inputs are set physically consistently: speed and Hertz pressure are
recomputed with ``tribo_extract.physics``; hardness follows the carbon level's mean.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tribo_extract.physics import two_stage_wear

from . import plotting as P
from .data import TARGETS, TriboData
from .evaluate import make_design
from .features import build_X, complete_inputs
from .metrics import calibration_curve
from .models.gpr import GPRModel

log = logging.getLogger("ml.interpret")


def _oof(preds: pd.DataFrame, target: str, model: str, fs: str, split: str = "gkf5",
         repeat: int = 0) -> pd.DataFrame:
    p = preds[(preds["target"] == target) & (preds["model"] == model)
              & (preds["feature_set"] == fs) & (preds["split"] == split) & (preds["repeat"] == repeat)]
    return p[~p["censored"].astype(bool)] if "censored" in p else p


@P.styled
def parity_figure(preds: pd.DataFrame, best: pd.DataFrame, out: Path) -> Path | None:
    """Out-of-fold predicted vs measured, best model per target (gkf5, first repeat)."""
    if best.empty:
        return None
    n = len(best)
    nc = min(4, n)
    nr = int(np.ceil(n / nc))
    fig, ax = P.new_fig(nr, nc, w=3.4 * nc, h=3.5 * nr)
    for k, b in enumerate(best.itertuples()):
        a = ax[k // nc, k % nc]
        p = _oof(preds, b.target, b.model, b.feature_set)
        P.parity(a, p["y_true"], p["y_pred"], P.model_color(b.model))
        a.set_title(f"{b.target}: {b.model} ({b.feature_set})", fontsize=9)
        a.text(0.03, 0.97, f"R² = {b.R2_mean:.2f}", transform=a.transAxes, va="top",
               color=P.INK2, fontsize=8.5)
        a.set_xlabel("measured", fontsize=8.5)
        a.set_ylabel("predicted (out-of-fold)" if k % nc == 0 else "", fontsize=8.5)
    for k in range(n, nr * nc):
        ax[k // nc, k % nc].set_visible(False)
    fig.tight_layout()
    return P.save(fig, out)


def fit_full_gpr(data: TriboData, target: str, fs: str, cfg: dict):
    d = make_design(data, target, fs, cfg, cfg["censoring"]["mode"])
    if d is None or len(d.frame) < 5:
        return None, None
    params = dict((cfg["models"].get("gpr") or {}).get("params") or {})
    return GPRModel(random_state=cfg["seed"], **params).fit(d.X, d.y), list(d.X.columns)


@P.styled
def gpr_surface(est, cols: list[str], data: TriboData, target: str, n: int, out: Path) -> Path:
    """GPR mean and model σ over Load × Freq, one column per carbon level, design points on top."""
    ref = data.runs
    carbons = sorted(ref["Carbon_pct"].unique())
    L = np.linspace(ref["Load_N"].min(), ref["Load_N"].max(), n)
    F = np.linspace(ref["Freq_Hz"].min(), ref["Freq_Hz"].max(), n)
    LL, FF = np.meshgrid(L, F)
    mus, sds = [], []
    for c in carbons:
        prim = pd.DataFrame({"Carbon_pct": c, "Load_N": LL.ravel(), "Freq_Hz": FF.ravel()})
        X = build_X(complete_inputs(prim, ref), cols)
        mus.append(est.predict(X).reshape(LL.shape))
        sds.append(est.predict_latent_std(X).reshape(LL.shape))
    fig, ax = P.new_fig(2, len(carbons), w=3.0 * len(carbons) + 0.8, h=5.8)
    for row, (vals, cmap, lab) in enumerate(((mus, P.CMAP_MEAN, f"mean {TARGETS[target].label}"),
                                             (sds, P.CMAP_SD, "σ (model, excl. noise)"))):
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        for k, c in enumerate(carbons):
            a = ax[row, k]
            cs = a.contourf(LL, FF, vals[k], levels=P.nice_levels(lo, hi), cmap=cmap)
            a.grid(False)
            pts = ref[np.isclose(ref["Carbon_pct"], c)]
            P.dots(a, pts["Load_N"], pts["Freq_Hz"], P.INK2, size=22)
            a.set_title(f"{c:g} % C" if row == 0 else "", fontsize=9.5)
            a.set_xlabel("load [N]" if row == 1 else "")
            a.set_ylabel("frequency [Hz]" if k == 0 else "")
        cb = fig.colorbar(cs, ax=list(ax[row, :]), fraction=0.025, pad=0.02)
        cb.set_label(lab, color=P.INK2)
        cb.outline.set_visible(False)
    fig.suptitle(f"GPR response surface, {target}: mean (top), σ (bottom); dots = design points "
                 f"(fitted noise sd {np.sqrt(est.noise_var_):.3g})", x=0.01, ha="left",
                 color=P.INK, fontsize=10)
    return P.save(fig, out)


@P.styled
def lengthscale_figure(ls: dict, out: Path) -> Path | None:
    """ARD length scales (standardised inputs; small = strong / curved effect)."""
    if not ls:
        return None
    keys = list(ls)
    fig, ax = P.new_fig(1, len(keys), w=3.1 * len(keys), h=3.3)
    for k, key in enumerate(keys):
        a = ax[0, k]
        names = list(ls[key])
        a.barh(names, [ls[key][n] for n in names], color=P.SLOTS[0], height=0.6)
        a.set_xscale("log")
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
        a.set_title(f"{key[0]} ({key[1]})", fontsize=9)
        a.set_xlabel("length scale")
    fig.tight_layout()
    return P.save(fig, out)


@P.styled
def loao_figure(preds: pd.DataFrame, summary: pd.DataFrame, targets: list[str], out: Path,
                seed: int = 0) -> tuple[Path | None, pd.DataFrame]:
    """Per held-out grade RMSE (bars) with bootstrap 95 % CI over that grade's runs (whiskers)."""
    lo = preds[preds["split"] == "loao"]
    targets = [t for t in targets if t in set(lo["target"])]
    if not targets:
        return None, pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    fig, ax = P.new_fig(1, len(targets), w=4.2 * len(targets), h=3.5)
    for k, t in enumerate(targets):
        a = ax[0, k]
        s = summary[(summary["target"] == t) & (summary["split"] == "loao")]
        models = s.sort_values("RMSE_mean").groupby("model").head(1)
        grades = sorted(lo["fold_label"].unique())
        w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models.itertuples()):
            p = lo[(lo["target"] == t) & (lo["model"] == m.model) & (lo["feature_set"] == m.feature_set)]
            p = p[~p["censored"].astype(bool)]
            vals, errs = [], []
            for g in grades:
                e = (p.loc[p["fold_label"] == g, "y_pred"] - p.loc[p["fold_label"] == g, "y_true"]).to_numpy()
                r = float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
                boot = [np.sqrt(np.mean(rng.choice(e, len(e)) ** 2)) for _ in range(200)] if len(e) else [np.nan]
                vals.append(r)
                errs.append([r - np.percentile(boot, 2.5), np.percentile(boot, 97.5) - r])
                rows.append({"target": t, "model": m.model, "feature_set": m.feature_set, "grade": g,
                             "RMSE": r, "n": len(e)})
            x = np.arange(len(grades)) + (i - (len(models) - 1) / 2) * w
            a.bar(x, vals, width=w * 0.9, color=P.model_color(m.model), label=m.model,
                  yerr=np.array(errs).T, error_kw={"ecolor": P.INK2, "elinewidth": 0.8})
        a.set_xticks(np.arange(len(grades)), grades)
        a.grid(axis="x", visible=False)
        a.set_title(f"LOAO, {t}", fontsize=9.5)
        a.set_xlabel("held-out grade")
        a.set_ylabel("RMSE" if k == 0 else "")
        if k == 0:
            a.legend(fontsize=7.5)
    fig.tight_layout()
    return P.save(fig, out), pd.DataFrame(rows)


@P.styled
def trajectory_figure(data: TriboData, route: pd.DataFrame, preds: pd.DataFrame,
                      best: pd.DataFrame, out: Path) -> Path | None:
    """Measured checkpoints vs the physics-route curve and the best direct model (one run per grade)."""
    if route is None or route.empty or data.checkpoints is None:
        return None
    r0 = route[route["repeat"] == 0]
    fs = r0["feature_set"].iloc[0]
    r0 = r0[r0["feature_set"] == fs].set_index("Experiment_ID")
    runs = data.runs.set_index("Experiment_ID")
    picks = []
    for spec, sub in data.runs.groupby("Specimen_ID"):
        cand = [e for e in sub.sort_values("Role", key=lambda s: s != "anchor_center")["Experiment_ID"]
                if e in r0.index]
        if cand:
            picks.append(cand[0])
    if not picks:
        return None
    bd = best[best["target"] == "log10_V_t"]
    direct = _oof(preds, "log10_V_t", bd["model"].iloc[0], bd["feature_set"].iloc[0]) if len(bd) else None
    fig, ax = P.new_fig(1, len(picks), w=3.4 * len(picks), h=3.2)
    tt = np.linspace(0, float(data.checkpoints["Time_s"].max()), 200)
    for k, e in enumerate(picks):
        a = ax[0, k]
        th = r0.loc[e, ["Vdot_steady", "dV_run", "tau_trans"]].to_numpy(float)
        a.plot(tt, two_stage_wear(tt, *th), color=P.model_color("physics_route"),
               label=f"physics route ({fs})")
        cp = data.checkpoints[data.checkpoints["Experiment_ID"] == e]
        det = cp["Wear_detected"].fillna(False).astype(bool)
        P.dots(a, cp.loc[det, "Time_s"], cp.loc[det, "Wear_Volume_mm3"], P.INK, size=26,
               label="measured")
        if (~det).any():
            a.scatter(cp.loc[~det, "Time_s"], cp.loc[~det, "Wear_Volume_LOD_mm3"], marker="v",
                      color=P.MUTED, s=26, label="< LOD")
        if direct is not None:
            dd = direct[direct["Experiment_ID"] == e]
            P.dots(a, dd["Time_s"], 10 ** dd["y_pred"], P.model_color(bd["model"].iloc[0]), size=26,
                   marker="D", label=f"direct {bd['model'].iloc[0]}")
        a.set_title(f"{e} ({runs.at[e, 'Specimen_ID']}, {runs.at[e, 'Load_N']:g} N, "
                    f"{runs.at[e, 'Freq_Hz']:g} Hz)", fontsize=9)
        a.set_xlabel("time [s]")
        a.set_ylabel("V_w [mm³]" if k == 0 else "")
        a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=len(lab), fontsize=8)
    fig.suptitle("Wear trajectory: out-of-fold reconstructions (gkf5, first repeat)", x=0.01,
                 ha="left", color=P.INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return P.save(fig, out)


@P.styled
def calibration_figure(preds: pd.DataFrame, summary: pd.DataFrame, levels: int,
                       out: Path) -> tuple[Path | None, pd.DataFrame]:
    """GPR calibration on gkf5 out-of-fold predictions, one line per target (best feature set)."""
    s = summary[(summary["model"] == "gpr") & (summary["split"] == "gkf5")]
    if s.empty:
        return None, pd.DataFrame()
    rows = []
    fig, ax = P.new_fig(w=4.6, h=4.2)
    a = ax[0, 0]
    a.plot([0, 1], [0, 1], color=P.AXIS, lw=1)
    for i, b in enumerate(s.sort_values("RMSE_mean").groupby("target").head(1).itertuples()):
        p = preds[(preds["target"] == b.target) & (preds["model"] == "gpr")
                  & (preds["feature_set"] == b.feature_set) & (preds["split"] == "gkf5")]
        p = p[~p["censored"].astype(bool)]
        cc = calibration_curve(p["y_true"], p["y_pred"], p["y_std"], np.linspace(0.05, 0.95, levels))
        a.plot(cc["nominal"], cc["observed"], color=P.SLOTS[i % 8], lw=1.6, label=b.target)
        rows.append(cc.assign(target=b.target, feature_set=b.feature_set))
    a.set_xlabel("nominal central-interval coverage")
    a.set_ylabel("observed coverage (gkf5 outer folds)")
    a.set_title("GPR calibration")
    a.set_aspect("equal")
    a.legend(fontsize=7.5, loc="upper left")
    return P.save(fig, out), pd.concat(rows, ignore_index=True)


def run_interpretation(data: TriboData, cfg: dict, preds: pd.DataFrame, summary: pd.DataFrame,
                       best: pd.DataFrame, route: pd.DataFrame, out_dir: Path) -> dict:
    """Produce every figure / table of B6; returns paths and tables for the report."""
    figs = Path(out_dir) / "figures"
    tables = Path(out_dir) / "interpretation"
    tables.mkdir(exist_ok=True)
    res: dict = {"figures": {}, "lengthscales": {}, "loao_grades": pd.DataFrame(), "notes": []}

    def safe(name, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:                                     # noqa: BLE001
            log.warning("figure %s failed: %s", name, exc)
            res["notes"].append(f"{name} failed: {type(exc).__name__}: {exc}")
            return None

    icfg = cfg["interpret"]
    endpoint = [t for t in cfg["regimes"]["endpoint"] if t in set(summary["target"])]
    res["figures"]["parity"] = safe("parity", parity_figure, preds, best, figs / "parity_best.png")
    for t in endpoint:
        for fs in cfg["features"]["use"]:
            est, cols = safe(f"gpr {t} {fs}", fit_full_gpr, data, t, fs, cfg) or (None, None)
            if est is None:
                continue
            res["lengthscales"][(t, fs)] = est.length_scales_
            if fs == "base":
                res["figures"][f"gpr_surface_{t}"] = safe(
                    f"gpr surface {t}", gpr_surface, est, cols, data, t, int(icfg.get("map_points", 41)),
                    figs / f"gpr_surface_{t}.png")
    res["figures"]["gpr_lengthscales"] = safe("length scales", lengthscale_figure,
                                              res["lengthscales"], figs / "gpr_lengthscales.png")
    if res["lengthscales"]:
        pd.DataFrame([{"target": t, "feature_set": fs, **ls} for (t, fs), ls in
                      res["lengthscales"].items()]).to_csv(tables / "gpr_lengthscales.csv", index=False)
    r = safe("loao", loao_figure, preds, summary, endpoint, figs / "loao_per_grade.png", cfg["seed"])
    if r:
        res["figures"]["loao_per_grade"], res["loao_grades"] = r
        r[1].to_csv(tables / "loao_per_grade.csv", index=False)
    res["figures"]["trajectory"] = safe("trajectory", trajectory_figure, data, route, preds, best,
                                        figs / "trajectory_reconstruction.png")
    r = safe("calibration", calibration_figure, preds, summary, int(icfg.get("calibration_levels", 19)),
             figs / "gpr_calibration.png")
    if r:
        res["figures"]["gpr_calibration"] = r[0]
        r[1].to_csv(tables / "gpr_calibration.csv", index=False)
    res["figures"] = {k: v for k, v in res["figures"].items() if v}
    return res
