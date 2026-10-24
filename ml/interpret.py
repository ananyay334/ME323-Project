"""Report figures and the "which factor controls COF and wear" tables.

Accuracy statements use out-of-fold predictions only. GPR length scales, response surfaces
and the Ridge exponents come from fits on all training rows (description, not evaluation).

* parity (CV, first repeat; and the test set with ``--final``);
* GPR mean ± σ over ln L × ln f at t = 60 and 120 s, one figure per target;
* GPR ARD length scales (standardised inputs: small = strong / curved effect);
* group-permutation importance of the inputs (from the CV tasks, first repeat);
* Ridge exponents d target / d ln x with group-bootstrap 95 % intervals;
* per-cycle COF of the centre replicates R04/R14/R20.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

from . import plotting as P
from .data import INPUTS, TARGETS, TriboData, load_cof_cycles
from .evaluate import fit_full
from .models.registry import get_spec

log = logging.getLogger("ml.interpret")


@P.styled
def parity_figure(preds: pd.DataFrame, best: pd.DataFrame, out: Path, title: str = "") -> Path | None:
    """Predicted vs measured for the best model per target, coloured by checkpoint."""
    if best.empty:
        return None
    n = len(best)
    fig, ax = P.new_fig(1, n, w=3.6 * n, h=3.7)
    for k, b in enumerate(best.itertuples()):
        a = ax[0, k]
        p = preds[(preds["target"] == b.target) & (preds["model"] == b.model)]
        p = p[~p["censored"].astype(bool)]
        lo = np.nanmin([p["y_true"].min(), p["y_pred"].min()])
        hi = np.nanmax([p["y_true"].max(), p["y_pred"].max()])
        for i, cp in enumerate(sorted(p["Checkpoint"].unique())):
            q = p[p["Checkpoint"] == cp]
            P.parity(a, q["y_true"], q["y_pred"], P.SLOTS[i], label=f"checkpoint {int(cp)}", lo=lo, hi=hi)
        a.set_title(f"{b.target}: {b.model}", fontsize=9.5)
        a.text(0.03, 0.97, f"R² = {b.R2_mean:.2f}", transform=a.transAxes, va="top", color=P.INK2, fontsize=8.5)
        a.set_xlabel("measured")
        a.set_ylabel("predicted" if k == 0 else "")
        a.legend(loc="lower right", fontsize=7.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=P.INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    return P.save(fig, out)


@P.styled
def gpr_surface(est, data: TriboData, target: str, times: list[float], n: int, out: Path) -> Path:
    """GPR mean (top) and model σ (bottom) over load x frequency at fixed checkpoint times."""
    L = np.geomspace(10, 100, n)
    F = np.geomspace(4, 15, n)
    LL, FF = np.meshgrid(L, F)
    mus, sds = [], []
    for t in times:
        X = pd.DataFrame({"ln_L": np.log(LL.ravel()), "ln_f": np.log(FF.ravel()), "ln_t": np.log(t)})
        mus.append(est.predict(X).reshape(LL.shape))
        sds.append(est.predict_latent_std(X).reshape(LL.shape))
    fig, ax = P.new_fig(2, len(times), w=3.4 * len(times) + 0.9, h=6.0)
    runs = data.frame.drop_duplicates("Experiment_ID")
    for row, (vals, cmap, lab) in enumerate(((mus, P.CMAP_MEAN, f"mean {TARGETS[target]}"),
                                             (sds, P.CMAP_SD, "σ (model, excl. noise)"))):
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        for k, t in enumerate(times):
            a = ax[row, k]
            cs = a.contourf(LL, FF, vals[k], levels=P.nice_levels(lo, hi), cmap=cmap)
            a.set_xscale("log")
            a.set_yscale("log")
            for axis, ticks in ((a.xaxis, [10, 20, 30, 50, 100]), (a.yaxis, [4, 6, 8, 10, 15])):
                axis.set_major_locator(FixedLocator(ticks))
                axis.set_minor_locator(NullLocator())
                axis.set_major_formatter(FormatStrFormatter("%g"))
            a.grid(False)
            P.dots(a, runs["Load_N"], runs["Freq_Hz"], P.INK2, size=20)
            a.set_title(f"t = {t:g} s" if row == 0 else "", fontsize=9.5)
            a.set_xlabel("load [N]" if row == 1 else "")
            a.set_ylabel("frequency [Hz]" if k == 0 else "")
        cb = fig.colorbar(cs, ax=list(ax[row, :]), fraction=0.03, pad=0.02)
        cb.set_label(lab, color=P.INK2)
        cb.outline.set_visible(False)
    fig.suptitle(f"GPR, {target}: mean (top) ± σ (bottom) over ln L × ln f; dots = training runs "
                 f"(fitted noise sd {np.sqrt(est.noise_var_):.3g})", x=0.01, ha="left", color=P.INK, fontsize=10)
    return P.save(fig, out)


@P.styled
def cof_replicates_figure(cycles: pd.DataFrame, out: Path) -> Path | None:
    """Per-cycle COF vs sliding time for the centre replicates."""
    if cycles is None or cycles.empty:
        return None
    fig, ax = P.new_fig(w=7.5, h=3.4)
    a = ax[0, 0]
    for i, (eid, c) in enumerate(cycles.groupby("Experiment_ID")):
        for s, cs in c.groupby("Step"):
            a.plot(cs["t_s"], cs["COF_c"], color=P.SLOTS[i % 8], lw=1.0, label=eid if s == c["Step"].min() else None)
    a.set_xlabel("sliding time [s]")
    a.set_ylabel("COF per cycle")
    a.set_title("Centre replicates: per-cycle COF")
    a.legend(loc="upper right")
    return P.save(fig, out)


def ridge_exponents(data: TriboData, cfg: dict, plan, seed: int) -> pd.DataFrame:
    """d target / d ln x of Ridge on standardised log inputs, with group-bootstrap 95 % CIs."""
    spec = get_spec("ridge")
    B = int(cfg["interpret"].get("bootstrap", 1000))
    rng = np.random.default_rng(seed)
    rows = []
    for j, t in enumerate(data.targets):
        est, params = fit_full(spec, data, cfg, plan, t)
        alpha = float(params.get("alpha", (cfg["models"]["ridge"].get("params") or {}).get("alpha", 1.0)))
        ok = np.isfinite(data.Y[:, j])
        X, y, g = data.X.to_numpy(float)[ok], data.Y[ok, j], data.groups[ok]
        point = _ridge_slopes(X, y, alpha)
        ug = np.unique(g)
        idx = {u: np.flatnonzero(g == u) for u in ug}
        boot = np.array([_ridge_slopes(X[s], y[s], alpha) for s in
                         (np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]) for _ in range(B))])
        for k, f in enumerate(INPUTS):
            rows.append({"target": t, "feature": f, "alpha": alpha, "exponent": point[k],
                         "ci_low": float(np.percentile(boot[:, k], 2.5)),
                         "ci_high": float(np.percentile(boot[:, k], 97.5)), "n_boot": B})
    return pd.DataFrame(rows)


def _ridge_slopes(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Same fit as StandardScaler + Ridge(alpha); coefficients mapped back to per unit ln x."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return b / sd


def run_interpretation(data: TriboData, cfg: dict, cv, summary: pd.DataFrame, best: pd.DataFrame,
                       plan, out_dir: Path, source: str) -> dict:
    """Every figure / table for the report; returns paths and tables."""
    figs = Path(out_dir) / "figures"
    tables = Path(out_dir) / "interpretation"
    tables.mkdir(exist_ok=True)
    res: dict = {"figures": {}, "lengthscales": pd.DataFrame(), "importance": pd.DataFrame(),
                 "exponents": pd.DataFrame(), "notes": []}

    def safe(name, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:                                     # noqa: BLE001
            log.warning("%s failed: %s", name, exc)
            res["notes"].append(f"{name} failed: {type(exc).__name__}: {exc}")
            return None

    icfg = cfg["interpret"]
    preds0 = cv.predictions[cv.predictions["repeat"] == 0]
    res["figures"]["parity"] = safe("parity", parity_figure, preds0, best, figs / "parity_cv.png",
                                    "Out-of-fold predictions (first repeat)")
    ls_rows = []
    gpr = get_spec("gpr")
    for t in data.targets:
        r = safe(f"gpr {t}", fit_full, gpr, data, cfg, plan, t)
        if r is None:
            continue
        est = r[0]
        ls_rows.append({"target": t, **est.length_scales_})
        res["figures"][f"gpr_surface_{t}"] = safe(
            f"gpr surface {t}", gpr_surface, est, data, t, list(icfg.get("surface_times_s", [60, 120])),
            int(icfg.get("map_points", 41)), figs / f"gpr_surface_{t}.png")
    res["lengthscales"] = pd.DataFrame(ls_rows)
    if len(cv.importance):
        res["importance"] = (cv.importance.groupby(["model", "target", "feature"])["delta_rmse"]
                             .agg(["mean", "std"]).reset_index())
    if "ridge" in set(summary["model"]):
        res["exponents"] = safe("ridge exponents", ridge_exponents, data, cfg, plan, cfg["seed"])
    cyc = load_cof_cycles(cfg, source, cfg["data"]["replicates"])
    res["figures"]["cof_replicates"] = safe("cof replicates", cof_replicates_figure, cyc,
                                            figs / "cof_replicates.png")
    for name in ("lengthscales", "importance", "exponents"):
        if res[name] is not None and len(res[name]):
            res[name].to_csv(tables / f"{name}.csv", index=False)
    res["figures"] = {k: v for k, v in res["figures"].items() if v}
    return res
