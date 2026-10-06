"""Regression and interval metrics, and run-level aggregation of COF(t) predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import COF_T_RUN, TARGETS

Z95 = float(norm.ppf(0.975))


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """R², RMSE, MAE (R² is NaN when y is constant)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0}
    res = y - yhat
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"R2": 1 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan,
            "RMSE": float(np.sqrt(np.mean(res ** 2))), "MAE": float(np.mean(np.abs(res))),
            "n": int(len(y))}


def linear_from_log10_metrics(y_log: np.ndarray, yhat_log: np.ndarray) -> dict[str, float]:
    """Errors of a log10-modelled target expressed in linear units (e.g. k in mm³/Nm)."""
    y_log, yhat_log = np.asarray(y_log, float), np.asarray(yhat_log, float)
    ok = np.isfinite(y_log) & np.isfinite(yhat_log)
    if not ok.any():
        return {"RMSE_lin": np.nan, "MAE_lin": np.nan, "MdAPE_lin_pct": np.nan, "factor_err": np.nan}
    y, yh = 10.0 ** y_log[ok], 10.0 ** yhat_log[ok]
    return {"RMSE_lin": float(np.sqrt(np.mean((y - yh) ** 2))),
            "MAE_lin": float(np.mean(np.abs(y - yh))),
            "MdAPE_lin_pct": float(100 * np.median(np.abs(yh / y - 1))),
            # typical multiplicative error: 10 ** mean|log10 error|
            "factor_err": float(10 ** np.mean(np.abs(y_log[ok] - yhat_log[ok])))}


def interval_metrics(y: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                     level: float = 0.95) -> dict[str, float]:
    """Coverage and mean width of the central ``level`` Gaussian interval, and mean NLL."""
    y, mu, sd = (np.asarray(a, float) for a in (y, mu, sd))
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    if not ok.any():
        return {"coverage95": np.nan, "width95": np.nan, "NLL": np.nan}
    z = float(norm.ppf(0.5 + level / 2))
    y, mu, sd = y[ok], mu[ok], sd[ok]
    return {"coverage95": float(np.mean(np.abs(y - mu) <= z * sd)),
            "width95": float(np.mean(2 * z * sd)),
            "NLL": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((y - mu) / sd) ** 2))}


def calibration_curve(y: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                      levels: np.ndarray | None = None) -> pd.DataFrame:
    """Observed vs nominal coverage of central Gaussian intervals."""
    levels = np.linspace(0.05, 0.95, 19) if levels is None else np.asarray(levels)
    y, mu, sd = (np.asarray(a, float) for a in (y, mu, sd))
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    if not ok.any():
        return pd.DataFrame({"nominal": levels, "observed": np.nan})
    z = np.abs(y[ok] - mu[ok]) / sd[ok]
    return pd.DataFrame({"nominal": levels,
                         "observed": [float(np.mean(z <= norm.ppf(0.5 + p / 2))) for p in levels]})


def aggregate_cof_to_run(pred: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-second COF(t) predictions -> run-level steady-state COF.

    For each run, measured and predicted COF are averaged over the same seconds: the
    steady phase as labelled by the extraction (``Phase == 'steady'``), or all seconds
    when no phase label is available.
    """
    p = pred.copy()
    steady = p["Phase"].astype(str).eq("steady") if "Phase" in p else pd.Series(True, index=p.index)
    has = steady.groupby(p["Experiment_ID"]).transform("any")
    p = p[steady | ~has]
    agg = {"y_true": ("y_true", "mean"), "y_pred": ("y_pred", "mean")}
    if "y_std" in p:
        agg["y_std"] = ("y_std", "mean")      # conservative: errors are autocorrelated in time
    out = p.groupby(keys + ["Experiment_ID"], dropna=False).agg(**agg).reset_index()
    out["target"] = COF_T_RUN
    return out


def score_frame(pred: pd.DataFrame, target: str) -> dict[str, float]:
    """All metrics for one block of predictions of one target."""
    m = regression_metrics(pred["y_true"], pred["y_pred"])
    if target in TARGETS and TARGETS[target].is_log10:
        m.update(linear_from_log10_metrics(pred["y_true"], pred["y_pred"]))
    if "y_std" in pred and np.isfinite(pred["y_std"].to_numpy(float)).any():
        m.update(interval_metrics(pred["y_true"], pred["y_pred"], pred["y_std"]))
    return m


def score_predictions(pred: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Metrics per block (e.g. target x model x feature set x repeat x fold).

    Censored rows (imputed at LOD/2 in the sensitivity run) are excluded from the
    scores so that both censoring modes are compared on the same detected runs.
    """
    rows = []
    sub_all = pred[~pred["censored"].astype(bool)] if "censored" in pred else pred
    for k, sub in sub_all.groupby(keys, dropna=False, sort=False):
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                     **score_frame(sub, sub["target"].iloc[0])})
    return pd.DataFrame(rows)


def summarize(fold_metrics: pd.DataFrame, keys: list[str],
              metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean and std over outer folds (and repeats) per ``keys``."""
    metrics = metrics or [c for c in ["R2", "RMSE", "MAE", "RMSE_lin", "MAE_lin", "MdAPE_lin_pct",
                                      "factor_err", "coverage95", "width95", "NLL"]
                          if c in fold_metrics]
    g = fold_metrics.groupby(keys, dropna=False)
    mean = g[metrics].mean().add_suffix("_mean")
    std = g[metrics].std(ddof=1).add_suffix("_std")
    n = g.size().rename("n_folds")
    out = pd.concat([mean, std, n], axis=1).reset_index()
    order = keys + [f"{m}_{s}" for m in metrics for s in ("mean", "std")] + ["n_folds"]
    return out[order]


def censored_consistency(pred: pd.DataFrame) -> float:
    """Fraction of censored runs whose prediction lies below their detection limit."""
    if "censored" not in pred or "y_lod" not in pred:
        return np.nan
    c = pred[pred["censored"].astype(bool) & np.isfinite(pred["y_lod"])]
    return float(np.mean(c["y_pred"] < c["y_lod"])) if len(c) else np.nan
