"""Regression and interval metrics, fold summaries and the replicate noise floor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


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


def interval_metrics(y: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                     level: float = 0.95) -> dict[str, float]:
    """Coverage and mean width of the central ``level`` Gaussian interval."""
    y, mu, sd = (np.asarray(a, float) for a in (y, mu, sd))
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    if not ok.any():
        return {"coverage95": np.nan, "width95": np.nan}
    z = float(norm.ppf(0.5 + level / 2))
    return {"coverage95": float(np.mean(np.abs(y[ok] - mu[ok]) <= z * sd[ok])),
            "width95": float(np.mean(2 * z * sd[ok]))}


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


def score_predictions(pred: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Metrics per block (e.g. target x model x feature set x split x repeat x fold).

    Censored rows (imputed at LOD/2 when ``censoring.mode = lod_half``) are not scored.
    """
    rows = []
    sub_all = pred[~pred["censored"].astype(bool)] if "censored" in pred else pred
    for k, sub in sub_all.groupby(keys, dropna=False, sort=False):
        m = regression_metrics(sub["y_true"], sub["y_pred"])
        if "y_std" in sub and np.isfinite(sub["y_std"].to_numpy(float)).any():
            m.update(interval_metrics(sub["y_true"], sub["y_pred"], sub["y_std"]))
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **m})
    return pd.DataFrame(rows)


def summarize(fold_metrics: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and std over outer folds (and repeats) per ``keys``."""
    metrics = [c for c in ["R2", "RMSE", "MAE", "coverage95", "width95"] if c in fold_metrics]
    g = fold_metrics.groupby(keys, dropna=False)
    out = pd.concat([g[metrics].mean().add_suffix("_mean"), g[metrics].std(ddof=1).add_suffix("_std"),
                     g.size().rename("n_folds")], axis=1).reset_index()
    return out[keys + [f"{m}_{s}" for m in metrics for s in ("mean", "std")] + ["n_folds"]]


def replicate_noise_floor(frame: pd.DataFrame, pairs: list[list[str]]) -> dict[str, float]:
    """Pooled SD of replicate pairs: sqrt(mean over pairs of (a - b)² / 2)."""
    y = frame.set_index("Experiment_ID")["y"]
    d2 = [(y[a] - y[b]) ** 2 / 2 for a, b in pairs if a in y.index and b in y.index]
    return {"noise_sd": float(np.sqrt(np.mean(d2))) if d2 else np.nan, "n_pairs": len(d2)}
