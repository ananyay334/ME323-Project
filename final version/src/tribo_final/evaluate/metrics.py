"""Metrics on the target's modelling scale (pooled out-of-fold predictions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def r2(y: np.ndarray, p: np.ndarray) -> float:
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - float(np.sum((y - p) ** 2)) / sst if sst > 0 else np.nan


def tercile_classes(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin measured and predicted values by the terciles of the MEASURED values."""
    edges = np.quantile(y, [1 / 3, 2 / 3])
    return np.digitize(y, edges), np.digitize(p, edges), edges


def tercile_confusion(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float, float]:
    cy, cp, _ = tercile_classes(y, p)
    cm = confusion_matrix(cy, cp, labels=[0, 1, 2])
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else np.nan
    kappa = float(cohen_kappa_score(cy, cp, labels=[0, 1, 2])) if len(np.unique(np.r_[cy, cp])) > 1 else 0.0
    return cm, acc, kappa


def compute_metrics(y, p, sd=None, rep_sd: float = np.nan, z: float = 1.959964) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    n = len(y)
    out = {
        "n": n,
        "R2": r2(y, p) if n > 1 else np.nan,
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "max_abs_err": float(np.max(np.abs(e))),
        "bias": float(np.mean(e)),
    }
    if n > 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
        out["pearson_r"] = float(stats.pearsonr(y, p)[0])
        out["spearman_rho"] = float(stats.spearmanr(y, p)[0])
        out["calib_slope"] = float(np.cov(y, p)[0, 1] / np.var(p, ddof=1))  # measured on predicted
    else:
        out.update(pearson_r=np.nan, spearman_rho=np.nan, calib_slope=np.nan)
    if np.isfinite(rep_sd) and rep_sd > 0:
        out["within_1sd_pct"] = 100 * float(np.mean(np.abs(e) <= rep_sd))
        out["within_2sd_pct"] = 100 * float(np.mean(np.abs(e) <= 2 * rep_sd))
    else:
        out["within_1sd_pct"] = out["within_2sd_pct"] = np.nan
    if sd is not None and np.all(np.isfinite(np.asarray(sd, float))):
        sd = np.asarray(sd, float)
        out["coverage95_pct"] = 100 * float(np.mean(np.abs(e) <= z * sd))
        out["interval_width"] = float(np.mean(2 * z * sd))
    else:
        out["coverage95_pct"] = out["interval_width"] = np.nan
    if n >= 6:
        _, acc, kappa = tercile_confusion(y, p)
        out["tercile_acc"] = acc
        out["tercile_kappa"] = kappa
    else:
        out["tercile_acc"] = out["tercile_kappa"] = np.nan
    return out


def metrics_table(oof: pd.DataFrame, rep_sd: dict, by=("target", "protocol", "model")) -> pd.DataFrame:
    rows = []
    for keys, g in oof.groupby(list(by), sort=False):
        sd = g["sd"].to_numpy() if g["sd"].notna().all() else None
        m = compute_metrics(g["y"].to_numpy(), g["pred"].to_numpy(), sd, rep_sd.get(keys[0], np.nan))
        rows.append({**dict(zip(by, keys)), **m})
    return pd.DataFrame(rows)


def one_se_tie(se_model: np.ndarray, se_best: np.ndarray) -> tuple[bool, float, float]:
    """1-SE rule on per-run squared errors: tied if mean(d) <= SE(d), d = se_model - se_best."""
    d = np.asarray(se_model) - np.asarray(se_best)
    if np.allclose(d, 0):
        return True, 0.0, 0.0
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    return bool(d.mean() <= se), float(d.mean()), se
