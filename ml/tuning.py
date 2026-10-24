"""Fixed search grids and inner-CV tuning (the inner loop of nested CV)."""
from __future__ import annotations

import inspect
import itertools
import logging
import warnings

import numpy as np
import pandas as pd

from .models.registry import ModelSpec, build_estimator

log = logging.getLogger("ml.tuning")


def grid_candidates(grid: dict | None, first_only: bool = False) -> list[dict]:
    """All combinations of a ``{param: [values]}`` grid (``--quick``: first value of each list)."""
    grid = grid or {}
    if not grid:
        return [{}]
    keys = sorted(grid)
    if first_only:
        return [{k: list(grid[k])[0] for k in keys}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*[list(grid[k]) for k in keys])]


def _rmse(a, b, scale=None) -> float:
    """RMSE over finite pairs; for (n, k) targets the mean of the per-column RMSE / ``scale``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim == 1:
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else np.nan
    cols = [_rmse(a[:, j], b[:, j]) / (scale[j] if scale is not None else 1.0) for j in range(a.shape[1])]
    return float(np.nanmean(cols))


def fit_estimator(est, X: pd.DataFrame, y: np.ndarray, groups=None):
    """``est.fit(X, y)``, passing ``groups`` to estimators whose fit takes them (the ANN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if groups is not None and "groups" in inspect.signature(est.fit).parameters:
            return est.fit(X, y, groups=groups)
        return est.fit(X, y)


def inner_score(spec: ModelSpec, params: dict, X: pd.DataFrame, y: np.ndarray,
                inner: list[tuple[np.ndarray, np.ndarray]], seed: int, ctx: dict,
                groups=None) -> float:
    """Mean RMSE over the inner folds for one hyperparameter set (multi-output: RMSE / SD)."""
    errs = []
    scale = np.nanstd(y, axis=0) if np.ndim(y) == 2 else None
    for tr, va in inner:
        try:
            est = build_estimator(spec, params, seed, ctx)
            fit_estimator(est, X.iloc[tr], y[tr], None if groups is None else np.asarray(groups)[tr])
            errs.append(_rmse(y[va], est.predict(X.iloc[va]), scale))
        except Exception as exc:                                     # noqa: BLE001
            log.debug("candidate %s failed: %s", params, exc)
            errs.append(np.inf)
    return float(np.mean(errs)) if errs else np.nan


def tune_and_fit(spec: ModelSpec, mcfg: dict, X: pd.DataFrame, y: np.ndarray,
                 inner: list[tuple[np.ndarray, np.ndarray]], seed: int, ctx: dict,
                 first_only: bool = False, always_score: bool = False, groups=None):
    """Pick hyperparameters by inner group-CV RMSE, then refit on all of (X, y).

    ``inner`` holds group-disjoint index pairs relative to the rows of ``X``. With a single
    candidate no inner fit is needed unless ``always_score`` (used to compare models).
    Returns ``(fitted_estimator, best_params, inner_rmse)``.
    """
    fixed = dict(mcfg.get("params") or {})
    cands = grid_candidates(mcfg.get("grid"), first_only)
    best, best_score = cands[0], np.nan
    if inner and (len(cands) > 1 or always_score):
        scores = [inner_score(spec, {**fixed, **c}, X, y, inner, seed, ctx, groups) for c in cands]
        i = int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))
        best, best_score = cands[i], scores[i]
    est = build_estimator(spec, {**fixed, **best}, seed, ctx)
    fit_estimator(est, X, y, groups)
    return est, best, best_score
