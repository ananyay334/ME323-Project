"""Search spaces and inner-CV hyperparameter tuning (the inner loop of nested CV)."""
from __future__ import annotations

import itertools
import logging
import math
import time
import warnings

import numpy as np
import pandas as pd

from .models.registry import ModelSpec, build_estimator

log = logging.getLogger("ml.tuning")


def _is_dist(v) -> bool:
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("loguniform", "uniform", "randint")


def sample_candidates(space: dict | None, n_iter: int, seed: int) -> list[dict]:
    """Candidate hyperparameter dicts from a config search space.

    A list is a categorical choice; ``{loguniform|uniform|randint: [lo, hi]}`` a distribution.
    If the space is purely categorical and its full grid has at most ``n_iter`` points the
    whole grid is returned (in a fixed order); otherwise ``n_iter`` random draws (deduplicated).
    """
    space = space or {}
    if not space:
        return [{}]
    keys = sorted(space)
    if not any(_is_dist(space[k]) for k in keys):
        grid = list(itertools.product(*[space[k] for k in keys]))
        if len(grid) <= n_iter:
            return [dict(zip(keys, combo)) for combo in grid]
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    for _ in range(n_iter * 4):
        c = {}
        for k in keys:
            v = space[k]
            if _is_dist(v):
                kind, (lo, hi) = next(iter(v.items()))
                if kind == "loguniform":
                    c[k] = float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))
                elif kind == "uniform":
                    c[k] = float(rng.uniform(lo, hi))
                else:
                    c[k] = int(rng.integers(lo, hi + 1))
            else:
                c[k] = v[int(rng.integers(len(v)))]
        sig = repr(sorted(c.items()))
        if sig not in seen:
            seen.add(sig)
            out.append(c)
        if len(out) == n_iter:
            break
    return out


def fit_estimator(spec: ModelSpec, est, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if spec.needs_groups:
            return est.fit(X, y, groups=groups)
        return est.fit(X, y)


def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def tune_and_fit(spec: ModelSpec, mcfg: dict, X: pd.DataFrame, y: np.ndarray,
                 groups: np.ndarray | None, inner: list[tuple[np.ndarray, np.ndarray]],
                 seed: int, ctx: dict, n_iter: int, cand_seed: int):
    """Pick hyperparameters by inner group-CV RMSE, then refit on all of (X, y).

    ``inner`` holds index pairs relative to the rows of ``X`` (group-disjoint).
    Returns ``(fitted_estimator, best_params, info)``.
    """
    fixed = dict(mcfg.get("params") or {})
    cands = sample_candidates(mcfg.get("search"), n_iter, cand_seed)
    best, scores = cands[0], []
    t0 = time.perf_counter()
    if len(cands) > 1 and inner:
        tune_fixed = dict(fixed)
        if "tune_n_seeds" in mcfg and "n_seeds" in fixed:
            tune_fixed["n_seeds"] = mcfg["tune_n_seeds"]
        for c in cands:
            errs = []
            for tr, va in inner:
                try:
                    est = build_estimator(spec, {**tune_fixed, **c}, seed, ctx)
                    fit_estimator(spec, est, X.iloc[tr], y[tr], None if groups is None else groups[tr])
                    errs.append(_rmse(y[va], est.predict(X.iloc[va])))
                except Exception as exc:                     # noqa: BLE001
                    log.debug("candidate %s failed: %s", c, exc)
                    errs.append(np.inf)
            scores.append(float(np.mean(errs)))
        best = cands[int(np.argmin(scores))]
    est = build_estimator(spec, {**fixed, **best}, seed, ctx)
    fit_estimator(spec, est, X, y, groups)
    info = {"inner_rmse": float(np.min(scores)) if scores else np.nan,
            "n_candidates": len(cands), "fit_seconds": time.perf_counter() - t0}
    return est, best, info
