"""Nested group cross-validation on the training rows, model selection and refits.

Outer folds come from one :class:`ml.splits.SplitPlan` (5-fold GroupKFold, groups shuffled
by seed, one repeat per seed), so every model and target sees the same runs in each test
fold. Hyperparameters are chosen by an inner GroupKFold on the outer training groups
(:func:`ml.tuning.tune_and_fit`); scaling happens inside the estimators (training rows only).
Single-output models are fitted per target on the rows that have it; the ANN is fitted once
on both targets (masked). In the first repeat each task also computes the group-permutation
importance of every input on its held-out fold. Model selection: lowest mean CV RMSE per
target among selectable models (Ridge is a reference).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import patch_loky_tracker_handoff, stable_seed
from .data import INPUTS, TriboData
from .metrics import score_predictions, summarize
from .models.registry import ModelSpec, get_spec
from .splits import SplitPlan, check_disjoint
from .tuning import _rmse, grid_candidates, tune_and_fit

log = logging.getLogger("ml.evaluate")

KEYS = ["target", "model"]
META_COLS = ["Experiment_ID", "group", "Role", "Checkpoint", "Time_s"]
COST = {"ann": 40.0, "tabpfn": 20.0, "gpr": 4.0}


@dataclass
class CVResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        if self.fold_metrics.empty:
            return pd.DataFrame()
        s = summarize(self.fold_metrics, KEYS)
        if not self.predictions.empty:
            s = s.merge(pooled_metrics(self.predictions), on=KEYS, how="left")
        return s.sort_values(["target", "RMSE_mean"]).reset_index(drop=True)


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """R² and RMSE of all out-of-fold predictions of a repeat together, averaged over repeats.

    With 3-4 run groups per outer fold, the fold-wise R² is dominated by the small spread of y
    inside each fold (it is often strongly negative even for a useful model); the pooled R² compares
    the errors with the spread of the whole training set and is the number to read.
    """
    p = pred[~pred["censored"].astype(bool) & np.isfinite(pred["y_true"])]
    rows = []
    for (t, m), g in p.groupby(KEYS):
        r2, rmse = [], []
        for _, gr in g.groupby("repeat"):
            e = (gr["y_pred"] - gr["y_true"]).to_numpy(float)
            ss = float(((gr["y_true"] - gr["y_true"].mean()) ** 2).sum())
            rmse.append(float(np.sqrt(np.mean(e ** 2))))
            r2.append(1.0 - float((e ** 2).sum()) / ss if ss > 0 else np.nan)
        rows.append({"target": t, "model": m, "R2_pooled": float(np.nanmean(r2)) if r2 else np.nan,
                     "RMSE_pooled": float(np.mean(rmse)) if rmse else np.nan})
    return pd.DataFrame(rows, columns=["target", "model", "R2_pooled", "RMSE_pooled"])


def run_parallel(func, jobs: list[tuple], n_jobs: int, label: str) -> list:
    """Run ``func(*args)`` for every job, logging progress; result order is irrelevant."""
    if not jobs:
        return []
    patch_loky_tracker_handoff()
    out, t0, last, step = [], time.perf_counter(), time.perf_counter(), max(len(jobs) // 10, 1)
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(func)(*a) for a in jobs) if n_jobs != 1 else (func(*a) for a in jobs)
    for i, r in enumerate(gen, 1):
        out.append(r)
        now = time.perf_counter()
        if i % step == 0 or i == len(jobs) or now - last > 60:
            log.info("  %s: %d/%d tasks (%.0f s)", label, i, len(jobs), now - t0)
            last = now
    return out


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def group_permutation_importance(est, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, cp: np.ndarray,
                                 n_repeats: int, seed: int) -> np.ndarray:
    """Increase in RMSE (per input x target) when an input is permuted *between groups*.

    Each held-out row takes the input value of the row with the same checkpoint in the group
    its own group is mapped to, so run- and replicate-level structure is kept.
    """
    Y = _as2d(Y)
    cols = list(INPUTS)
    base = _as2d(est.predict(pd.DataFrame(X, columns=cols)))
    b0 = np.array([_rmse(Y[:, k], base[:, k]) for k in range(Y.shape[1])])
    ug = np.unique(groups)
    out = np.zeros((X.shape[1], Y.shape[1]))
    if len(ug) < 2 or n_repeats < 1:
        return out
    rng = np.random.default_rng(seed)
    first = {}
    for i, (g, c) in enumerate(zip(groups, cp)):
        first.setdefault((g, c), i)
        first.setdefault((g, None), i)
    for j in range(X.shape[1]):
        acc = np.zeros(Y.shape[1])
        for _ in range(n_repeats):
            m = dict(zip(ug, rng.permutation(ug)))
            src = [first.get((m[g], c), first[(m[g], None)]) for g, c in zip(groups, cp)]
            Xp = X.copy()
            Xp[:, j] = X[src, j]
            p = _as2d(est.predict(pd.DataFrame(Xp, columns=cols)))
            acc += np.array([_rmse(Y[:, k], p[:, k]) for k in range(Y.shape[1])]) - b0
        out[j] = acc / n_repeats
    return out


def _cv_task(key: dict, X: np.ndarray, Y: np.ndarray, groups: np.ndarray, cp: np.ndarray,
             tr: np.ndarray, te: np.ndarray, inner: list, mcfg: dict, seed: int, first_only: bool,
             n_perm: int) -> dict:
    """Tune on the inner folds, fit on the outer-train rows, predict the outer-test rows."""
    spec = get_spec(key["model"])
    t0 = time.perf_counter()
    cols = list(INPUTS)
    try:
        y = Y if spec.multi_output else Y[:, key["target_idx"]]
        est, best, score = tune_and_fit(spec, mcfg, pd.DataFrame(X[tr], columns=cols), y[tr], inner, seed,
                                        {}, first_only, groups=groups[tr])
        Xte = pd.DataFrame(X[te], columns=cols)
        yhat = _as2d(est.predict(Xte))
        ystd = _as2d(est.predict_std(Xte)) if spec.has_std else None
        imp = group_permutation_importance(est, X[te], y[te], groups[te], cp[te], n_perm, seed) \
            if key["repeat"] == 0 and n_perm else None
        return {**key, "test_idx": te, "yhat": yhat, "ystd": ystd, "params": best, "inner_rmse": score,
                "importance": imp, "seconds": time.perf_counter() - t0}
    except Exception as exc:                                         # noqa: BLE001
        return {**key, "error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}


def task_blocks(spec: ModelSpec, data: TriboData) -> list[tuple[int | None, str, np.ndarray]]:
    """(target index or None, target label, row indices) to fit for a model."""
    Y = data.Y
    if spec.multi_output:
        return [(None, "joint", np.flatnonzero(np.isfinite(Y).any(axis=1)))]
    return [(j, t, np.flatnonzero(np.isfinite(Y[:, j]))) for j, t in enumerate(data.targets)]


def cross_validate(data: TriboData, specs: list[ModelSpec], cfg: dict, plan: SplitPlan,
                   n_jobs: int = 1, label: str = "CV") -> CVResult:
    """Nested group CV of every model (per target, or jointly for the ANN) on every repeat."""
    first_only = bool(cfg.get("grid_first_only", False))
    n_perm = int(cfg["interpret"].get("permutation_repeats", 10)) if cfg["interpret"].get("enabled", True) else 0
    X, Y, groups = data.X.to_numpy(float), data.Y, data.groups
    cp = data.frame["Checkpoint"].to_numpy()
    jobs, cost = [], []
    for spec in specs:
        mcfg = cfg["models"].get(spec.name) or {}
        n_cand = len(grid_candidates(mcfg.get("grid"), first_only))
        for tidx, tname, rows in task_blocks(spec, data):
            for r in range(len(plan.seeds)):
                for f, a, b in plan.outer(groups[rows], r):
                    tr, te = rows[a], rows[b]
                    check_disjoint(groups, tr, te)
                    inner = plan.inner(groups[tr])
                    for ia, ib in inner:
                        check_disjoint(groups[tr], ia, ib)
                    key = {"model": spec.name, "target_idx": tidx, "target": tname, "repeat": r, "fold": f}
                    seed = stable_seed(cfg["seed"], spec.name, tname, r, f)
                    jobs.append((key, X, Y, groups, cp, tr, te, inner, mcfg, seed, first_only, n_perm))
                    cost.append(COST.get(spec.name, 1.0) * (1 + (n_cand * len(inner) if n_cand > 1 else 0)))
    jobs = [jobs[i] for i in np.argsort(cost, kind="stable")[::-1]]   # longest first
    log.info("%s: %d fit tasks on %s worker(s)", label, len(jobs), n_jobs)
    results = run_parallel(_cv_task, jobs, n_jobs, label)
    return assemble(data, results)


def assemble(data: TriboData, results: list[dict]) -> CVResult:
    preds, params, imps, failures = [], [], [], []
    meta = data.frame[[c for c in META_COLS if c in data.frame]].reset_index(drop=True)
    for r in results:
        key = {k: r[k] for k in ("model", "target", "repeat", "fold")}
        if "error" in r:
            failures.append({**key, "error": r["error"]})
            log.warning("FAILED %s: %s", key, r["error"])
            continue
        tnames = data.targets if r["target_idx"] is None else [data.targets[r["target_idx"]]]
        tcols = range(len(data.targets)) if r["target_idx"] is None else [r["target_idx"]]
        for k, (tname, tc) in enumerate(zip(tnames, tcols)):
            p = meta.iloc[r["test_idx"]].reset_index(drop=True)
            p["target"], p["model"], p["repeat"], p["fold"] = tname, r["model"], r["repeat"], r["fold"]
            p["y_true"] = data.Y[r["test_idx"], tc]
            p["y_pred"] = r["yhat"][:, k]
            p["y_std"] = r["ystd"][:, k] if r["ystd"] is not None else np.nan
            p["censored"] = data.censored[r["test_idx"], tc]
            preds.append(p[np.isfinite(p["y_true"])])
            if r.get("importance") is not None:
                for j, feat in enumerate(INPUTS):
                    imps.append({"model": r["model"], "target": tname, "feature": feat,
                                 "repeat": r["repeat"], "fold": r["fold"],
                                 "delta_rmse": float(r["importance"][j, k])})
        params.append({**key, "params": json.dumps(r["params"], default=str),
                       "inner_rmse": r.get("inner_rmse"), "seconds": r["seconds"]})
    res = CVResult(failures=failures)
    order = KEYS + ["repeat", "fold"]
    if preds:
        res.predictions = pd.concat(preds, ignore_index=True)
        res.fold_metrics = score_predictions(res.predictions, order).sort_values(order).reset_index(drop=True)
    if params:
        res.params = pd.DataFrame(params).sort_values(["model", "target", "repeat", "fold"]).reset_index(drop=True)
    if imps:
        res.importance = pd.DataFrame(imps)
    return res


def select_best(summary: pd.DataFrame) -> pd.DataFrame:
    """Best model per target: lowest mean CV RMSE, selectable models only (Ridge excluded)."""
    from .models.registry import registry
    if summary.empty:
        return pd.DataFrame(columns=["target", "model", "R2_mean", "RMSE_mean"])
    reg = registry()
    s = summary[summary["model"].map(lambda m: m in reg and reg[m].selectable)]
    best = s.sort_values("RMSE_mean").groupby("target", sort=False).head(1)
    return best[["target", "model", "R2_mean", "RMSE_mean"]].sort_values("target").reset_index(drop=True)


def fit_full(spec: ModelSpec, data: TriboData, cfg: dict, plan: SplitPlan, target: str | None = None,
             seed: int | None = None):
    """Tune (inner GroupKFold on all training groups) and fit one model on all training rows.

    ``target`` = a target name for single-output models; ignored for the ANN (both targets).
    """
    mcfg = cfg["models"].get(spec.name) or {}
    j = None if spec.multi_output else data.targets.index(target)
    Y = data.Y if spec.multi_output else data.Y[:, j]
    rows = np.flatnonzero(np.isfinite(_as2d(Y)).any(axis=1))
    groups = data.groups[rows]
    seed = stable_seed(cfg["seed"], spec.name, "full", target if j is not None else "joint") if seed is None else seed
    est, best, _ = tune_and_fit(spec, mcfg, data.X.iloc[rows].reset_index(drop=True), Y[rows],
                                plan.inner(groups), seed, {}, bool(cfg.get("grid_first_only", False)),
                                groups=groups)
    return est, best


def predict_target(est, spec: ModelSpec, X: pd.DataFrame, targets: list[str], target: str):
    """(mean, std or None) of ``target`` from a fitted model (picks the ANN's column)."""
    mu = _as2d(est.predict(X))
    sd = _as2d(est.predict_std(X)) if spec.has_std else None
    k = targets.index(target) if spec.multi_output else 0
    return mu[:, k], (sd[:, k] if sd is not None else None)
