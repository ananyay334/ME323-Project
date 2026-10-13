"""Nested group cross-validation over gkf5 (repeated) and leave-one-alloy-out.

Outer folds come from one :class:`ml.splits.SplitPlan`, so every regime, target, feature
set and model sees the same runs in each test fold. Hyperparameters are chosen by an
inner GroupKFold on the outer training groups (:func:`ml.tuning.tune_and_fit`); all
scaling / imputation is fitted inside the estimators, i.e. on training rows only.
Model selection: lowest mean gkf5 RMSE among selectable models (Ridge is a reference).
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
from .data import TARGETS, TriboData, target_frame
from .features import build_X, check_inputs, resolve_feature_set, time_columns
from .metrics import score_predictions, summarize
from .models.registry import ModelSpec, get_spec
from .splits import SplitPlan, check_disjoint
from .tuning import tune_and_fit

log = logging.getLogger("ml.evaluate")

KEYS = ["regime", "target", "model", "feature_set", "split"]
HEAVY_GPR_ROWS = 400            # GPR fits on more training rows than this are chained
META_COLS = ["Experiment_ID", "group", "Specimen_ID", "Time_s", "censored", "y_lod"]


@dataclass
class Design:
    """Rows + model matrix for one (target, feature set, censoring mode)."""
    target: str
    regime: str
    feature_set: str
    censor_mode: str
    frame: pd.DataFrame          # meta columns + y
    X: pd.DataFrame
    info: dict

    @property
    def y(self) -> np.ndarray:
        return self.frame["y"].to_numpy(float)

    @property
    def groups(self) -> np.ndarray:
        return self.frame["group"].astype(str).to_numpy()

    @property
    def run_ids(self) -> np.ndarray:
        return self.frame["Experiment_ID"].astype(str).to_numpy()


def feature_columns(data: TriboData, regime: str, feature_set: str, cfg: dict) -> list[str]:
    cols = resolve_feature_set(feature_set, data, cfg) + time_columns(regime)
    check_inputs(cols, regime)
    return cols


def make_design(data: TriboData, target: str, feature_set: str, cfg: dict,
                censor_mode: str = "exclude") -> Design | None:
    """Design for ``target`` x ``feature_set`` (None if the target has no rows)."""
    spec = TARGETS[target]
    df, info = target_frame(data, spec, censor_mode)
    if df.empty:
        return None
    cols = feature_columns(data, spec.regime, feature_set, cfg)
    X = build_X(df, cols)
    meta = df[[c for c in META_COLS if c in df]].reset_index(drop=True).copy()
    meta["y"] = df["y"].to_numpy(float)
    return Design(target, spec.regime, feature_set, censor_mode, meta, X, info)


# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: list = field(default_factory=list)

    def extend(self, other: "CVResult") -> "CVResult":
        cat = lambda a, b: pd.concat([a, b], ignore_index=True) if len(b) else a   # noqa: E731
        return CVResult(cat(self.predictions, other.predictions),
                        cat(self.fold_metrics, other.fold_metrics), cat(self.params, other.params),
                        self.failures + other.failures)

    def summary(self) -> pd.DataFrame:
        if self.fold_metrics.empty:
            return pd.DataFrame()
        return summarize(self.fold_metrics, KEYS).sort_values(
            ["regime", "target", "split", "RMSE_mean"]).reset_index(drop=True)


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


def _cv_task(key: dict, X: np.ndarray, columns: list[str], y: np.ndarray, train_idx: np.ndarray,
             test_idx: np.ndarray, inner: list, mcfg: dict, seed: int, first_only: bool) -> dict:
    """Tune on the inner folds, fit on the outer-train rows, predict the outer-test rows."""
    spec = get_spec(key["model"])
    t0 = time.perf_counter()
    try:
        est, best, score = tune_and_fit(spec, mcfg, pd.DataFrame(X[train_idx], columns=columns),
                                        y[train_idx], inner, seed, {}, first_only)
        Xte = pd.DataFrame(X[test_idx], columns=columns)
        yhat = np.asarray(est.predict(Xte), float).ravel()
        ystd = np.asarray(est.predict_std(Xte), float).ravel() if spec.has_std else None
        return {**key, "test_idx": test_idx, "yhat": yhat, "ystd": ystd, "params": best,
                "inner_rmse": score, "seconds": time.perf_counter() - t0}
    except Exception as exc:                                         # noqa: BLE001
        return {**key, "error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}


def task_cost(model: str, n_rows: int, n_inner: int, mcfg: dict, first_only: bool) -> float:
    """Rough relative cost of one outer task (for longest-first scheduling)."""
    from .tuning import grid_candidates
    n_cand = len(grid_candidates(mcfg.get("grid"), first_only))
    fits = 1 + (n_cand * n_inner if n_cand > 1 else 0)
    if model == "gpr":                                   # O(n^3) per likelihood evaluation
        return fits * (1 + int((mcfg.get("params") or {}).get("n_restarts_optimizer", 0))) \
            * (n_rows / 100.0) ** 3
    return fits * n_rows / 100.0


def cross_validate(designs: list[Design], specs: list[ModelSpec], cfg: dict, plan: SplitPlan,
                   n_jobs: int = 1, label: str = "CV") -> CVResult:
    """Nested CV of every (design, model) pair on every split (gkf5 repeats + LOAO)."""
    first_only = bool(cfg.get("grid_first_only", False))
    jobs, cost = [], []
    for di, d in enumerate(designs):
        X, cols, groups = d.X.to_numpy(float), list(d.X.columns), d.groups
        for spec in specs:
            mcfg = cfg["models"].get(spec.name) or {}
            for split, r in plan.splits():
                for f, tr, te in plan.outer(d.run_ids, split, r):
                    check_disjoint(groups, tr, te)
                    inner = plan.inner(groups[tr])
                    for a, b in inner:
                        check_disjoint(groups[tr], a, b)
                    key = {"design": di, "regime": d.regime, "target": d.target, "model": spec.name,
                           "feature_set": d.feature_set, "split": split, "repeat": r, "fold": f}
                    seed = stable_seed(cfg["seed"], d.target, spec.name, d.feature_set, split, r, f)
                    jobs.append((key, X, cols, d.y, tr, te, inner, mcfg, seed, first_only))
                    cost.append(task_cost(spec.name, len(tr), len(inner), mcfg, first_only))
    jobs = [jobs[i] for i in np.argsort(cost, kind="stable")[::-1]]   # longest first
    # Large GPR fits are memory-bandwidth bound: chain them into a few sequential bundles so
    # that at most `heavy_task_slots` run at once; light tasks fill the other workers.
    heavy = [j for j in jobs if j[0]["model"] == "gpr" and len(j[4]) > HEAVY_GPR_ROWS]
    light = [(j,) for j in jobs if not (j[0]["model"] == "gpr" and len(j[4]) > HEAVY_GPR_ROWS)]
    slots = max(1, int(cfg["cv"].get("heavy_task_slots", 3)))
    bundles = [(heavy[i::slots],) for i in range(min(slots, len(heavy)))]
    log.info("%s: %d fit tasks (%d large GPR fits in %d chains) on %s worker(s)", label, len(jobs),
             len(heavy), len(bundles), n_jobs)
    results = run_parallel(_cv_bundle, bundles + light, n_jobs, label)
    return assemble(designs, [r for b in results for r in b], plan)


def _cv_bundle(tasks) -> list[dict]:
    """Run one task (tuple of args) or a chain of tasks (list) in one worker."""
    return [_cv_task(*t) for t in tasks] if isinstance(tasks, list) else [_cv_task(*tasks)]


def assemble(designs: list[Design], results: list[dict], plan: SplitPlan) -> CVResult:
    preds, params, failures = [], [], []
    order = KEYS + ["repeat", "fold"]
    for r in results:
        key = {k: r[k] for k in order}
        if "error" in r:
            failures.append({**key, "error": r["error"]})
            log.warning("FAILED %s: %s", key, r["error"])
            continue
        d = designs[r["design"]]
        p = d.frame.iloc[r["test_idx"]].reset_index(drop=True).rename(columns={"y": "y_true"})
        p["y_pred"] = r["yhat"]
        p["y_std"] = r["ystd"] if r["ystd"] is not None else np.nan
        for k, v in key.items():
            p[k] = v
        p["fold_label"] = plan.fold_label(key["split"], key["fold"])
        preds.append(p)
        params.append({**key, "params": json.dumps(r["params"], default=str),
                       "inner_rmse": r.get("inner_rmse"), "seconds": r["seconds"]})
    res = CVResult(failures=failures)
    if preds:
        res.predictions = pd.concat(preds, ignore_index=True)
        res.fold_metrics = score_predictions(res.predictions, order + ["fold_label"]) \
            .sort_values(order).reset_index(drop=True)
    if params:
        res.params = pd.DataFrame(params).sort_values(order).reset_index(drop=True)
    return res


def select_best(summary: pd.DataFrame) -> pd.DataFrame:
    """Best (model, feature set) per target: lowest gkf5 RMSE, selectable models only."""
    from .models.registry import registry
    if summary.empty:
        return pd.DataFrame(columns=["regime", "target", "model", "feature_set", "RMSE_mean"])
    reg = registry()
    s = summary[(summary["split"] == "gkf5")
                & summary["model"].map(lambda m: m in reg and reg[m].selectable)]
    best = s.sort_values("RMSE_mean").groupby("target", sort=False).head(1)
    return best[["regime", "target", "model", "feature_set", "R2_mean", "RMSE_mean"]] \
        .sort_values(["regime", "target"]).reset_index(drop=True)
