"""Nested group cross-validation, joint (multi-task) CV and the final refit / unseen-point test.

Outer loop: group K-fold (repeated) from a single :class:`ml.splits.FoldPlan`, so every
target, feature set and model sees the same runs in each test fold.
Inner loop: group K-fold inside the outer training set, used only to choose
hyperparameters (:func:`ml.tuning.tune_and_fit`). All preprocessing is fitted inside the
estimators, i.e. on training rows only.

COF(t) models are fitted on a thinned set of seconds per run (``time_regime``) but scored
on every second of the test runs, and also on the run-level steady-state COF obtained by
averaging the predicted curve over the steady phase (target ``COF_t@ss``).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from .config import patch_loky_tracker_handoff, stable_seed
from .data import COF_T_RUN, TARGETS, TriboData, target_frame
from .features import build_X, check_inputs, resolve_feature_set, time_columns
from .metrics import aggregate_cof_to_run, score_predictions, summarize
from .models.registry import ModelSpec, get_spec
from .splits import FoldPlan, check_disjoint
from .tuning import tune_and_fit

log = logging.getLogger("ml.evaluate")

KEYS = ["target", "model", "feature_set", "censor_mode"]
META_COLS = ["Experiment_ID", "group", "Time_s", "Phase", "censored", "y_lod"]
_COST = {"gpr": 3, "svr": 3, "mlp": 8, "kan": 10, "mt_mlp": 12, "mt_kan": 15, "tabpfn": 6}


# ---------------------------------------------------------------------------
@dataclass
class Design:
    """Rows + model matrix for one (target, feature set, censoring mode, split)."""
    target: str
    feature_set: str
    regime: str
    family: str
    censor_mode: str
    split: str
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
    def time(self) -> np.ndarray | None:
        return self.frame["Time_s"].to_numpy(float) if self.regime == "time" else None


def make_design(data: TriboData, target: str, feature_set: str, cfg: dict,
                censor_mode: str = "exclude", split: str = "cv") -> Design | None:
    """Build the design for ``target`` x ``feature_set`` (None if there are no rows)."""
    spec = TARGETS[target]
    df, info = target_frame(data, spec, censor_mode, split)
    if df.empty:
        return None
    cols = resolve_feature_set(feature_set, data, cfg)
    if spec.regime == "time":
        cols = cols + time_columns(cfg)
    check_inputs(cols, spec.regime)
    X = build_X(df, cols)
    meta = df[[c for c in META_COLS if c in df]].reset_index(drop=True).copy()
    meta["y"] = df["y"].to_numpy(float)
    return Design(target, feature_set, spec.regime, spec.family, censor_mode, split, meta, X, info)


def time_grid(n_points: int, t_max: float) -> np.ndarray:
    """Seconds used for fitting: half log-spaced (running-in), half evenly spaced."""
    n_points = max(int(n_points), 2)
    h = max(n_points // 2, 1)
    pts = np.r_[np.geomspace(1, t_max, h), np.linspace(1, t_max, n_points - h)]
    return np.unique(np.round(pts))


def thin_rows(time_s: np.ndarray, n_points: int | None) -> np.ndarray:
    """Boolean mask selecting about ``n_points`` seconds per run."""
    if not n_points:
        return np.ones(len(time_s), bool)
    grid = time_grid(n_points, float(np.nanmax(time_s)))
    return np.isin(np.round(time_s), grid)


def n_points_for(cfg: dict, model: str) -> int:
    base = int(cfg["time_regime"]["train_points_per_run"])
    m = (cfg["models"].get(model) or {}).get("max_points_per_run")
    return min(base, int(m)) if m else base


def regime_cfg(mcfg: dict, regime: str) -> dict:
    """Model config for a regime: ``params_time`` overrides ``params`` for COF(t) designs."""
    if regime == "time" and mcfg.get("params_time"):
        return {**mcfg, "params": {**(mcfg.get("params") or {}), **mcfg["params_time"]}}
    return mcfg


def search_budget(cfg: dict, mcfg: dict) -> tuple[int, int]:
    """(candidates, inner folds) for a model: per-model overrides of the ``cv`` defaults."""
    return (int(mcfg.get("search_iter", cfg["cv"]["search_iter"])),
            int(mcfg.get("inner_folds", cfg["cv"]["inner_folds"])))


def model_ctx(cfg: dict, regime: str, columns: list[str]) -> dict:
    return {"collinear_threshold": cfg["features"].get("collinear_threshold", 0.995),
            "regime": regime, "feature_names": list(columns)}


# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)   # run-level (+ COF_t@ss)
    predictions_time: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-second COF(t)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: list = field(default_factory=list)

    def extend(self, other: "CVResult") -> "CVResult":
        cat = lambda a, b: pd.concat([a, b], ignore_index=True) if len(b) else a   # noqa: E731
        return CVResult(cat(self.predictions, other.predictions),
                        cat(self.predictions_time, other.predictions_time),
                        cat(self.fold_metrics, other.fold_metrics), cat(self.params, other.params),
                        self.failures + other.failures)

    def summary(self) -> pd.DataFrame:
        if self.fold_metrics.empty:
            return pd.DataFrame()
        s = summarize(self.fold_metrics, KEYS)
        oof = self.oof_metrics()
        if not oof.empty:
            s = s.merge(oof, on=KEYS, how="left")
        return s.sort_values(["target", "censor_mode", "RMSE_mean"]).reset_index(drop=True)

    def oof_metrics(self) -> pd.DataFrame:
        """R² of the pooled out-of-fold predictions (per repeat, then averaged)."""
        frames = []
        for p in (self.predictions,):
            if p.empty:
                continue
            m = score_predictions(p, KEYS + ["repeat"])
            frames.append(m.groupby(KEYS)[["R2"]].mean().rename(columns={"R2": "R2_oof"}).reset_index())
        return pd.concat(frames) if frames else pd.DataFrame()


def _cv_task(key: dict, X: np.ndarray, columns: list[str], y: np.ndarray, gcodes: np.ndarray,
             train_idx: np.ndarray, test_idx: np.ndarray, inner: list, mcfg: dict, ctx: dict,
             seed: int, n_iter: int, cand_seed: int) -> dict:
    """Inner-tune, fit on the outer-train rows, predict the outer-test rows (runs in a worker)."""
    spec = get_spec(key["model"])
    t0 = time.perf_counter()
    try:
        Xtr = pd.DataFrame(X[train_idx], columns=columns)
        est, best, info = tune_and_fit(spec, mcfg, Xtr, y[train_idx], gcodes[train_idx], inner,
                                       seed, ctx, n_iter, cand_seed)
        Xte = pd.DataFrame(X[test_idx], columns=columns)
        yhat = np.asarray(est.predict(Xte), float).ravel()
        ystd = np.asarray(est.predict_std(Xte), float).ravel() if spec.has_std else None
        return {**key, "test_idx": test_idx, "yhat": yhat, "ystd": ystd, "params": best,
                "seconds": time.perf_counter() - t0, **{k: v for k, v in info.items()
                                                         if k != "fit_seconds"}}
    except Exception as exc:                                         # noqa: BLE001
        return {**key, "error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}


def run_parallel(func, jobs: list[tuple], n_jobs: int, label: str) -> list[dict]:
    """Run ``func(*args)`` for every job, logging progress; order of results is irrelevant."""
    if not jobs:
        return []
    patch_loky_tracker_handoff()
    out, t0, step = [], time.perf_counter(), max(len(jobs) // 20, 1)
    last = t0
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(func)(*a) for a in jobs) if n_jobs != 1 else (func(*a) for a in jobs)
    for i, r in enumerate(gen, 1):
        out.append(r)
        now = time.perf_counter()
        if i % step == 0 or i == len(jobs) or now - last > 60:
            log.info("  %s: %d/%d tasks (%.0f s)", label, i, len(jobs), now - t0)
            last = now
    return out


def _assemble(designs: list[Design], results: list[dict]) -> CVResult:
    preds, preds_t, params, failures = [], [], [], []
    for r in results:
        d = designs[r["design"]]
        key = {k: r[k] for k in KEYS + ["repeat", "fold"]}
        if "error" in r:
            failures.append({**key, "error": r["error"]})
            log.warning("FAILED %s: %s", key, r["error"])
            continue
        meta = d.frame.iloc[r["test_idx"]].reset_index(drop=True)
        p = meta.rename(columns={"y": "y_true"})
        p["y_pred"] = r["yhat"]
        p["y_std"] = r["ystd"] if r["ystd"] is not None else np.nan
        for k, v in key.items():
            p[k] = v
        (preds_t if d.regime == "time" else preds).append(p)
        params.append({**key, "params": json.dumps(r["params"], default=str),
                       "inner_rmse": r.get("inner_rmse"), "n_candidates": r.get("n_candidates"),
                       "seconds": r["seconds"]})
    res = CVResult()
    order = KEYS + ["repeat", "fold"]
    if preds:
        res.predictions = pd.concat(preds, ignore_index=True)
    if preds_t:
        res.predictions_time = pd.concat(preds_t, ignore_index=True)
        agg = aggregate_cof_to_run(res.predictions_time, order)
        agg["censored"] = False
        res.predictions = pd.concat([res.predictions, agg], ignore_index=True)
    if params:
        res.params = pd.DataFrame(params).sort_values(order).reset_index(drop=True)
    res.failures = failures
    metrics = []
    if not res.predictions.empty:
        metrics.append(score_predictions(res.predictions, order))
    if not res.predictions_time.empty:
        metrics.append(score_predictions(res.predictions_time, order))
    if metrics:
        res.fold_metrics = pd.concat(metrics, ignore_index=True).sort_values(order).reset_index(drop=True)
    return res


def cross_validate(designs: list[Design], specs: list[ModelSpec], cfg: dict, plan: FoldPlan,
                   n_jobs: int = 1, label: str = "CV") -> CVResult:
    """Nested group CV of every compatible (design, model) pair."""
    jobs, cost = [], []
    for di, d in enumerate(designs):
        X = d.X.to_numpy(float)
        cols = list(d.X.columns)
        groups = d.groups
        gcodes = pd.factorize(groups)[0]
        for spec in specs:
            if "joint" in spec.regimes or not spec.supports(d.regime, d.family, d.target):
                continue
            mcfg = regime_cfg(cfg["models"].get(spec.name) or {}, d.regime)
            n_iter, n_inner = search_budget(cfg, mcfg)
            n_pts = n_points_for(cfg, spec.name) if d.regime == "time" else None
            for r in range(plan.repeats):
                for f, tr, te in plan.outer(groups, r):
                    check_disjoint(groups, tr, te)
                    if d.regime == "time":
                        tr = tr[thin_rows(d.time[tr], n_pts)]
                    inner = plan.inner(groups[tr], r, f, n_inner)
                    for a, b in inner:
                        check_disjoint(groups[tr], a, b)
                    key = {"design": di, "target": d.target, "model": spec.name,
                           "feature_set": d.feature_set, "censor_mode": d.censor_mode,
                           "repeat": r, "fold": f}
                    seed = stable_seed(cfg["seed"], d.target, spec.name, d.feature_set, r, f)
                    cand_seed = stable_seed(cfg["seed"], "cand", d.target, spec.name)
                    jobs.append((key, X, cols, d.y, gcodes, tr, te, inner, mcfg,
                                 model_ctx(cfg, d.regime, cols), seed, n_iter, cand_seed))
                    cost.append(len(tr) * _COST.get(spec.name, 1) * max(len(inner), 1))
    order = np.argsort(cost)[::-1]                     # longest tasks first (load balancing)
    jobs = [jobs[i] for i in order]
    log.info("%s: %d fit tasks on %s worker(s)", label, len(jobs), n_jobs)
    return _assemble(designs, run_parallel(_cv_task, jobs, n_jobs, label))


# ---------------------------------------------------------------------------
def select_final(summary: pd.DataFrame, cfg: dict, censor_mode: str) -> pd.DataFrame:
    """(target, model, feature_set) pairs to refit for the unseen-point test.

    Per target: the ``top_k`` pairs by mean outer-fold RMSE, plus the best feature set of
    each model in ``final.always_include`` that was evaluated for that target.
    """
    fc = cfg["final"]
    if summary.empty:
        return pd.DataFrame(columns=["target", "model", "feature_set", "reason", "RMSE_mean"])
    s = summary[(summary["censor_mode"] == censor_mode) & (summary["target"] != COF_T_RUN)]
    rows = []
    for t, sub in s.groupby("target"):
        sub = sub.sort_values("RMSE_mean")
        pick = sub.head(int(fc.get("top_k", 2)))
        rows.append(pick.assign(reason="top_k"))
        for m in fc.get("always_include", []) + list(cfg.get("interpret", {}).get("refit_models", [])):
            ms = sub[sub["model"] == m]
            if len(ms) and not ((pick["model"] == m).any()):
                rows.append(ms.head(1).assign(reason="always_include"))
    if not rows:
        return pd.DataFrame(columns=["target", "model", "feature_set", "reason"])
    out = pd.concat(rows)[["target", "model", "feature_set", "reason", "RMSE_mean"]]
    return out.drop_duplicates(["target", "model", "feature_set"]).reset_index(drop=True)


def _final_task(sel: dict, dcv: Design, dho: Design | None, plan: FoldPlan, cfg: dict) -> dict:
    spec = get_spec(sel["model"])
    mcfg = cfg["models"].get(spec.name) or {}
    groups = dcv.groups
    tr = np.arange(len(dcv.frame))
    if dcv.regime == "time":
        tr = tr[thin_rows(dcv.time, n_points_for(cfg, spec.name))]
    g = groups[tr]
    tuning = [(a, b) for _, a, b in plan.outer(g, 0)]       # group K-fold over all CV data
    for a, b in tuning:
        check_disjoint(g, a, b)
    cols = list(dcv.X.columns)
    seed = stable_seed(cfg["seed"], "final", sel["target"], spec.name, sel["feature_set"])
    t0 = time.perf_counter()
    try:
        est, best, _ = tune_and_fit(spec, mcfg, dcv.X.iloc[tr].reset_index(drop=True),
                                    dcv.y[tr], pd.factorize(g)[0], tuning, seed,
                                    model_ctx(cfg, dcv.regime, cols), int(cfg["cv"]["search_iter"]),
                                    stable_seed(cfg["seed"], "cand", sel["target"], spec.name))
    except Exception as exc:                                         # noqa: BLE001
        return {**sel, "error": f"{type(exc).__name__}: {exc}"}
    out = {**sel, "estimator": est, "params": best, "seconds": time.perf_counter() - t0,
           "has_std": spec.has_std}
    if dho is not None:
        out["yhat"] = np.asarray(est.predict(dho.X), float).ravel()
        out["ystd"] = np.asarray(est.predict_std(dho.X), float).ravel() if spec.has_std else None
    return out


@dataclass
class FinalResult:
    selections: pd.DataFrame
    predictions: pd.DataFrame          # holdout predictions (run level, incl. COF_t@ss)
    predictions_time: pd.DataFrame     # holdout per-second COF(t)
    metrics: pd.DataFrame
    estimators: dict                   # (target, model, feature_set) -> fitted estimator
    params: dict
    failures: list


def final_refit(data: TriboData, cfg: dict, plan: FoldPlan, selections: pd.DataFrame,
                cv: CVResult, models_dir: Path | None, n_jobs: int = 1,
                censor_mode: str = "exclude") -> FinalResult:
    """Refit the selected models on all CV data (tuned by group CV), predict the unseen points.

    Prediction intervals: Gaussian ``mu ± z sigma`` for models with a predictive std, and
    for every model a CV-residual interval ``mu ± q`` where ``q`` is the ``interval_level``
    quantile of that model's absolute out-of-fold errors (split-conformal style).
    """
    level = float(cfg["final"].get("interval_level", 0.95))
    jobs, joint = [], set()
    for sel in selections.to_dict("records"):
        if get_spec(sel["model"]).regimes == ("joint",):
            joint.add((sel["model"], sel["feature_set"]))
            continue
        dcv = make_design(data, sel["target"], sel["feature_set"], cfg, censor_mode, "cv")
        dho = make_design(data, sel["target"], sel["feature_set"], cfg, censor_mode, "holdout")
        if dcv is not None:
            jobs.append((sel, dcv, dho, plan, cfg))
    for model, fs in sorted(joint):
        jcv = make_joint_design(data, fs, cfg, censor_mode, "cv")
        jho = make_joint_design(data, fs, cfg, censor_mode, "holdout")
        if jcv is not None:
            jobs.append(({"target": "joint", "model": model, "feature_set": fs}, jcv, jho, plan, cfg))
    log.info("final refit: %d model(s) on all CV data", len(jobs))
    results = run_parallel(_final_dispatch, jobs, n_jobs, "final")

    blocks, ests, params, failures = [], {}, {}, []
    for r in results:
        k = (r["target"], r["model"], r["feature_set"])
        if "error" in r:
            failures.append({"target": k[0], "model": k[1], "feature_set": k[2], "error": r["error"]})
            log.warning("final refit FAILED %s: %s", k, r["error"])
            continue
        ests[k], params[k] = r["estimator"], r["params"]
        if models_dir is not None:
            try:
                joblib.dump(r["estimator"], Path(models_dir) / f"{k[0]}__{k[1]}__{k[2]}.joblib")
            except Exception as exc:                                 # noqa: BLE001
                log.warning("could not save model %s: %s", k, exc)
        for target, meta, yhat, ystd in r.get("holdout", []):
            blocks.append(_holdout_block(meta, yhat, ystd, (target, k[1], k[2]), censor_mode,
                                         cv, level))
    return _final_result(selections, blocks, ests, params, failures)


def _final_dispatch(sel, dcv, dho, plan, cfg) -> dict:
    if isinstance(dcv, JointDesign):
        return _final_joint_task(sel, dcv, dho, plan, cfg)
    return _final_task(sel, dcv, dho, plan, cfg)


def _final_task(sel: dict, dcv: Design, dho: Design | None, plan: FoldPlan, cfg: dict) -> dict:
    spec = get_spec(sel["model"])
    mcfg = regime_cfg(cfg["models"].get(spec.name) or {}, dcv.regime)
    tr = np.arange(len(dcv.frame))
    if dcv.regime == "time":
        tr = tr[thin_rows(dcv.time, n_points_for(cfg, spec.name))]
    g = dcv.groups[tr]
    n_iter, _ = search_budget(cfg, mcfg)
    tuning = [(a, b) for _, a, b in plan.outer(g, 0)]       # group K-fold over all CV data
    for a, b in tuning:
        check_disjoint(g, a, b)
    cols = list(dcv.X.columns)
    seed = stable_seed(cfg["seed"], "final", sel["target"], spec.name, sel["feature_set"])
    t0 = time.perf_counter()
    try:
        est, best, _ = tune_and_fit(spec, mcfg, dcv.X.iloc[tr].reset_index(drop=True),
                                    dcv.y[tr], pd.factorize(g)[0], tuning, seed,
                                    model_ctx(cfg, dcv.regime, cols), n_iter,
                                    stable_seed(cfg["seed"], "cand", sel["target"], spec.name))
        out = {**sel, "estimator": est, "params": best, "seconds": time.perf_counter() - t0}
        if dho is not None:
            yhat = np.asarray(est.predict(dho.X), float).ravel()
            ystd = np.asarray(est.predict_std(dho.X), float).ravel() if spec.has_std else None
            out["holdout"] = [(sel["target"], dho.frame, yhat, ystd)]
        return out
    except Exception as exc:                                         # noqa: BLE001
        return {**sel, "error": f"{type(exc).__name__}: {exc}"}


def _holdout_block(meta: pd.DataFrame, yhat, ystd, key: tuple, censor_mode: str, cv: CVResult,
                   level: float) -> pd.DataFrame:
    """Holdout predictions with Gaussian and CV-residual prediction intervals."""
    z = float(norm.ppf(0.5 + level / 2))
    p = meta.reset_index(drop=True).rename(columns={"y": "y_true"})
    p["y_pred"] = np.asarray(yhat, float)
    p["y_std"] = np.asarray(ystd, float) if ystd is not None else np.nan
    p["target"], p["model"], p["feature_set"] = key
    p["censor_mode"] = censor_mode
    q = _cv_residual_quantile(cv, key, level, TARGETS[key[0]].regime)
    p["pi_cv_lo"], p["pi_cv_hi"] = p["y_pred"] - q, p["y_pred"] + q
    has = np.isfinite(p["y_std"].to_numpy(float))
    p["pi_lo"] = np.where(has, p["y_pred"] - z * p["y_std"], p["pi_cv_lo"])
    p["pi_hi"] = np.where(has, p["y_pred"] + z * p["y_std"], p["pi_cv_hi"])
    p["inside_pi"] = ((p["y_true"] >= p["pi_lo"]) & (p["y_true"] <= p["pi_hi"])).astype(float)
    p["regime"] = TARGETS[key[0]].regime
    return p


def _final_result(selections, blocks, ests, params, failures) -> FinalResult:
    run_blocks = [b for b in blocks if b["regime"].iloc[0] == "run"] if blocks else []
    time_blocks = [b for b in blocks if b["regime"].iloc[0] == "time"] if blocks else []
    pr = pd.concat(run_blocks, ignore_index=True) if run_blocks else pd.DataFrame()
    pt = pd.concat(time_blocks, ignore_index=True) if time_blocks else pd.DataFrame()
    if len(pt):
        agg = aggregate_cof_to_run(pt, KEYS)
        agg["censored"] = False
        z = float(norm.ppf(0.975))
        agg["pi_lo"], agg["pi_hi"] = agg["y_pred"] - z * agg["y_std"], agg["y_pred"] + z * agg["y_std"]
        agg["inside_pi"] = np.where(np.isfinite(agg["y_std"]),
                                    ((agg["y_true"] >= agg["pi_lo"]) & (agg["y_true"] <= agg["pi_hi"]))
                                    .astype(float), np.nan)
        pr = pd.concat([pr, agg], ignore_index=True)
    metrics = []
    if len(pr):
        metrics.append(score_predictions(pr, KEYS))
    if len(pt):
        metrics.append(score_predictions(pt, KEYS))
    m = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    if len(m):
        cov = []
        for df in (pr, pt):
            if len(df) and "inside_pi" in df:
                c = df[~df["censored"].astype(bool)] if "censored" in df else df
                cov.append(c.dropna(subset=["inside_pi"]).groupby(KEYS)["inside_pi"].mean())
        if cov:
            m = m.merge(pd.concat(cov).rename("pi_coverage").reset_index(), on=KEYS, how="left")
    return FinalResult(selections, pr.drop(columns=["regime"], errors="ignore"),
                       pt.drop(columns=["regime"], errors="ignore"), m, ests, params, failures)


def _cv_residual_quantile(cv: CVResult, key: tuple, level: float, regime: str) -> float:
    src = cv.predictions_time if regime == "time" else cv.predictions
    if src.empty:
        return np.nan
    s = src[(src["target"] == key[0]) & (src["model"] == key[1]) & (src["feature_set"] == key[2])]
    if "censored" in s:
        s = s[~s["censored"].astype(bool)]
    err = np.abs(s["y_true"] - s["y_pred"]).to_numpy(float)
    return float(np.quantile(err[np.isfinite(err)], level)) if np.isfinite(err).any() else np.nan


# ---------------------------------------------------------------------------
# Joint (multi-task) models: COF(t) + log10 k from one network
JOINT_WEAR, JOINT_COF = "log10_k", "COF_t"


@dataclass
class JointDesign:
    """Run table (features + wear target) and per-second COF rows linked by ``row_run``."""
    feature_set: str
    censor_mode: str
    split: str
    runs: pd.DataFrame           # Experiment_ID, group, y_wear, censored, y_lod
    R: pd.DataFrame              # run features
    rows: pd.DataFrame           # Experiment_ID, group, Time_s, Phase, y (COF)
    T: pd.DataFrame              # time encoding
    row_run: np.ndarray          # row -> index into runs

    @property
    def run_groups(self) -> np.ndarray:
        return self.runs["group"].astype(str).to_numpy()


def make_joint_design(data: TriboData, feature_set: str, cfg: dict, censor_mode: str = "exclude",
                      split: str = "cv") -> JointDesign | None:
    w, _ = target_frame(data, JOINT_WEAR, censor_mode, split)
    c, _ = target_frame(data, JOINT_COF, split=split)
    if c.empty:
        return None
    base = data.runs[data.runs["is_holdout"] == (split == "holdout")]
    ids = set(c["Experiment_ID"]) | (set(w["Experiment_ID"]) if len(w) else set())
    runs = base[base["Experiment_ID"].isin(ids)].sort_values("Experiment_ID").reset_index(drop=True)
    cols = resolve_feature_set(feature_set, data, cfg)
    check_inputs(cols, "run")
    tcols = time_columns(cfg)
    check_inputs(cols + tcols, "time")
    meta = runs[["Experiment_ID", "group"]].copy()
    if len(w):
        wi = w.set_index("Experiment_ID")
        meta["y_wear"] = meta["Experiment_ID"].map(wi["y"]).astype(float)
        meta["censored"] = meta["Experiment_ID"].map(wi["censored"]).eq(True)
        meta["y_lod"] = meta["Experiment_ID"].map(wi["y_lod"]).astype(float)
    else:
        meta["y_wear"], meta["censored"], meta["y_lod"] = np.nan, False, np.nan
    pos = {e: i for i, e in enumerate(runs["Experiment_ID"])}
    rows = c[["Experiment_ID", "group", "Time_s", "Phase", "y"]].reset_index(drop=True)
    return JointDesign(feature_set, censor_mode, split, meta, build_X(runs, cols), rows,
                       build_X(c, tcols), rows["Experiment_ID"].map(pos).to_numpy(int))


def _joint_fit(spec, params, seed, jd: JointDesign, run_idx: np.ndarray, row_mask: np.ndarray):
    """Fit a joint model on the runs ``run_idx`` and their (thinned) rows ``row_mask``."""
    est = spec.build(dict(params), seed, {})
    local = np.full(len(jd.runs), -1)
    local[run_idx] = np.arange(len(run_idx))
    sel = np.flatnonzero(row_mask & (local[jd.row_run] >= 0))
    est.fit(jd.R.iloc[run_idx].reset_index(drop=True), jd.runs["y_wear"].to_numpy(float)[run_idx],
            local[jd.row_run[sel]], jd.T.iloc[sel].reset_index(drop=True),
            jd.rows["y"].to_numpy(float)[sel], groups=jd.run_groups[run_idx])
    return est


def _joint_score(est, jd: JointDesign, run_idx: np.ndarray, row_mask: np.ndarray,
                 sd_w: float, sd_c: float) -> float:
    local = np.full(len(jd.runs), -1)
    local[run_idx] = np.arange(len(run_idx))
    sel = np.flatnonzero(row_mask & (local[jd.row_run] >= 0))
    p = est.predict_both(jd.R.iloc[run_idx], local[jd.row_run[sel]], jd.T.iloc[sel])
    yw = jd.runs["y_wear"].to_numpy(float)[run_idx]
    ok = np.isfinite(yw)
    rw = float(np.sqrt(np.mean((p["wear"][ok] - yw[ok]) ** 2))) / sd_w if ok.any() else 0.0
    rc = float(np.sqrt(np.mean((p["cof"] - jd.rows["y"].to_numpy(float)[sel]) ** 2))) / sd_c
    return rw + rc


def _joint_tune_fit(spec, mcfg, jd, train_runs, row_thin, inner, seed, n_iter, cand_seed):
    from .tuning import sample_candidates
    fixed = dict(mcfg.get("params") or {})
    cands = sample_candidates(mcfg.get("search"), n_iter, cand_seed)
    best, scores = cands[0], []
    yw = jd.runs["y_wear"].to_numpy(float)[train_runs]
    sd_w = float(np.nanstd(yw) or 1.0)
    sd_c = float(np.std(jd.rows["y"].to_numpy(float)[np.isin(jd.row_run, train_runs)]) or 1.0)
    if len(cands) > 1 and inner:
        tune_fixed = {**fixed, **({"n_seeds": mcfg["tune_n_seeds"]} if "tune_n_seeds" in mcfg else {})}
        for c in cands:
            errs = []
            for a, b in inner:
                try:
                    est = _joint_fit(spec, {**tune_fixed, **c}, seed, jd, train_runs[a], row_thin)
                    errs.append(_joint_score(est, jd, train_runs[b], row_thin, sd_w, sd_c))
                except Exception as exc:                             # noqa: BLE001
                    log.debug("joint candidate %s failed: %s", c, exc)
                    errs.append(np.inf)
            scores.append(float(np.mean(errs)))
        best = cands[int(np.argmin(scores))]
    est = _joint_fit(spec, {**fixed, **best}, seed, jd, train_runs, row_thin)
    return est, best, (float(np.min(scores)) if scores else np.nan)


def _joint_task(key: dict, jd: JointDesign, train_runs, test_runs, inner, row_thin, mcfg,
                seed, n_iter, cand_seed) -> dict:
    spec = get_spec(key["model"])
    t0 = time.perf_counter()
    try:
        est, best, inner_score = _joint_tune_fit(spec, mcfg, jd, train_runs, row_thin, inner,
                                                 seed, n_iter, cand_seed)
        local = np.full(len(jd.runs), -1)
        local[test_runs] = np.arange(len(test_runs))
        rows_te = np.flatnonzero(local[jd.row_run] >= 0)               # every second is scored
        p = est.predict_both(jd.R.iloc[test_runs], local[jd.row_run[rows_te]], jd.T.iloc[rows_te])
        return {**key, "test_runs": test_runs, "rows_te": rows_te, "pred": p, "params": best,
                "inner_rmse": inner_score, "seconds": time.perf_counter() - t0}
    except Exception as exc:                                         # noqa: BLE001
        return {**key, "error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}


def cross_validate_joint(jdesigns: list[JointDesign], specs: list[ModelSpec], cfg: dict,
                         plan: FoldPlan, n_jobs: int = 1, targets: tuple = (JOINT_WEAR, JOINT_COF),
                         label: str = "joint CV") -> CVResult:
    """Nested group CV of the multi-task models; scored as ``log10_k`` and ``COF_t`` rows."""
    jobs = []
    for jd in jdesigns:
        rg = jd.run_groups
        for spec in specs:
            if spec.regimes != ("joint",):
                continue
            mcfg = cfg["models"].get(spec.name) or {}
            n_iter, n_inner = search_budget(cfg, mcfg)
            row_thin = thin_rows(jd.rows["Time_s"].to_numpy(float), n_points_for(cfg, spec.name))
            for r in range(plan.repeats):
                for f, tr, te in plan.outer(rg, r):
                    check_disjoint(rg, tr, te)
                    inner = plan.inner(rg[tr], r, f, n_inner)
                    for a, b in inner:
                        check_disjoint(rg[tr], a, b)
                    key = {"model": spec.name, "feature_set": jd.feature_set,
                           "censor_mode": jd.censor_mode, "repeat": r, "fold": f}
                    seed = stable_seed(cfg["seed"], "joint", spec.name, jd.feature_set, r, f)
                    jobs.append((key, jd, tr, te, inner, row_thin, mcfg, seed, n_iter,
                                 stable_seed(cfg["seed"], "cand", "joint", spec.name)))
    log.info("%s: %d fit tasks on %s worker(s)", label, len(jobs), n_jobs)
    results = run_parallel(_joint_task, jobs, n_jobs, label)
    by_fs = {jd.feature_set: jd for jd in jdesigns}
    preds, preds_t, params, failures = [], [], [], []
    for r in results:
        key = {k: r[k] for k in ("model", "feature_set", "censor_mode", "repeat", "fold")}
        if "error" in r:
            failures.append({**key, "target": "joint", "error": r["error"]})
            log.warning("FAILED %s: %s", key, r["error"])
            continue
        jd = by_fs[r["feature_set"]]
        if JOINT_WEAR in targets:
            m = jd.runs.iloc[r["test_runs"]].reset_index(drop=True)
            ok = np.isfinite(m["y_wear"].to_numpy(float))
            pw = m.loc[ok, ["Experiment_ID", "group", "censored", "y_lod"]].copy()
            pw["y_true"] = m.loc[ok, "y_wear"].to_numpy(float)
            pw["y_pred"] = r["pred"]["wear"][ok]
            pw["y_std"] = r["pred"]["wear_std"][ok]
            preds.append(pw.assign(target=JOINT_WEAR, **key))
        if JOINT_COF in targets:
            pc = jd.rows.iloc[r["rows_te"]].reset_index(drop=True).rename(columns={"y": "y_true"})
            pc["y_pred"], pc["y_std"] = r["pred"]["cof"], r["pred"]["cof_std"]
            pc["censored"], pc["y_lod"] = False, np.nan
            preds_t.append(pc.assign(target=JOINT_COF, **key))
        params.append({**key, "target": "joint", "params": json.dumps(r["params"], default=str),
                       "inner_rmse": r["inner_rmse"], "n_candidates": np.nan,
                       "seconds": r["seconds"]})
    res = CVResult(failures=failures)
    order = KEYS + ["repeat", "fold"]
    if preds:
        res.predictions = pd.concat(preds, ignore_index=True)
    if preds_t:
        res.predictions_time = pd.concat(preds_t, ignore_index=True)
        agg = aggregate_cof_to_run(res.predictions_time, order)
        agg["censored"] = False
        res.predictions = pd.concat([res.predictions, agg], ignore_index=True)
    if params:
        res.params = pd.DataFrame(params)
    metrics = [score_predictions(df, order) for df in (res.predictions, res.predictions_time)
               if not df.empty]
    if metrics:
        res.fold_metrics = pd.concat(metrics, ignore_index=True)
    return res


def _final_joint_task(sel: dict, jcv: JointDesign, jho: JointDesign | None, plan: FoldPlan,
                      cfg: dict) -> dict:
    spec = get_spec(sel["model"])
    mcfg = cfg["models"].get(spec.name) or {}
    rg = jcv.run_groups
    tuning = [(a, b) for _, a, b in plan.outer(rg, 0)]
    row_thin = thin_rows(jcv.rows["Time_s"].to_numpy(float), n_points_for(cfg, spec.name))
    t0 = time.perf_counter()
    try:
        est, best, _ = _joint_tune_fit(
            spec, mcfg, jcv, np.arange(len(jcv.runs)), row_thin, tuning,
            stable_seed(cfg["seed"], "final", "joint", spec.name, sel["feature_set"]),
            search_budget(cfg, mcfg)[0], stable_seed(cfg["seed"], "cand", "joint", spec.name))
        out = {**sel, "estimator": est, "params": best, "seconds": time.perf_counter() - t0,
               "holdout": []}
        if jho is not None:
            p = est.predict_both(jho.R, jho.row_run, jho.T)
            ok = np.isfinite(jho.runs["y_wear"].to_numpy(float))
            mw = jho.runs.loc[ok, ["Experiment_ID", "group", "censored", "y_lod"]].copy()
            mw["y"] = jho.runs.loc[ok, "y_wear"].to_numpy(float)
            out["holdout"].append((JOINT_WEAR, mw, p["wear"][ok], p["wear_std"][ok]))
            mc = jho.rows.copy()
            mc["censored"], mc["y_lod"] = False, np.nan
            out["holdout"].append((JOINT_COF, mc, p["cof"], p["cof_std"]))
        return out
    except Exception as exc:                                         # noqa: BLE001
        return {**sel, "error": f"{type(exc).__name__}: {exc}"}
