"""Physics route for the wear trajectory (B5).

For every gkf5 fold: the three eq.-1 parameters (log10 V̇_steady, ΔV_run, τ_trans) are
predicted for the test runs by the best ``trajectory_params`` model, chosen *inside* the
outer training set (lowest inner-CV RMSE among the selectable models, each tuned on the
same inner folds). The curve

    V_w(t) = V̇_steady·t + ΔV_run·(1 − e^(−t/τ_trans))    (tribo_extract.physics.two_stage_wear)

is then evaluated at the test runs' checkpoint times and scored as log10 V_w(t) on the
fold's ``wear_checkpoints`` rows, next to the direct ``wear_checkpoints`` models.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from tribo_extract.physics import two_stage_wear

from .config import stable_seed
from .data import TriboData
from .evaluate import KEYS, CVResult, Design, feature_columns, make_design, run_parallel
from .features import build_X
from .metrics import score_predictions
from .models.registry import ModelSpec, get_spec
from .splits import SplitPlan, check_disjoint
from .tuning import tune_and_fit

log = logging.getLogger("ml.trajectory")

PARAMS = ["log10_Vdot_steady", "log10_dV_run", "log10_tau_trans"]
MODEL_NAME = "physics_route"


def _route_task(key: dict, pdesigns: dict[str, Design], train_runs: set, test_runs: list[str],
                X_test_runs: pd.DataFrame, plan: SplitPlan, names: list[str], cfg: dict,
                seed: int) -> dict:
    """Choose a model per parameter by inner CV, predict the test runs' eq.-1 parameters."""
    t0 = time.perf_counter()
    first_only = bool(cfg.get("grid_first_only", False))
    out = {**key, "test_runs": test_runs, "pred": {}, "chosen": {}}
    try:
        for p, d in pdesigns.items():
            tr = np.flatnonzero(np.isin(d.run_ids, list(train_runs)))
            groups = d.groups[tr]
            inner = plan.inner(groups)
            Xtr, ytr = d.X.iloc[tr].reset_index(drop=True), d.y[tr]
            best = None
            for name in names:
                spec = get_spec(name)
                mcfg = cfg["models"].get(name) or {}
                est, params, score = tune_and_fit(spec, mcfg, Xtr, ytr, inner, seed, {},
                                                  first_only, always_score=True)
                if best is None or (np.isfinite(score) and not best[2] <= score):
                    best = (name, est, score, params)
            out["pred"][p] = np.asarray(best[1].predict(X_test_runs[list(d.X.columns)]), float)
            out["chosen"][p] = {"model": best[0], "inner_rmse": best[2], "params": best[3]}
    except Exception as exc:                                         # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.perf_counter() - t0
    return out


def physics_route(data: TriboData, cfg: dict, plan: SplitPlan, specs: list[ModelSpec],
                  n_jobs: int = 1, censor_mode: str = "exclude") -> tuple[CVResult, pd.DataFrame]:
    """Out-of-fold V_w(t) reconstructions on gkf5; returns (CVResult, per-run parameters)."""
    names = [s.name for s in specs if s.selectable]
    empty = (CVResult(), pd.DataFrame())
    if not names:
        return empty
    jobs, ckpt = [], {}
    for fs in cfg["features"]["use"]:
        pdesigns = {p: make_design(data, p, fs, cfg, censor_mode) for p in PARAMS}
        cd = make_design(data, "log10_V_t", fs, cfg, censor_mode)
        if cd is None or any(d is None or len(d.frame) < 5 for d in pdesigns.values()):
            log.warning("physics route (%s) skipped: no wear_checkpoints rows or too few eq.-1 fits",
                        fs)
            continue
        ckpt[fs] = cd
        cols = feature_columns(data, "endpoint", fs, cfg)
        X_runs = build_X(data.runs, cols).set_index(data.runs["Experiment_ID"].to_numpy())
        for r in range(len(plan.seeds)):
            for f, tr, te in plan.outer(cd.run_ids, "gkf5", r):
                check_disjoint(cd.groups, tr, te)
                train_runs = {e for e, fo in plan.gkf5[r].items() if fo != f}
                test_runs = sorted(set(cd.run_ids[te]))
                key = {"feature_set": fs, "repeat": r, "fold": f}
                jobs.append((key, pdesigns, train_runs, test_runs, X_runs.loc[test_runs], plan,
                             names, cfg, stable_seed(cfg["seed"], MODEL_NAME, fs, r, f)))
    if not jobs:
        return empty
    log.info("physics route: %d fold tasks", len(jobs))
    results = run_parallel(_route_task, jobs, n_jobs, "physics route")
    preds, params, failures, run_params = [], [], [], []
    for res in results:
        key = {"regime": "wear_checkpoints", "target": "log10_V_t", "model": MODEL_NAME,
               "feature_set": res["feature_set"], "split": "gkf5", "repeat": res["repeat"],
               "fold": res["fold"]}
        if "error" in res:
            failures.append({**key, "error": res["error"]})
            log.warning("physics route FAILED %s: %s", key, res["error"])
            continue
        cd = ckpt[res["feature_set"]]
        theta = pd.DataFrame({p: 10 ** res["pred"][p] for p in PARAMS}, index=res["test_runs"])
        rows = cd.frame[cd.frame["Experiment_ID"].isin(res["test_runs"])].reset_index(drop=True)
        th = theta.loc[rows["Experiment_ID"]].to_numpy()
        v = two_stage_wear(rows["Time_s"].to_numpy(float), th[:, 0], th[:, 1], th[:, 2])
        p = rows.rename(columns={"y": "y_true"})
        p["y_pred"] = np.log10(np.clip(v, 1e-300, None))
        p["y_std"] = np.nan
        for k, val in key.items():
            p[k] = val
        p["fold_label"] = str(res["fold"])
        preds.append(p)
        params.append({**key, "params": json.dumps(res["chosen"], default=str), "inner_rmse": np.nan,
                       "seconds": res["seconds"]})
        run_params.append(theta.rename(columns=lambda c: c.replace("log10_", ""))
                          .rename_axis("Experiment_ID").reset_index()
                          .assign(feature_set=res["feature_set"], repeat=res["repeat"],
                                  fold=res["fold"], chosen=json.dumps(
                                      {q: v["model"] for q, v in res["chosen"].items()})))
    out = CVResult(failures=failures)
    if preds:
        out.predictions = pd.concat(preds, ignore_index=True)
        order = KEYS + ["repeat", "fold"]
        out.fold_metrics = score_predictions(out.predictions, order + ["fold_label"])
        out.params = pd.DataFrame(params)
    return out, (pd.concat(run_params, ignore_index=True) if run_params else pd.DataFrame())
