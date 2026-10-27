"""Helpers shared by the comparison studies."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from ..evaluate.cv import collect_oof, protocol_splits, run_tasks
from ..evaluate.interpret import group_bootstrap, noise_floor
from ..evaluate.metrics import compute_metrics
from ..models.registry import TargetSpec, make_model, target_frame


def adhoc_target(column: str, log10: bool, physics: str, subset: str = "all", label: str | None = None) -> TargetSpec:
    return TargetSpec(column, column, log10, subset, physics, label or column, label or column)


def cv_oof(cfg: dict, runs: pd.DataFrame, t: TargetSpec, model: str, protocol: str = "P1", n_jobs: int = 1,
           fixed: dict | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = target_frame(runs, t) if df is None else df
    fac = partial(make_model, model, cfg, t, cfg["seed"], fixed) if fixed else partial(make_model, model, cfg, t, cfg["seed"])
    tasks = [{"model": model, "protocol": protocol, "fold": i, "factory": fac, "df": df, "tr": tr, "te": te}
             for i, (tr, te) in enumerate(protocol_splits(df, protocol))]
    if not tasks:
        return pd.DataFrame(columns=["model", "protocol", "run_id", "y", "pred", "sd"])
    return collect_oof(tasks, run_tasks(tasks, n_jobs), ["model", "protocol"])


def score(oof: pd.DataFrame, rep_sd: float = np.nan) -> dict:
    if oof.empty:
        return {"n": 0, "R2": np.nan, "RMSE": np.nan}
    sd = oof["sd"].to_numpy() if oof["sd"].notna().all() else None
    return compute_metrics(oof.y.to_numpy(), oof.pred.to_numpy(), sd, rep_sd)


def replicate_sd(df: pd.DataFrame) -> float:
    return noise_floor(df)["replicate_sd"]


def ols_with_ci(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    """OLS coefficients (with intercept) and group-bootstrap 95 % intervals."""
    A = np.column_stack([np.ones(len(y)), X])
    est = np.linalg.lstsq(A, y, rcond=None)[0]
    boots = []
    for idx in group_bootstrap(groups, n_boot, seed):
        if len(np.unique(X[idx], axis=0)) < A.shape[1]:
            continue
        boots.append(np.linalg.lstsq(A[idx], y[idx], rcond=None)[0])
    B = np.array(boots)
    return pd.DataFrame({"term": ["intercept"] + names, "estimate": est, "ci_lo": np.percentile(B, 2.5, axis=0),
                         "ci_hi": np.percentile(B, 97.5, axis=0)})
