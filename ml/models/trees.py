"""Regularised tree ensembles: random forest and XGBoost (shallow trees, min leaf size)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .registry import ModelSpec


def build_rf(params: dict, seed: int, ctx: dict):
    params.setdefault("n_jobs", 1)               # parallelism is across CV tasks
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", RandomForestRegressor(random_state=seed, **params))])


def build_xgb(params: dict, seed: int, ctx: dict):
    from xgboost import XGBRegressor             # optional dependency
    params.setdefault("n_jobs", 1)
    params.setdefault("tree_method", "hist")
    params.setdefault("verbosity", 0)
    return XGBRegressor(random_state=seed, **params)      # handles NaN inputs natively


SPECS = [
    ModelSpec("rf", build_rf, "Random forest (shallow, min leaf size; tuned in nested CV)"),
    ModelSpec("xgb", build_xgb, "XGBoost (shallow, L1/L2-regularised; tuned in nested CV)",
              requires=("xgboost",)),
]
