"""Regularised tree ensembles: random forest and XGBoost (depth <= 3, min leaf >= 3)."""
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
    ModelSpec("rf", build_rf, "Random forest (depth <= 3, min leaf >= 3; grid in nested CV)"),
    ModelSpec("xgb", build_xgb, "XGBoost (depth <= 3, min child weight >= 3, lr <= 0.05)",
              requires=("xgboost",)),
]
