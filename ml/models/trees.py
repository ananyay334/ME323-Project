"""Shallow XGBoost (depth 1-2, min child weight >= 2, lr <= 0.05, subsample 0.8)."""
from __future__ import annotations

from .registry import ModelSpec


def build_xgb(params: dict, seed: int, ctx: dict):
    from xgboost import XGBRegressor             # optional dependency
    params.setdefault("n_jobs", 1)
    params.setdefault("tree_method", "hist")
    params.setdefault("verbosity", 0)
    return XGBRegressor(random_state=seed, **params)      # handles NaN inputs natively


SPECS = [
    ModelSpec("xgb", build_xgb, "XGBoost, shallow (max_depth 1-2; grid in nested CV)",
              requires=("xgboost",)),
]
