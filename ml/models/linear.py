"""Ridge regression: reference baseline only (reported, never selected as best)."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .registry import ModelSpec


def build_ridge(params: dict, seed: int, ctx: dict):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", Ridge(**params))])


SPECS = [ModelSpec("ridge", build_ridge, "Ridge regression (alpha tuned; reference baseline)",
                   selectable=False)]
