"""Support-vector regression with an RBF kernel (C, gamma, epsilon tuned)."""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .registry import ModelSpec


def build_svr(params: dict, seed: int, ctx: dict):
    # y is standardised (inside the fold) so epsilon / C mean the same for every target
    params.setdefault("cache_size", 500)          # MB of kernel cache: faster on COF(t) rows
    inner = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("model", SVR(kernel="rbf", **params))])
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


SPECS = [ModelSpec("svr", build_svr, "SVR, RBF kernel (C, gamma, epsilon tuned; y standardised)")]
