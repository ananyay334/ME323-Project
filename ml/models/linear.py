"""Linear baselines: OLS (exactly collinear columns removed per fold) and Ridge."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .registry import ModelSpec


class DropCollinear(BaseEstimator, TransformerMixin):
    """Drop constant columns and the later column of any pair with |r| > ``threshold``.

    Fitted on the training fold only (it sits inside the Pipeline). With a fixed stroke,
    ``Speed_mps`` is an exact multiple of ``Freq_Hz`` and is removed for plain OLS.
    """

    def __init__(self, threshold: float = 0.995):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        sd = X.std(axis=0)
        keep: list[int] = []
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
        for j in range(X.shape[1]):
            if sd[j] <= 1e-12:
                continue
            if all(abs(corr[j, k]) <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep, int)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


def _pre() -> list:
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


def build_linear(params: dict, seed: int, ctx: dict):
    thr = params.pop("collinear_threshold", ctx.get("collinear_threshold", 0.995))
    return Pipeline(_pre() + [("drop_collinear", DropCollinear(thr)), ("model", LinearRegression())])


def build_ridge(params: dict, seed: int, ctx: dict):
    return Pipeline(_pre() + [("model", Ridge(**params))])


SPECS = [
    ModelSpec("linear", build_linear, "OLS on standardised inputs (main effects; collinear columns "
              "dropped in-fold)"),
    ModelSpec("ridge", build_ridge, "Ridge regression (alpha tuned in the inner CV)"),
]
