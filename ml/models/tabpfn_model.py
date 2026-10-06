"""TabPFN regressor: a small-N specialist benchmark for the wear targets only.

Needs ``pip install tabpfn`` (v2+, which provides ``TabPFNRegressor``); the first fit
downloads the model weights. Missing package -> the model is skipped with a warning.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .registry import ModelSpec


class TabPFNModel(BaseEstimator, RegressorMixin):
    """Thin wrapper (numpy in, CPU only); ``predict_std`` from predicted quantiles if available."""

    def __init__(self, n_estimators: int = 8, random_state: int = 0, device: str = "cpu"):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, groups=None):
        from tabpfn import TabPFNRegressor                     # optional dependency
        self.model_ = TabPFNRegressor(n_estimators=self.n_estimators, device=self.device,
                                      random_state=self.random_state)
        self.model_.fit(np.asarray(X, np.float32), np.asarray(y, np.float32))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(X, np.float32)), float)

    def predict_std(self, X) -> np.ndarray:
        """(q84 - q16) / 2 of TabPFN's predictive distribution (NaN if unsupported)."""
        try:
            q = self.model_.predict(np.asarray(X, np.float32), output_type="quantiles",
                                    quantiles=[0.1587, 0.8413])
            return (np.asarray(q[1], float) - np.asarray(q[0], float)) / 2.0
        except Exception:                                      # noqa: BLE001
            return np.full(len(X), np.nan)


def build_tabpfn(params: dict, seed: int, ctx: dict):
    return TabPFNModel(random_state=seed, **params)


SPECS = [ModelSpec("tabpfn", build_tabpfn, "TabPFN (small-N benchmark, wear only)",
                   regimes=("run",), families=("wear",), requires=("tabpfn",), has_std=True)]
