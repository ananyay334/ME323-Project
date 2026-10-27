"""Common model interface, OLS helpers and inner cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

INPUTS = ["lnL", "lnf", "lnt"]


class Model:
    """fit(X, y, groups) / predict(X) -> (mean, sd or None). X is a DataFrame with INPUTS (+ physics columns)."""

    name = "model"
    has_uncertainty = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None) -> "Model":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def n_params(self) -> int | str:
        return "?"


class Standardiser:
    def fit(self, A: np.ndarray) -> "Standardiser":
        self.mu = A.mean(axis=0)
        self.sd = A.std(axis=0)
        self.sd = np.where(self.sd > 1e-12, self.sd, 1.0)
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        return (A - self.mu) / self.sd


class OLS:
    """Ordinary least squares with prediction intervals (t distribution)."""

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> "OLS":
        n, p = Phi.shape
        self.coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        res = y - Phi @ self.coef
        self.dof = max(n - p, 1)
        self.s2 = float(res @ res / self.dof)
        self.XtX_inv = np.linalg.pinv(Phi.T @ Phi)
        return self

    def predict(self, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = Phi @ self.coef
        se_pred = np.sqrt(self.s2 * (1 + np.einsum("ij,jk,ik->i", Phi, self.XtX_inv, Phi)))
        # express the t-interval as an equivalent normal SD so all models share the +-1.96 SD interval
        sd = se_pred * stats.t.ppf(0.975, self.dof) / 1.959964
        return mean, sd


def inner_cv_splits(groups: np.ndarray | None, n: int, n_folds: int):
    g = np.arange(n) if groups is None else np.asarray(groups)
    k = min(n_folds, len(np.unique(g)))
    return list(GroupKFold(n_splits=k).split(np.zeros(n), groups=g))


def grid_search(make, grid: list[dict], X: np.ndarray, y: np.ndarray, groups, n_folds: int) -> dict:
    """Pick the grid point with the lowest inner GroupKFold MSE. `make(**params)` returns an sklearn regressor."""
    splits = inner_cv_splits(groups, len(y), n_folds)
    best, best_mse = grid[0], np.inf
    for params in grid:
        err = 0.0
        for tr, te in splits:
            m = make(**params).fit(X[tr], y[tr])
            err += float(np.sum((m.predict(X[te]) - y[te]) ** 2))
        mse = err / len(y)
        if mse < best_mse - 1e-15:
            best, best_mse = params, mse
    return best
