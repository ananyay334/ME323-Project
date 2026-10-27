"""Comparison models on ln L, ln f, ln t: constant, power law (OLS), Ridge, SVR, shallow XGBoost."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from .base import INPUTS, OLS, Model, Standardiser, grid_search


class Constant(Model):
    name = "Constant"
    has_uncertainty = True

    def fit(self, X, y, groups=None):
        y = np.asarray(y, float)
        self.ols = OLS().fit(np.ones((len(y), 1)), y)
        return self

    def predict(self, X):
        return self.ols.predict(np.ones((len(X), 1)))

    def n_params(self):
        return 1


class PowerLaw(Model):
    """y = b0 + b1 ln L + b2 ln f + b3 ln t (a power law when y is a log)."""

    name = "Power law (OLS)"
    has_uncertainty = True

    def __init__(self, inputs: list[str] | None = None):
        self.inputs = inputs or INPUTS

    def _phi(self, X):
        return np.column_stack([np.ones(len(X)), X[self.inputs].to_numpy()])

    def fit(self, X, y, groups=None):
        self.ols = OLS().fit(self._phi(X), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.ols.predict(self._phi(X))

    @property
    def coef(self) -> dict:
        return dict(zip(["b0"] + self.inputs, self.ols.coef))

    def n_params(self):
        return 1 + len(self.inputs)


class _Scaled(Model):
    """Standardise inputs (and target) inside the training fold, tune by inner GroupKFold."""

    def __init__(self, cfg: dict, inner_folds: int, seed: int):
        self.cfg = cfg
        self.inner_folds = inner_folds
        self.seed = seed

    def _prep(self, X, y=None, fit=False):
        A = X[INPUTS].to_numpy()
        if fit:
            self.sx = Standardiser().fit(A)
            self.ymu, self.ysd = float(np.mean(y)), float(np.std(y)) or 1.0
        return self.sx.transform(A)


class RidgeCV(_Scaled):
    name = "Ridge"

    def fit(self, X, y, groups=None):
        A = self._prep(X, y, fit=True)
        t = (np.asarray(y) - self.ymu) / self.ysd
        grid = [{"alpha": a} for a in self.cfg["alpha"]]
        self.best = grid_search(lambda **p: Ridge(**p), grid, A, t, groups, self.inner_folds)
        self.m = Ridge(**self.best).fit(A, t)
        return self

    def predict(self, X):
        return self.m.predict(self._prep(X)) * self.ysd + self.ymu, None

    def n_params(self):
        return 4


class SVRCV(_Scaled):
    name = "SVR"

    def fit(self, X, y, groups=None):
        A = self._prep(X, y, fit=True)
        t = (np.asarray(y) - self.ymu) / self.ysd
        grid = [{"C": c, "gamma": g, "epsilon": e} for c, g, e in
                itertools.product(self.cfg["C"], self.cfg["gamma"], self.cfg["epsilon"])]
        self.best = grid_search(lambda **p: SVR(kernel="rbf", **p), grid, A, t, groups, self.inner_folds)
        self.m = SVR(kernel="rbf", **self.best).fit(A, t)
        return self

    def predict(self, X):
        return self.m.predict(self._prep(X)) * self.ysd + self.ymu, None

    def n_params(self):
        return f"{len(self.m.support_)} SV" if hasattr(self, "m") else "SVs"


def xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


class XGBCV(_Scaled):
    name = "XGBoost"

    def fit(self, X, y, groups=None):
        from xgboost import XGBRegressor

        A = self._prep(X, y, fit=True)
        t = (np.asarray(y) - self.ymu) / self.ysd

        def make(**p):
            return XGBRegressor(subsample=self.cfg["subsample"], random_state=self.seed, n_jobs=1, verbosity=0, **p)

        grid = [{"max_depth": d, "n_estimators": n, "learning_rate": lr} for d, n, lr in
                itertools.product(self.cfg["max_depth"], self.cfg["n_estimators"], self.cfg["learning_rate"])]
        self.best = grid_search(make, grid, A, t, groups, self.inner_folds)
        self.m = make(**self.best).fit(A, t)
        return self

    def predict(self, X):
        return self.m.predict(self._prep(X)) * self.ysd + self.ymu, None

    def n_params(self):
        return f"{self.best['n_estimators']} trees, depth {self.best['max_depth']}" if hasattr(self, "best") else "trees"
