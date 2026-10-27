"""Physics baselines.

COF          COF = mu0 + c * L^(-1/3)       Hertz: contact area ~ L^(2/3), adhesion term ~ L^(-1/3)   (mu0, c by OLS)
Wear volume  log10 V = log10 k + log10(L S)  Archard with slope fixed at 1                          (intercept only)
Track width  log10 w = a + b ln L            Hertz: w ~ L^(1/3) -> b = (1/3) / ln 10 = 0.145            (a, b)
Wear rate    log10 k = const                 Archard: k independent of the test settings              (intercept only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import OLS, Model

HERTZ_WIDTH_B = (1 / 3) / np.log(10)


class LinearPhysics(Model):
    """y - offset(X) = design(X) @ beta, fitted by OLS (with prediction intervals)."""

    has_uncertainty = True

    def __init__(self, kind: str):
        self.kind = kind
        self.name = PHYSICS_LABEL[kind]

    def design(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        one = np.ones(n)
        if self.kind == "cof":
            return np.column_stack([one, np.exp(X["lnL"].to_numpy()) ** (-1 / 3)]), np.zeros(n)
        if self.kind == "archard":
            return one[:, None], X["log10_LS"].to_numpy()
        if self.kind == "width":
            return np.column_stack([one, X["lnL"].to_numpy()]), np.zeros(n)
        if self.kind == "constant_k":
            return one[:, None], np.zeros(n)
        raise ValueError(self.kind)

    def fit(self, X, y, groups=None):
        Phi, off = self.design(X)
        self.ols = OLS().fit(Phi, np.asarray(y, float) - off)
        return self

    def predict(self, X):
        Phi, off = self.design(X)
        m, sd = self.ols.predict(Phi)
        return m + off, sd

    @property
    def params(self) -> dict:
        c = self.ols.coef
        names = PARAM_NAMES[self.kind]
        return dict(zip(names, c))

    def n_params(self):
        return len(PARAM_NAMES[self.kind])


PHYSICS_LABEL = {
    "cof": "Physics: μ0 + c·L^-1/3",
    "archard": "Physics: Archard (slope 1)",
    "width": "Physics: a + b·ln L",
    "constant_k": "Physics: Archard (constant k)",
}
PARAM_NAMES = {
    "cof": ["mu0", "c"],
    "archard": ["log10_k"],
    "width": ["a", "b"],
    "constant_k": ["log10_k"],
}


def physics_param_ols_matrices(kind: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix and offset (used by the vectorised permutation test)."""
    return LinearPhysics(kind).design(X)
