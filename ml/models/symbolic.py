"""Symbolic regression with PySR (optional; needs ``pip install pysr`` and Julia).

Used (a) as an optional CV model for the wear targets (``models.pysr.enabled``) and
(b) post hoc by :func:`ml.interpret.symbolic_analysis`, which fits it on all CV runs
and compares the discovered expressions with Archard's law (V = k F S, k constant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .registry import ModelSpec


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """PySR wrapper with deterministic, serial search."""

    def __init__(self, niterations: int = 40, maxsize: int = 20, populations: int = 15,
                 binary_operators=("+", "-", "*", "/"), unary_operators=("log", "exp", "square"),
                 random_state: int = 0):
        self.niterations = niterations
        self.maxsize = maxsize
        self.populations = populations
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        from pysr import PySRRegressor                         # optional dependency (Julia)
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.model_ = PySRRegressor(
            niterations=self.niterations, maxsize=self.maxsize, populations=self.populations,
            binary_operators=list(self.binary_operators), unary_operators=list(self.unary_operators),
            random_state=self.random_state, deterministic=True, parallelism="serial",
            progress=False, verbosity=0, temp_equation_file=True)
        Xa = np.asarray(X, float)
        Xa = np.where(np.isfinite(Xa), Xa, np.nanmedian(Xa, axis=0))
        self.model_.fit(Xa, np.asarray(y, float), variable_names=self.feature_names_)
        return self

    def predict(self, X) -> np.ndarray:
        Xa = np.asarray(X, float)
        return np.asarray(self.model_.predict(np.where(np.isfinite(Xa), Xa, 0.0)), float)

    def equations(self) -> pd.DataFrame:
        eq = self.model_.equations_
        return eq[[c for c in ("complexity", "loss", "score", "equation") if c in eq]].copy()


def build_pysr(params: dict, seed: int, ctx: dict):
    for k in ("binary_operators", "unary_operators"):
        if k in params:
            params[k] = tuple(params[k])
    return SymbolicRegressor(random_state=seed, **params)


SPECS = [ModelSpec("pysr", build_pysr, "PySR symbolic regression (optional, wear)",
                   regimes=("run",), families=("wear",), requires=("pysr",))]
