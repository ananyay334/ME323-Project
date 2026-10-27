"""Gaussian-process regression: Constant x Matern(nu=2.5, ARD) + White, normalize_y, standardised inputs."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .base import INPUTS, Model, Standardiser


class GPR(Model):
    name = "GPR"
    has_uncertainty = True

    def __init__(self, cfg: dict, seed: int = 0, n_restarts: int | None = None, inputs: list[str] | None = None):
        self.cfg = cfg
        self.seed = seed
        self.n_restarts = cfg["n_restarts"] if n_restarts is None else n_restarts
        self.inputs = inputs or INPUTS

    def kernel(self):
        d = len(self.inputs)
        return (ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=self.cfg["nu"])
                + WhiteKernel(0.1, (1e-6, 1e1)))

    def fit(self, X, y, groups=None):
        A = X[self.inputs].to_numpy(float)
        y = np.asarray(y, float)
        self.sx = Standardiser().fit(A)
        self.flat = float(np.std(y)) < 1e-12
        self.ymu = float(np.mean(y))
        if self.flat:
            return self
        self.m = GaussianProcessRegressor(self.kernel(), normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                          random_state=self.seed).fit(self.sx.transform(A), y)
        return self

    def predict(self, X):
        if self.flat:
            return np.full(len(X), self.ymu), np.zeros(len(X))
        m, sd = self.m.predict(self.sx.transform(X[self.inputs].to_numpy(float)), return_std=True)
        return m, sd

    @property
    def length_scales(self) -> dict:
        if self.flat:
            return {k: np.nan for k in self.inputs}
        ls = np.atleast_1d(self.m.kernel_.k1.k2.length_scale)
        return dict(zip(self.inputs, ls))

    def n_params(self):
        return len(self.inputs) + 2  # length scales + amplitude + noise (hyper-parameters)
