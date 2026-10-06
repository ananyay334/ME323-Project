"""Gaussian-process regression: anisotropic Matérn-5/2 + WhiteKernel, normalize_y.

Kernel hyperparameters (signal variance, one length scale per input = ARD, noise) are
fitted by maximising the marginal likelihood on the training fold, so the GPR has no
inner-CV search. ``predict_std`` is the predictive std of a new *observation* (it
includes the fitted white noise); ``predict_latent_std`` excludes it (used by the
active-learning acquisition, since repeating a condition cannot remove noise).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .registry import ModelSpec


def make_kernel(n_features: int, nu: float = 2.5, length_scale_bounds=(1e-2, 1e3),
                noise_bounds=(1e-6, 1.0)):
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_features), length_scale_bounds=length_scale_bounds, nu=nu)
            + WhiteKernel(1e-2, noise_bounds))


class GPRModel(BaseEstimator, RegressorMixin):
    """Imputer + scaler (fitted on the training fold) + ARD Matérn GPR."""

    def __init__(self, nu: float = 2.5, n_restarts_optimizer: int = 3,
                 length_scale_bounds: tuple = (1e-2, 1e3), noise_bounds: tuple = (1e-6, 1.0),
                 random_state: int = 0, kernel=None):
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.length_scale_bounds = length_scale_bounds
        self.noise_bounds = noise_bounds
        self.random_state = random_state
        self.kernel = kernel                       # optional warm start (e.g. active learning)

    def fit(self, X, y, groups=None):
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(self.imputer_.fit_transform(np.asarray(X, float)))
        kernel = self.kernel if self.kernel is not None else make_kernel(
            Z.shape[1], self.nu, tuple(self.length_scale_bounds), tuple(self.noise_bounds))
        self.gp_ = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=self.n_restarts_optimizer,
                                            random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp_.fit(Z, np.asarray(y, float))
        return self

    def _z(self, X):
        return self.scaler_.transform(self.imputer_.transform(np.asarray(X, float)))

    def predict(self, X, return_std: bool = False):
        return self.gp_.predict(self._z(X), return_std=return_std)

    def predict_std(self, X) -> np.ndarray:
        return self.gp_.predict(self._z(X), return_std=True)[1]

    @property
    def noise_var_(self) -> float:
        """Fitted white-noise variance in the units of y."""
        return float(self.gp_.kernel_.k2.noise_level * self.gp_._y_train_std ** 2)

    def predict_latent_std(self, X) -> np.ndarray:
        sd = self.predict_std(X)
        return np.sqrt(np.maximum(sd ** 2 - self.noise_var_, 0.0))

    @property
    def length_scales_(self) -> dict:
        """ARD length scales in standardised input units (small = relevant)."""
        ls = np.atleast_1d(self.gp_.kernel_.k1.k2.length_scale)
        names = self.feature_names_ or [f"x{i}" for i in range(len(ls))]
        return dict(zip(names, map(float, ls)))


def build_gpr(params: dict, seed: int, ctx: dict):
    return GPRModel(random_state=seed, **params)


SPECS = [ModelSpec("gpr", build_gpr, "GPR, anisotropic Matérn-5/2 + white noise, normalize_y "
                   "(hyperparameters by marginal likelihood)", has_std=True)]
