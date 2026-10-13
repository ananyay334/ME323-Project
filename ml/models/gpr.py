"""Gaussian-process regression: ConstantKernel × Matérn-5/2 (ARD) + WhiteKernel, normalize_y.

Kernel hyperparameters (signal variance, one length scale per input = ARD, noise) are
fitted by maximising the log marginal likelihood (LML) on the training fold, so the GPR
has no inner-CV search. ``predict_std`` is the predictive std of a new *observation*
(it includes the fitted white noise); ``predict_latent_std`` excludes it.

The hyperparameter search evaluates the LML and its gradient with :class:`FastLML`
(same kernel, same bounds, same restart starting points as sklearn, but the per-dimension
squared distances are computed once per fit instead of ~10 fresh n x n x d arrays per
optimizer step); the chosen hyperparameters are then handed to sklearn's
``GaussianProcessRegressor`` (``optimizer=None``), which provides the predictions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .registry import ModelSpec


def lbfgs(obj_func, initial_theta, bounds):
    """L-BFGS-B on the log marginal likelihood with ftol 1e-6 / gtol 1e-4.

    sklearn's default (ftol ~2e-9) gives the same optimum to 4 significant digits on this
    data at about twice the cost; the kernel, bounds and restarts are unchanged.
    """
    r = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True,
                                bounds=bounds, options={"ftol": 1e-6, "gtol": 1e-4, "maxiter": 500})
    return r.x, r.fun


class FastLML:
    """LML and gradient of C·Matérn-5/2(ARD) + White on normalised y, in sklearn's log-θ.

    θ = [log C, log ℓ_1 .. log ℓ_d, log σ²] (the order of ``make_kernel(d).theta``).
    """

    def __init__(self, Z: np.ndarray, y: np.ndarray):
        self.n, d = Z.shape
        diff = Z[:, None, :] - Z[None, :, :]
        self.P = (diff * diff).reshape(-1, d)          # squared distance per dimension, once
        self.y = y
        self.eye = np.eye(self.n)

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        n = self.n
        c, ls2, s2 = np.exp(theta[0]), np.exp(2 * theta[1:-1]), np.exp(theta[-1])
        t = np.sqrt(5.0 * (self.P @ (1.0 / ls2)))       # √5 r
        e = np.exp(-t)
        M = ((1.0 + t + t * t / 3.0) * e).reshape(n, n)
        K = c * M
        K[np.diag_indices(n)] += s2 + 1e-10           # sklearn's default alpha jitter
        try:
            L = scipy.linalg.cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf, np.zeros_like(theta)
        alpha = scipy.linalg.cho_solve((L, True), self.y, check_finite=False)
        lml = -0.5 * self.y @ alpha - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)
        W = np.outer(alpha, alpha) - scipy.linalg.cho_solve((L, True), self.eye, check_finite=False)
        grad = np.empty_like(theta)
        grad[0] = 0.5 * np.sum(W * K) - 0.5 * s2 * np.trace(W)          # ∂/∂log C  (C·M part)
        V = W.ravel() * ((1.0 + t) * e)
        grad[1:-1] = 0.5 * c * (5.0 / 3.0) * (self.P.T @ V) / ls2       # ∂/∂log ℓ_d
        grad[-1] = 0.5 * s2 * np.trace(W)                               # ∂/∂log σ²
        return float(lml), grad


def optimise_theta(kernel, Z: np.ndarray, y_norm: np.ndarray, n_restarts: int, random_state):
    """Maximise the LML from the kernel's θ plus ``n_restarts`` log-uniform starts (as sklearn)."""
    f = FastLML(Z, y_norm)

    def obj(theta):
        lml, g = f(theta)
        return (1e25, np.zeros_like(theta)) if not np.isfinite(lml) else (-lml, -g)

    bounds = kernel.bounds
    starts = [kernel.theta]
    rng = check_random_state(random_state)
    starts += [rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(int(n_restarts))]
    best = None
    for th0 in starts:
        x, fun = lbfgs(obj, th0, bounds)
        if best is None or fun < best[1]:
            best = (x, fun)
    return kernel.clone_with_theta(best[0])


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
        y = np.asarray(y, float)
        if self.kernel is None and self.nu == 2.5:       # fast path for the DOE's kernel
            kernel = make_kernel(Z.shape[1], self.nu, tuple(self.length_scale_bounds),
                                 tuple(self.noise_bounds))
            sd = y.std() or 1.0
            kernel = optimise_theta(kernel, Z, (y - y.mean()) / sd, self.n_restarts_optimizer,
                                    self.random_state)
            self.gp_ = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)
        else:
            kernel = self.kernel if self.kernel is not None else make_kernel(
                Z.shape[1], self.nu, tuple(self.length_scale_bounds), tuple(self.noise_bounds))
            self.gp_ = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=lbfgs,
                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp_.fit(Z, y)
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


SPECS = [ModelSpec("gpr", build_gpr, "GPR (primary): ARD Matérn-5/2 + white noise, normalize_y, "
                   "standardised X; hyperparameters by marginal likelihood", has_std=True)]
