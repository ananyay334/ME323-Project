"""ANN ensemble: sklearn MLPRegressor, one tanh hidden layer, lbfgs, L2 alpha.

Hidden size and alpha are chosen by inner GroupKFold on the training fold (with a smaller ensemble for speed);
the reported prediction is the mean of `n_seeds` nets, the uncertainty their SD.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import INPUTS, Model, Standardiser, inner_cv_splits


def _mlp(h: int, alpha: float, seed: int, max_iter: int) -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(h,), activation="tanh", solver="lbfgs", alpha=alpha,
                        max_iter=max_iter, random_state=seed)


class ANNEnsemble(Model):
    name = "ANN"
    has_uncertainty = True

    def __init__(self, cfg: dict, seed: int = 0, fixed: dict | None = None, n_seeds: int | None = None,
                 inputs: list[str] | None = None):
        self.cfg = cfg
        self.seed = seed
        self.fixed = fixed
        self.n_seeds = n_seeds or cfg["n_seeds"]
        self.inputs = inputs or INPUTS

    def _ens_predict(self, A, nets):
        return np.array([m.predict(A) for m in nets])

    def fit(self, X, y, groups=None):
        A = X[self.inputs].to_numpy(float)
        y = np.asarray(y, float)
        self.sx = Standardiser().fit(A)
        A = self.sx.transform(A)
        self.ymu, self.ysd = float(y.mean()), float(y.std())
        self.flat = self.ysd < 1e-12
        if self.flat:  # nothing to learn (e.g. zero residuals)
            self.best = self.fixed or {"hidden": self.cfg["hidden"][0], "alpha": self.cfg["alpha"][0]}
            return self
        t = (y - self.ymu) / self.ysd
        if self.fixed is not None:
            self.best = dict(self.fixed)
        else:
            splits = inner_cv_splits(groups, len(t), self.cfg["inner_folds"])
            best, best_mse = None, np.inf
            for h, a in itertools.product(self.cfg["hidden"], self.cfg["alpha"]):
                err = 0.0
                for tr, te in splits:
                    nets = [_mlp(h, a, self.seed + s, self.cfg["max_iter"]).fit(A[tr], t[tr])
                            for s in range(self.cfg["n_inner_seeds"])]
                    err += float(np.sum((self._ens_predict(A[te], nets).mean(0) - t[te]) ** 2))
                if err < best_mse - 1e-12:
                    best, best_mse = {"hidden": h, "alpha": a}, err
            self.best = best
        self.nets = [_mlp(self.best["hidden"], self.best["alpha"], self.seed + s, self.cfg["max_iter"]).fit(A, t)
                     for s in range(self.n_seeds)]
        return self

    def predict(self, X):
        if self.flat:
            return np.full(len(X), self.ymu), np.zeros(len(X))
        P = self._ens_predict(self.sx.transform(X[self.inputs].to_numpy(float)), self.nets) * self.ysd + self.ymu
        return P.mean(0), P.std(0)

    def n_params(self):
        h = self.best["hidden"] if hasattr(self, "best") else max(self.cfg["hidden"])
        return (len(self.inputs) + 1) * h + h + 1
