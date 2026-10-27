"""TabPFN regressor (optional). Needs `pip install tabpfn` (pulls torch) and its model weights, which are
downloaded on first use after a one-time licence acceptance (set TABPFN_TOKEN). If it cannot run, the model is
reported as "not run" and every evaluation simply skips it.
"""

from __future__ import annotations

import numpy as np

from .base import INPUTS, Model

_STATUS: dict = {}


def tabpfn_status() -> tuple[bool, str]:
    """(available, reason). Tries one tiny fit so a missing licence / weights is caught once, up front."""
    if "ok" in _STATUS:
        return _STATUS["ok"], _STATUS["reason"]
    try:
        from tabpfn import TabPFNRegressor

        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 3))
        TabPFNRegressor(device="cpu", random_state=0).fit(X, X[:, 0]).predict(X[:2])
        _STATUS.update(ok=True, reason="ok")
    except ImportError:
        _STATUS.update(ok=False, reason="not run (install tabpfn)")
    except Exception as exc:  # licence / weights download / torch problems
        msg = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
        _STATUS.update(ok=False, reason=f"not run (tabpfn installed but cannot load weights: {type(exc).__name__}: {msg})")
    return _STATUS["ok"], _STATUS["reason"]


class TabPFN(Model):
    name = "TabPFN"
    has_uncertainty = True

    def __init__(self, cfg: dict, seed: int = 0, inputs: list[str] | None = None):
        self.cfg = cfg
        self.seed = seed
        self.inputs = inputs or INPUTS

    def fit(self, X, y, groups=None):
        from tabpfn import TabPFNRegressor

        y = np.asarray(y, float)
        self.flat = float(np.std(y)) < 1e-12
        self.ymu = float(np.mean(y))
        if not self.flat:
            self.m = TabPFNRegressor(device=self.cfg.get("device", "cpu"), random_state=self.seed)
            self.m.fit(X[self.inputs].to_numpy(float), y)
        return self

    def predict(self, X):
        if self.flat:
            return np.full(len(X), self.ymu), np.zeros(len(X))
        A = X[self.inputs].to_numpy(float)
        mean = np.asarray(self.m.predict(A), float)
        try:
            q = self.m.predict(A, output_type="quantiles", quantiles=[0.025, 0.975])
            sd = (np.asarray(q[1]) - np.asarray(q[0])) / (2 * 1.959964)
        except Exception:
            sd = None
        return mean, sd

    def n_params(self):
        return "pre-trained (~7M)"
