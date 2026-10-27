"""Physics-informed hybrid: physics baseline + ML model fitted on its residuals (inputs ln L, ln f, ln t)."""

from __future__ import annotations

import numpy as np

from .base import Model
from .physics import LinearPhysics


class Hybrid(Model):
    has_uncertainty = True

    def __init__(self, physics: LinearPhysics, residual: Model):
        self.physics = physics
        self.residual = residual
        self.name = f"{residual.name} + physics"

    def fit(self, X, y, groups=None):
        y = np.asarray(y, float)
        self.physics.fit(X, y)
        base, _ = self.physics.predict(X)
        self.residual.fit(X, y - base, groups)
        return self

    def predict(self, X):
        base, _ = self.physics.predict(X)
        r, sd = self.residual.predict(X)
        return base + r, sd

    def n_params(self):
        a, b = self.physics.n_params(), self.residual.n_params()
        return a + b if isinstance(a, int) and isinstance(b, int) else f"{a} + {b}"
