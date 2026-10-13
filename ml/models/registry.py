"""Model registry: one :class:`ModelSpec` per model.

Interface (sklearn estimator protocol; every model is ``clone``-able):

    est = spec.build(params, seed, ctx)      # params: fixed + grid-selected hyperparameters
    est.fit(X: pd.DataFrame, y: np.ndarray)
    est.predict(X) -> np.ndarray
    est.predict_std(X) -> np.ndarray         # only if spec.has_std

Every model runs on every regime and both feature sets. ``selectable=False`` marks a
reference baseline that is reported but never chosen as "best".

Adding a model: write ``ml/models/<name>.py`` with ``SPECS = [ModelSpec(...)]`` (import
optional dependencies inside the build function), append the module to
``MODEL_MODULES`` and add a ``models: <name>:`` block (params / grid) to the config.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..config import has_module

log = logging.getLogger("ml.models")

MODEL_MODULES = [
    "ml.models.gpr",
    "ml.models.tabpfn_model",
    "ml.models.svr",
    "ml.models.trees",
    "ml.models.linear",
]


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a model."""
    name: str
    build: Callable[[dict, int, dict], Any]       # (params, seed, ctx) -> unfitted estimator
    description: str = ""
    requires: tuple[str, ...] = ()                # optional modules that must all be installed
    has_std: bool = False
    selectable: bool = True                       # False: reference baseline, never "best"

    def missing(self) -> list[str]:
        """Missing optional dependencies (empty list = available)."""
        return [m for m in self.requires if not has_module(m)]


_REGISTRY: dict[str, ModelSpec] | None = None


def registry() -> dict[str, ModelSpec]:
    """All registered models (modules are imported once; optional deps are not)."""
    global _REGISTRY
    if _REGISTRY is None:
        reg: dict[str, ModelSpec] = {}
        for mod in MODEL_MODULES:
            for spec in importlib.import_module(mod).SPECS:
                if spec.name in reg:
                    raise ValueError(f"duplicate model name {spec.name!r} in {mod}")
                reg[spec.name] = spec
        _REGISTRY = reg
    return _REGISTRY


def get_spec(name: str) -> ModelSpec:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(reg)}")
    return reg[name]


def select_models(cfg: dict, names: list[str] | None = None) -> tuple[list[ModelSpec], dict[str, str]]:
    """Enabled + available models (optionally restricted to ``names``) and the skipped ones.

    A model whose optional dependency is missing is skipped with a loud warning, never an error.
    """
    reg = registry()
    if names:
        unknown = [n for n in names if n not in reg]
        if unknown:
            raise KeyError(f"unknown model(s) {unknown}; registered: {sorted(reg)}")
    chosen, skipped = [], {}
    for name, spec in reg.items():
        if names and name not in names:
            continue
        if not names and not (cfg["models"].get(name) or {}).get("enabled", False):
            continue
        miss = spec.missing()
        if miss:
            skipped[name] = f"missing optional dependency {', '.join(miss)}"
            log.warning("!!! model %s NOT RUN (%s) - install it to include this model !!!",
                        name, skipped[name])
            continue
        chosen.append(spec)
    return chosen, skipped


def build_estimator(spec: ModelSpec, params: dict, seed: int, ctx: dict | None = None):
    return spec.build(dict(params), int(seed), dict(ctx or {}))
