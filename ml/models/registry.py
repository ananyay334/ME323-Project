"""Model registry: one :class:`ModelSpec` per model.

Interface (sklearn estimator protocol; every model is ``clone``-able):

    est = spec.build(params, seed, ctx)      # params: fixed + sampled hyperparameters
    est.fit(X: pd.DataFrame, y: np.ndarray[, groups=np.ndarray])   # groups if spec.needs_groups
    est.predict(X) -> np.ndarray
    est.predict_std(X) -> np.ndarray         # only if spec.has_std (predictive std, same units as y)

Joint multi-task models (``regimes == ("joint",)``) implement the richer interface
documented in :mod:`ml.models.multitask` and are evaluated by
:func:`ml.evaluate.cross_validate_joint`.

Adding a model
--------------
1. write ``ml/models/<name>.py`` defining ``SPECS = [ModelSpec(...)]`` (import any
   optional dependency *inside* the build function, never at module level);
2. append the module to ``MODEL_MODULES`` below;
3. add a ``models: <name>:`` block (enabled / params / search) to the config.
Splits, tuning, metrics, reporting and the final test pick it up automatically.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..config import has_module

log = logging.getLogger("ml.models")

MODEL_MODULES = [
    "ml.models.linear",
    "ml.models.trees",
    "ml.models.svr",
    "ml.models.gpr",
    "ml.models.mlp",
    "ml.models.kan",
    "ml.models.multitask",
    "ml.models.tabpfn_model",
    "ml.models.symbolic",
]


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a model."""
    name: str
    build: Callable[[dict, int, dict], Any]       # (params, seed, ctx) -> unfitted estimator
    description: str = ""
    regimes: tuple[str, ...] = ("run", "time")    # "run" | "time" | "joint"
    families: tuple[str, ...] | None = None       # allowed target families; None = all
    targets: tuple[str, ...] | None = None        # allowed target names; None = all
    requires: tuple[str, ...] = ()                # optional modules that must all be installed
    requires_any: tuple[tuple[str, ...], ...] = ()  # per tuple: at least one must be installed
    has_std: bool = False
    needs_groups: bool = False

    def missing(self) -> list[str]:
        """Missing optional dependencies (empty list = available)."""
        miss = [m for m in self.requires if not has_module(m)]
        for alts in self.requires_any:
            if not any(has_module(m) for m in alts):
                miss.append(" or ".join(alts))
        return miss

    def supports(self, regime: str, family: str, target: str | None = None) -> bool:
        return (regime in self.regimes
                and (self.families is None or family in self.families)
                and (self.targets is None or target is None or target in self.targets))


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

    A model whose optional dependency is missing is skipped with a warning, never an error.
    """
    reg = registry()
    if names:
        unknown = [n for n in names if n not in reg]
        if unknown:
            raise KeyError(f"unknown model(s) {unknown}; registered: {sorted(reg)}")
    chosen, skipped = [], {}
    for name, spec in reg.items():
        mcfg = cfg["models"].get(name, {})
        if names and name not in names:
            continue
        if not names and not mcfg.get("enabled", False):
            continue
        miss = spec.missing()
        if miss:
            skipped[name] = f"missing optional dependency: {', '.join(miss)}"
            log.warning("model %s skipped (%s)", name, skipped[name])
            continue
        chosen.append(spec)
    return chosen, skipped


def model_params(cfg: dict, name: str) -> dict:
    return dict((cfg["models"].get(name) or {}).get("params") or {})


def build_estimator(spec: ModelSpec, params: dict, seed: int, ctx: dict | None = None):
    return spec.build(dict(params), int(seed), dict(ctx or {}))
