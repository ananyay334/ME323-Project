"""Small MLP ensemble: 1-2 hidden layers (< 32 units), weight decay, dropout, early stopping."""
from __future__ import annotations

from ._torch import TorchEnsembleRegressor
from .registry import ModelSpec

MAX_UNITS, MAX_LAYERS = 31, 2


def make_mlp(n_in: int, hidden: tuple[int, ...], dropout: float, n_out: int = 1):
    import torch.nn as nn
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


class MLPEnsemble(TorchEnsembleRegressor):
    """Seed ensemble of small MLPs (predict_std = ensemble spread + validation noise)."""

    def __init__(self, hidden=(16,), dropout: float = 0.1, weight_decay: float = 1e-3,
                 lr: float = 3e-3, epochs: int = 600, patience: int = 60, batch_size: int = 512,
                 n_seeds: int = 5, val_frac: float = 0.2, random_state: int = 0):
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.n_seeds = n_seeds
        self.val_frac = val_frac
        self.random_state = random_state

    def _validate(self) -> None:
        h = tuple(self.hidden) if isinstance(self.hidden, (list, tuple)) else (int(self.hidden),)
        if not 1 <= len(h) <= MAX_LAYERS or max(h) > MAX_UNITS:
            raise ValueError(f"MLP must have 1-{MAX_LAYERS} hidden layers of < 32 units, got {h}")

    def _make_net(self, n_in: int):
        h = tuple(self.hidden) if isinstance(self.hidden, (list, tuple)) else (int(self.hidden),)
        return make_mlp(n_in, h, float(self.dropout))


def build_mlp(params: dict, seed: int, ctx: dict):
    if "hidden" in params:
        params["hidden"] = tuple(params["hidden"]) if isinstance(params["hidden"], (list, tuple)) \
            else (int(params["hidden"]),)
    return MLPEnsemble(random_state=seed, **params)


SPECS = [ModelSpec("mlp", build_mlp, "MLP seed ensemble (1-2 layers < 32 units, AdamW weight "
                   "decay, dropout, group-aware early stopping)",
                   targets=("log10_k", "log10_V", "COF_t"),          # plan: COF(t) and wear
                   requires=("torch",), has_std=True, needs_groups=True)]
