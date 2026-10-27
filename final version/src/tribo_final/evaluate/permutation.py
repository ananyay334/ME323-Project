"""Permutation tests: is the P1 cross-validated R² better than with a shuffled target?

p = (1 + #{R²_perm >= R²_obs}) / (n_perm + 1). The physics baselines (linear least squares) are done exactly and
vectorised over all shuffles; ML hybrids refit inside every LOGO fold of every shuffle with their
hyper-parameters fixed at the values chosen on the real data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..models.physics import LinearPhysics
from ..utils import quiet_warnings
from .cv import protocol_splits


def _r2_cols(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return 1 - np.sum((Y - P) ** 2, axis=0) / np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)


def shuffles(y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(y) for _ in range(n_perm)])


def perm_test_physics(kind: str, df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    Phi, off = LinearPhysics(kind).design(df)
    y = df["y"].to_numpy(float)
    Y = np.column_stack([y, shuffles(y, n_perm, seed)])
    P = np.zeros_like(Y)
    for tr, te in protocol_splits(df, "P1"):
        B, *_ = np.linalg.lstsq(Phi[tr], Y[tr] - off[tr, None], rcond=None)
        P[te] = Phi[te] @ B + off[te, None]
    r2 = _r2_cols(Y, P)
    return {"R2_obs": r2[0], "p_value": (1 + np.sum(r2[1:] >= r2[0])) / (n_perm + 1), "n_perm": n_perm,
            "R2_perm_mean": float(r2[1:].mean()), "R2_perm_p95": float(np.percentile(r2[1:], 95))}


def _cv_r2_chunk(factory: Callable, df: pd.DataFrame, Ycols: np.ndarray, splits) -> np.ndarray:
    quiet_warnings()
    out = []
    for j in range(Ycols.shape[1]):
        y = Ycols[:, j]
        p = np.zeros_like(y)
        for tr, te in splits:
            m = factory().fit(df.iloc[tr], y[tr], df["group"].to_numpy()[tr])
            p[te] = m.predict(df.iloc[te])[0]
        out.append(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return np.array(out)


def perm_test_model(factory: Callable, df: pd.DataFrame, n_perm: int, seed: int, n_jobs: int, r2_obs: float | None = None) -> dict:
    y = df["y"].to_numpy(float)
    splits = protocol_splits(df, "P1")
    Y = shuffles(y, n_perm, seed)
    if r2_obs is None:
        r2_obs = float(_cv_r2_chunk(factory, df, y[:, None], splits)[0])
    chunks = np.array_split(np.arange(n_perm), max(1, min(n_perm, 64)))
    res = Parallel(n_jobs=n_jobs)(delayed(_cv_r2_chunk)(factory, df, Y[:, c], splits) for c in chunks if len(c))
    r2p = np.concatenate(res)
    return {"R2_obs": r2_obs, "p_value": (1 + np.sum(r2p >= r2_obs)) / (n_perm + 1), "n_perm": n_perm,
            "R2_perm_mean": float(r2p.mean()), "R2_perm_p95": float(np.percentile(r2p, 95))}
