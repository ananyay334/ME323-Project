"""Groups and splits on the training rows (test runs never enter a split).

Group = ``Replicate_of`` (followed to the root of a chain) when set, else ``Experiment_ID``:
R04/R14/R20 form one group, and both checkpoints of a run share their run's group.

* outer: 5-fold ``GroupKFold`` with the groups shuffled by seed (the group labels are
  relabelled by a seeded permutation before the split), one repeat per seed in
  ``cv.seeds``. Folds are assigned per group once, so every target sees the same runs in
  each test fold.
* inner: ``GroupKFold(cv.inner_folds)`` on the outer training groups (nested tuning).
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def assign_groups(runs: pd.DataFrame) -> pd.Series:
    """Group label per row: ``Replicate_of`` (followed to the root of a chain) or own ID."""
    ids = runs["Experiment_ID"].astype(str)
    rep = runs.get("Replicate_of", pd.Series(np.nan, index=runs.index))
    parent = {}
    for i, r in zip(ids, rep):
        if isinstance(r, str) and r.strip() not in ("", "nan"):
            parent[i] = r.strip()

    def root(i: str) -> str:
        seen = set()
        while parent.get(i) and i not in seen:
            seen.add(i)
            i = parent[i]
        return i

    return pd.Series([root(i) for i in ids], index=runs.index, name="group")


def shuffled_group_kfold(groups: np.ndarray, n_splits: int, seed: int) -> dict[str, int]:
    """Group -> fold: GroupKFold on seed-permuted group labels."""
    ug = np.unique(np.asarray(groups).astype(str))
    perm = np.random.default_rng(seed).permutation(len(ug))
    code = dict(zip(ug, perm))
    g = np.array([code[x] for x in np.asarray(groups).astype(str)])
    k = min(n_splits, len(ug))
    out: dict[str, int] = {}
    for f, (_, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros((len(g), 1)), groups=g)):
        for x in set(np.asarray(groups).astype(str)[te]):
            out[x] = f
    return out


class SplitPlan:
    """Group -> fold maps for every outer repeat, plus inner GroupKFold splits."""

    def __init__(self, groups: np.ndarray, cfg: dict):
        c = cfg["cv"]
        self.groups = np.asarray(groups).astype(str)
        self.n_splits = int(c.get("n_splits", 5))
        self.seeds = [int(s) for s in c.get("seeds", [0])]
        self.n_inner = int(c.get("inner_folds", 4))
        self.fold_maps = [shuffled_group_kfold(self.groups, self.n_splits, s) for s in self.seeds]

    def outer(self, groups: np.ndarray, repeat: int = 0) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(fold, train_idx, test_idx)`` over rows labelled with ``groups``."""
        fm = self.fold_maps[repeat]
        groups = np.asarray(groups).astype(str)
        missing = set(groups) - set(fm)
        if missing:
            raise KeyError(f"groups not in the split plan: {sorted(missing)[:5]}")
        fold = np.array([fm[g] for g in groups])
        for f in range(max(fm.values()) + 1):
            te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            if len(te) and len(tr):
                yield f, tr, te

    def inner(self, groups_train: np.ndarray, n_inner: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Inner GroupKFold over the rows of an outer training set."""
        g = np.asarray(groups_train).astype(str)
        k = min(int(n_inner or self.n_inner), len(set(g)))
        if k < 2:
            return []
        return list(GroupKFold(n_splits=k).split(np.zeros((len(g), 1)), groups=g))


def check_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise if any group has rows on both sides of a split."""
    groups = np.asarray(groups).astype(str)
    both = set(groups[train_idx]) & set(groups[test_idx])
    if both:
        raise AssertionError(f"group leakage across train/test: {sorted(both)[:5]}")
