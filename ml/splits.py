"""Groups and deterministic (nested) group K-fold splits.

Group = ``Replicate_of`` when it is filled, else ``Experiment_ID``, so a replicate
always shares a group with its original condition, and all seconds of one run share
the run's group. Folds are assigned per *group* from a seeded shuffle (optionally
stratified by carbon level), once per repeat, and every table (runs, COF(t), joint)
maps its rows through the same group -> fold table. Hence a run sits in the same
outer test fold for every target and model, which keeps the comparisons paired.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .config import stable_seed


def assign_groups(runs: pd.DataFrame) -> pd.Series:
    """Group label per run: ``Replicate_of`` (followed to the root of a chain) or own ID."""
    ids = runs["Experiment_ID"].astype(str)
    rep = runs.get("Replicate_of", pd.Series(np.nan, index=runs.index))
    parent = {i: (str(r).strip() if isinstance(r, str) and str(r).strip() not in ("", "nan") else None)
              for i, r in zip(ids, rep)}

    def root(i: str) -> str:
        seen = set()
        while parent.get(i) and i not in seen:
            seen.add(i)
            i = parent[i]
        return i

    return pd.Series([root(i) for i in ids], index=runs.index, name="group")


def assign_group_folds(groups: np.ndarray | list, n_splits: int, seed: int,
                       strata: dict | None = None) -> dict:
    """Deterministic group -> fold map.

    Groups are shuffled with ``seed``; with ``strata`` (group -> stratum) the shuffled
    groups are dealt stratum by stratum, round-robin, so every fold receives an even
    share of each stratum (e.g. each carbon level).
    """
    uniq = sorted(set(map(str, groups)))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(uniq))
    if strata:
        by: dict = {}
        for g in order:
            by.setdefault(strata.get(g), []).append(g)
        order = [g for s in sorted(by, key=lambda s: (s is None, str(s))) for g in by[s]]
    offset = int(rng.integers(n_splits))
    return {g: (i + offset) % n_splits for i, g in enumerate(order)}


class FoldPlan:
    """Outer folds (per repeat) and inner folds (per repeat x outer fold), all by group."""

    def __init__(self, groups: pd.Series | np.ndarray, n_outer: int = 5, n_inner: int = 4,
                 repeats: int = 1, seed: int = 0, strata: dict | None = None):
        self.groups = sorted(set(map(str, np.asarray(groups))))
        self.n_outer = min(n_outer, len(self.groups))
        self.n_inner = n_inner
        self.repeats = repeats
        self.seed = seed
        self.strata = strata
        if self.n_outer < 2:
            raise ValueError(f"need at least 2 groups for CV, got {len(self.groups)}")
        self._maps = [assign_group_folds(self.groups, self.n_outer, stable_seed(seed, "outer", r),
                                         strata) for r in range(repeats)]

    def fold_map(self, repeat: int) -> dict:
        return self._maps[repeat]

    def outer(self, groups: np.ndarray, repeat: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(fold, train_idx, test_idx)`` over rows labelled with ``groups``."""
        groups = np.asarray(groups).astype(str)
        fm = self._maps[repeat]
        missing = set(groups) - set(fm)
        if missing:
            raise KeyError(f"groups not in the fold plan (holdout leak?): {sorted(missing)[:5]}")
        fold = np.array([fm[g] for g in groups])
        for f in range(self.n_outer):
            test = np.flatnonzero(fold == f)
            train = np.flatnonzero(fold != f)
            if len(test) and len(train):
                yield f, train, test

    def inner(self, groups_train: np.ndarray, repeat: int, fold: int,
              n_inner: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Inner group K-fold over the rows of an outer training set."""
        return inner_splits(groups_train, n_inner or self.n_inner,
                            stable_seed(self.seed, "inner", repeat, fold), self.strata)


def inner_splits(groups: np.ndarray, n_splits: int, seed: int,
                 strata: dict | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group K-fold over ``groups`` (deterministic); fewer folds if there are few groups."""
    groups = np.asarray(groups).astype(str)
    n = min(n_splits, len(set(groups)))
    if n < 2:
        return []
    fm = assign_group_folds(groups, n, seed, strata)
    fold = np.array([fm[g] for g in groups])
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(n)
            if (fold == f).any() and (fold != f).any()]


def check_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise if any group has rows on both sides of a split."""
    groups = np.asarray(groups).astype(str)
    both = set(groups[train_idx]) & set(groups[test_idx])
    if both:
        raise AssertionError(f"group leakage across train/test: {sorted(both)[:5]}")


def plan_from_data(runs_cv: pd.DataFrame, cfg: dict) -> FoldPlan:
    """FoldPlan over the CV runs' groups using the ``cv`` section of the config."""
    c = cfg["cv"]
    strata = None
    if c.get("stratify") and c["stratify"] in runs_cv:
        strata = runs_cv.groupby("group")[c["stratify"]].first().astype(str).to_dict()
    return FoldPlan(runs_cv["group"], c["outer_folds"], c["inner_folds"], c["repeats"],
                    cfg["seed"], strata)
