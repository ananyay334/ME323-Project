"""Groups and splits: repeated 5-fold stratified group K-fold (gkf5) and leave-one-alloy-out.

Group = ``Replicate_of`` (followed to the root of a chain) when set, else
``Experiment_ID``: each centre replicate pair is one group. Folds are assigned per
*run* once (from the run table) and every regime maps its rows through the same
run -> fold table, so the seconds / checkpoints of a run always share its fold.

* ``gkf5`` : ``StratifiedGroupKFold(n_splits=5, shuffle=True)`` stratified by
  ``Specimen_ID``, one repeat per seed in ``cv.gkf5.seeds``. Every fold must hold
  n/5 runs (8 for N = 40); a split that violates it is redrawn with the next seed
  offset (``random_state = seed + k * seed_offset_step``).
* ``loao`` : one fold per ``Specimen_ID`` (alloy) held out.
* inner : ``GroupKFold(cv.inner_folds)`` on the training groups (nested tuning).
"""
from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


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


def gkf5_folds(groups: np.ndarray, strata: np.ndarray, n_splits: int, seed: int,
               step: int = 1000, max_tries: int = 500) -> tuple[np.ndarray, int]:
    """Fold index per run with every fold holding floor/ceil(n / n_splits) runs.

    Returns ``(fold, offset)`` where ``random_state = seed + offset * step`` produced it.
    """
    n = len(groups)
    allowed = {n // n_splits, math.ceil(n / n_splits)}
    g = pd.factorize(np.asarray(groups).astype(str))[0]
    s = pd.factorize(np.asarray(strata).astype(str))[0]
    for offset in range(max_tries):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + offset * step)
        fold = np.full(n, -1)
        for f, (_, te) in enumerate(sgkf.split(np.zeros((n, 1)), s, g)):
            fold[te] = f
        sizes = np.bincount(fold, minlength=n_splits)
        if set(sizes.tolist()) <= allowed and (fold >= 0).all():
            return fold, offset
    raise RuntimeError(f"no {n_splits}-fold group split with {sorted(allowed)} runs per fold "
                       f"found in {max_tries} tries (seed {seed})")


class SplitPlan:
    """Run -> fold maps for every gkf5 repeat and for LOAO."""

    def __init__(self, runs: pd.DataFrame, cfg: dict):
        g = cfg["cv"]["gkf5"]
        self.run_ids = runs["Experiment_ID"].astype(str).to_numpy()
        self.groups = runs["group"].astype(str).to_numpy()
        self.specimens = runs[g.get("stratify", "Specimen_ID")].astype(str).to_numpy()
        self.n_splits = int(g.get("n_splits", 5))
        self.seeds = [int(x) for x in g.get("seeds", [0])]
        self.n_inner = int(cfg["cv"].get("inner_folds", 4))
        self.gkf5, self.offsets = [], []
        for seed in self.seeds:
            fold, off = gkf5_folds(self.groups, self.specimens, self.n_splits, seed,
                                   int(g.get("seed_offset_step", 1000)), int(g.get("max_tries", 500)))
            self.gkf5.append(dict(zip(self.run_ids, fold)))
            self.offsets.append(off)
        self.loao_labels = sorted(set(self.specimens))
        self.loao = {r: self.loao_labels.index(s) for r, s in zip(self.run_ids, self.specimens)}
        self.use_loao = bool(cfg["cv"].get("loao", True)) and len(self.loao_labels) > 1

    def splits(self) -> list[tuple[str, int]]:
        """(split name, repeat) pairs to evaluate."""
        out = [("gkf5", r) for r in range(len(self.seeds))]
        return out + ([("loao", 0)] if self.use_loao else [])

    def fold_map(self, split: str, repeat: int = 0) -> dict:
        return self.gkf5[repeat] if split == "gkf5" else self.loao

    def fold_label(self, split: str, fold: int) -> str:
        return self.loao_labels[fold] if split == "loao" else str(fold)

    def outer(self, run_ids: np.ndarray, split: str, repeat: int = 0
              ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(fold, train_idx, test_idx)`` over rows labelled with ``run_ids``."""
        fm = self.fold_map(split, repeat)
        run_ids = np.asarray(run_ids).astype(str)
        missing = set(run_ids) - set(fm)
        if missing:
            raise KeyError(f"runs not in the split plan: {sorted(missing)[:5]}")
        fold = np.array([fm[r] for r in run_ids])
        for f in range(max(fm.values()) + 1):
            te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            if len(te) and len(tr):
                yield f, tr, te

    def inner(self, groups_train: np.ndarray, n_inner: int | None = None
              ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Inner GroupKFold over the rows of an outer training set."""
        g = np.asarray(groups_train).astype(str)
        k = min(int(n_inner or self.n_inner), len(set(g)))
        if k < 2:
            return []
        return [(tr, va) for tr, va in GroupKFold(n_splits=k).split(np.zeros((len(g), 1)), groups=g)]


def check_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise if any group has rows on both sides of a split."""
    groups = np.asarray(groups).astype(str)
    both = set(groups[train_idx]) & set(groups[test_idx])
    if both:
        raise AssertionError(f"group leakage across train/test: {sorted(both)[:5]}")
