"""Leakage and split rules: gkf5 (8 runs per fold) and leave-one-alloy-out."""
import numpy as np
import pandas as pd

from ml.data import target_frame
from ml.splits import SplitPlan, assign_groups, gkf5_folds


def _full_plan(data, cfg):
    return SplitPlan(data.runs, {**cfg, "cv": {**cfg["cv"], "gkf5": {**cfg["cv"]["gkf5"],
                                                                    "seeds": [0, 1, 2, 3, 4]}}})


def test_no_group_in_train_and_test(data, cfg):
    plan = _full_plan(data, cfg)
    for target in ("log10_k", "log10_tau_trans", "log10_V_t", "COF_t"):
        df, _ = target_frame(data, target)
        g, ids = df["group"].to_numpy(), df["Experiment_ID"].to_numpy()
        n = 0
        for split, r in plan.splits():
            for _, tr, te in plan.outer(ids, split, r):
                assert not set(g[tr]) & set(g[te]), (target, split)
                assert not set(ids[tr]) & set(ids[te])
                for a, b in plan.inner(g[tr]):
                    assert not set(g[tr][a]) & set(g[tr][b])
                n += 1
        assert n == 5 * 5 + 4


def test_every_gkf5_fold_has_8_runs(data, cfg):
    plan = _full_plan(data, cfg)
    assert len(data.runs) == 40
    for fm in plan.gkf5:
        sizes = pd.Series(fm).value_counts()
        assert len(sizes) == 5 and (sizes == 8).all()
    assert len({tuple(sorted(fm.items())) for fm in plan.gkf5}) == 5      # repeats differ


def test_each_loao_fold_is_one_specimen(data, cfg):
    plan = _full_plan(data, cfg)
    ids = data.runs["Experiment_ID"].to_numpy()
    spec = data.runs.set_index("Experiment_ID")["Specimen_ID"]
    folds = list(plan.outer(ids, "loao"))
    assert len(folds) == 4
    for f, tr, te in folds:
        held = set(spec[ids[te]])
        assert held == {plan.fold_label("loao", f)} and len(te) == 10
        assert not held & set(spec[ids[tr]])


def test_replicate_pairs_share_a_group(data):
    reps = data.runs.dropna(subset=["Replicate_of"])
    assert len(reps) == 4 and (reps["group"] == reps["Replicate_of"]).all()
    assert data.runs["group"].nunique() == 36


def test_rows_inherit_their_runs_fold(data, cfg):
    plan = _full_plan(data, cfg)
    df, _ = target_frame(data, "COF_t")
    for r in range(len(plan.seeds)):
        fold = np.empty(len(df), int)
        for f, _, te in plan.outer(df["Experiment_ID"].to_numpy(), "gkf5", r):
            fold[te] = f
        assert (pd.Series(fold).groupby(df["Experiment_ID"]).nunique() == 1).all()
        runs_fold = df.assign(f=fold).groupby("Experiment_ID")["f"].first()
        assert (runs_fold == pd.Series(plan.gkf5[r])[runs_fold.index]).all()


def test_fold_size_retry_and_chain_groups():
    groups = np.array([f"g{i}" for i in range(36)] + ["g0", "g10", "g20", "g30"])
    strata = np.repeat(["S1", "S2", "S3", "S4"], 10)
    fold, offset = gkf5_folds(groups, strata, 5, seed=0)
    assert set(np.bincount(fold)) == {8} and offset >= 0
    runs = pd.DataFrame({"Experiment_ID": ["A", "B", "C"], "Replicate_of": [np.nan, "A", "B"]})
    assert assign_groups(runs).tolist() == ["A", "A", "A"]
