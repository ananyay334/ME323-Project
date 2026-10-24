"""Groups and folds: no group on both sides of any outer or inner split."""
import numpy as np

from ml.splits import SplitPlan, check_disjoint, shuffled_group_kfold
from ml.tests._util import synth_config


def test_groups(data):
    f = data.frame
    g = dict(zip(f["Experiment_ID"], f["group"]))
    assert g["R04"] == g["R14"] == g["R20"] == "R04"
    assert (f.groupby("Experiment_ID")["group"].nunique() == 1).all()       # both checkpoints together
    assert f["group"].nunique() == 22


def test_no_group_in_train_and_test(data, synth_dir):
    cfg = synth_config(synth_dir, quick=False)
    plan = SplitPlan(data.groups, cfg)
    assert len(plan.seeds) == 5
    for r in range(len(plan.seeds)):
        seen = []
        for f, tr, te in plan.outer(data.groups, r):
            check_disjoint(data.groups, tr, te)
            assert not set(data.groups[tr]) & set(data.groups[te])
            seen += list(te)
            for a, b in plan.inner(data.groups[tr]):
                assert not set(data.groups[tr][a]) & set(data.groups[tr][b])
        assert sorted(seen) == list(range(len(data.groups)))                # every row tested once
        assert len(set(plan.fold_maps[r].values())) == 5


def test_groups_shuffled_by_seed(data):
    a = shuffled_group_kfold(data.groups, 5, 0)
    assert a == shuffled_group_kfold(data.groups, 5, 0)
    assert a != shuffled_group_kfold(data.groups, 5, 1)
