"""Determinism of the synthetic data and the fold plan."""
import pandas as pd

from ml.splits import FoldPlan, plan_from_data
from ml.synthetic import generate


def test_synthetic_is_deterministic(synth_dir, tmp_path):
    generate(tmp_path, seed=323, verbose=False)
    for f in ("runs_targets.csv", "cof_timeseries.csv", "wear_checkpoints.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(synth_dir / f), pd.read_csv(tmp_path / f))


def test_fold_plan_is_deterministic(data, cfg):
    a, b = plan_from_data(data.cv_runs, cfg), plan_from_data(data.cv_runs, cfg)
    assert a.fold_map(0) == b.fold_map(0)
    g = data.cv_runs["group"].to_numpy()
    ia = [(f, tuple(tr), tuple(te)) for f, tr, te in a.outer(g, 0)]
    ib = [(f, tuple(tr), tuple(te)) for f, tr, te in b.outer(g, 0)]
    assert ia == ib
    assert [tuple(map(tuple, s)) for s in a.inner(g[:60], 0, 1)] == \
           [tuple(map(tuple, s)) for s in b.inner(g[:60], 0, 1)]


def test_repeats_differ_and_strata_balanced(data, cfg):
    strata = data.cv_runs.groupby("group")["Carbon_pct"].first().astype(str).to_dict()
    plan = FoldPlan(data.cv_runs["group"], 5, 3, repeats=2, seed=7, strata=strata)
    assert plan.fold_map(0) != plan.fold_map(1)
    per = pd.DataFrame({"g": list(strata), "c": list(strata.values())})
    per["fold"] = per["g"].map(plan.fold_map(0))
    counts = per.groupby(["c", "fold"]).size().unstack()
    assert (counts.max(axis=1) - counts.min(axis=1) <= 1).all()   # each carbon level spread evenly


def test_cv_predictions_are_reproducible(data, cfg):
    """Same seed -> identical outer-fold predictions (incl. torch models and parallel runs)."""
    from ml.evaluate import cross_validate, make_design
    from ml.models.registry import select_models
    c = {**cfg, "cv": {**cfg["cv"], "search_iter": 2},
         "models": {**cfg["models"],
                    "mlp": {**cfg["models"]["mlp"],
                            "params": {**cfg["models"]["mlp"]["params"], "n_seeds": 2,
                                       "epochs": 40}}}}
    plan = plan_from_data(data.cv_runs, c)
    d = make_design(data, "log10_k", "base", c)
    specs, _ = select_models(c, ["ridge", "gpr", "mlp"])
    a = cross_validate([d], specs, c, plan, n_jobs=1).predictions
    b = cross_validate([d], specs, c, plan, n_jobs=2).predictions
    key = ["model", "repeat", "fold", "Experiment_ID"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + ["y_pred", "y_std"]], b[key + ["y_pred", "y_std"]])
