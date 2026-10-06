"""Leakage guards: group-disjoint splits everywhere, holdout isolation, in-fold preprocessing."""
import numpy as np
import pandas as pd

from ml.data import target_frame
from ml.splits import FoldPlan, assign_groups, check_disjoint, plan_from_data


def _all_splits(plan: FoldPlan, groups: np.ndarray):
    for r in range(plan.repeats):
        for f, tr, te in plan.outer(groups, r):
            yield tr, te
            for itr, iva in plan.inner(groups[tr], r, f):
                yield tr[itr], tr[iva]


def test_no_group_in_train_and_test(data, cfg):
    cfg = {**cfg, "cv": {**cfg["cv"], "repeats": 2}}
    plan = plan_from_data(data.cv_runs, cfg)
    for target in ("log10_k", "COF_ss_mean", "COF_t"):
        df, _ = target_frame(data, target)
        g = df["group"].to_numpy()
        n = 0
        for tr, te in _all_splits(plan, g):
            assert not set(g[tr]) & set(g[te]), target
            assert not set(df["Experiment_ID"].to_numpy()[tr]) & set(df["Experiment_ID"].to_numpy()[te])
            check_disjoint(g, tr, te)
            n += 1
        assert n > 10


def test_every_cv_row_tested_once_per_repeat(data, cfg):
    plan = plan_from_data(data.cv_runs, cfg)
    df, _ = target_frame(data, "COF_t")
    seen = np.zeros(len(df), int)
    for _, _, te in plan.outer(df["group"].to_numpy(), 0):
        seen[te] += 1
    assert (seen == 1).all()


def test_replicates_share_group(data):
    reps = data.runs.dropna(subset=["Replicate_of"])
    assert len(reps) >= 10
    assert (reps["group"] == reps["Replicate_of"]).all()
    orig = data.runs.set_index("Experiment_ID").loc[reps["Replicate_of"], "group"]
    assert (orig.to_numpy() == reps["group"].to_numpy()).all()


def test_replicate_chain_resolves_to_root():
    runs = pd.DataFrame({"Experiment_ID": ["A", "B", "C", "D"],
                         "Replicate_of": [np.nan, "A", "B", np.nan]})
    assert assign_groups(runs).tolist() == ["A", "A", "A", "D"]


def test_holdout_never_in_cv(data, cfg):
    plan = plan_from_data(data.cv_runs, cfg)
    hold_groups = set(data.holdout_runs["group"])
    assert hold_groups and not hold_groups & set(plan.groups)
    df, _ = target_frame(data, "COF_t", split="holdout")
    assert set(df["Experiment_ID"]) == set(data.holdout_ids)


def test_time_rows_follow_run_groups(data, cfg):
    """Seconds of one run never straddle a split, in every repeat."""
    plan = FoldPlan(data.cv_runs["group"], 5, 3, repeats=3, seed=1)
    df, _ = target_frame(data, "COF_t")
    for r in range(3):
        fold_of_row = np.empty(len(df), int)
        for f, _, te in plan.outer(df["group"].to_numpy(), r):
            fold_of_row[te] = f
        assert (pd.Series(fold_of_row).groupby(df["Experiment_ID"]).nunique() == 1).all()


def test_canary_run_level_noise_is_not_learnable(data, cfg):
    """A random value per run, repeated on every second, must be unpredictable under group CV.

    If seconds of a run leaked across folds, a forest would memorise the run from its
    inputs and score R² close to 1 (shown with a naive row-level split as a control).
    """
    from ml.evaluate import cross_validate, make_design
    from ml.models.registry import select_models
    from ml.splits import FoldPlan
    d = make_design(data, "COF_t", "base", cfg)
    rng = np.random.default_rng(0)
    noise = {e: rng.normal() for e in d.frame["Experiment_ID"].unique()}
    d.frame["y"] = d.frame["Experiment_ID"].map(noise).to_numpy(float)
    specs, _ = select_models(cfg, ["rf"])
    c = {**cfg, "cv": {**cfg["cv"], "search_iter": 1},
         "models": {**cfg["models"], "rf": {"params": {"n_estimators": 30, "max_depth": 8}}}}
    plan = plan_from_data(data.cv_runs, c)
    r2 = cross_validate([d], specs, c, plan, n_jobs=1).fold_metrics
    assert r2[r2["target"] == "COF_t"]["R2"].mean() < 0.1
    # control: the same model with seconds split at random across folds does leak
    rows = pd.Series(np.arange(len(d.frame)).astype(str))
    leaky = FoldPlan(rows, 5, 2, 1, seed=0)
    d.frame["group"] = rows.to_numpy()
    r2_leaky = cross_validate([d], specs, c, leaky, n_jobs=1).fold_metrics
    assert r2_leaky[r2_leaky["target"] == "COF_t"]["R2"].mean() > 0.8


def test_preprocessing_fitted_on_training_fold_only(data, cfg):
    from ml.evaluate import make_design
    from ml.models.gpr import GPRModel
    from ml.models.mlp import MLPEnsemble
    d = make_design(data, "log10_k", "base", cfg)
    tr = np.arange(60)
    Xtr = d.X.iloc[tr]
    gp = GPRModel(n_restarts_optimizer=0).fit(Xtr, d.y[tr])
    med = Xtr.median()
    assert np.allclose(gp.scaler_.mean_, Xtr.fillna(med).mean().to_numpy())
    assert not np.allclose(gp.scaler_.mean_, d.X.fillna(d.X.median()).mean().to_numpy())
    mlp = MLPEnsemble(n_seeds=1, epochs=5).fit(Xtr, d.y[tr])
    assert np.isclose(mlp.y_mu_, d.y[tr].mean())
    assert np.allclose(mlp.pre_.scaler.mean_, Xtr.fillna(med).mean().to_numpy())
