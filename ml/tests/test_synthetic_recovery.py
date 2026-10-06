"""The pipeline recovers the effects planted in the synthetic data (ml/synthetic.py)."""
import numpy as np
import pandas as pd
import pytest

from ml.evaluate import cross_validate, make_design
from ml.interpret import archard_check, curvature, partial_dependence
from ml.models.gpr import GPRModel
from ml.models.registry import select_models
from ml.splits import plan_from_data
from ml.synthetic import NOISE_LOG10_K


@pytest.fixture(scope="module")
def gpr_wear(data, cfg):
    d = make_design(data, "log10_k", "base", cfg)
    return GPRModel(n_restarts_optimizer=2, random_state=0).fit(d.X, d.y), list(d.X.columns)


@pytest.fixture(scope="module")
def gpr_cof(data, cfg):
    d = make_design(data, "COF_ss_mean", "base", cfg)
    return GPRModel(n_restarts_optimizer=2, random_state=0).fit(d.X, d.y), list(d.X.columns)


def test_wear_rate_is_predictable_to_the_noise_floor(data, cfg):
    plan = plan_from_data(data.cv_runs, cfg)
    specs, _ = select_models(cfg, ["gpr", "ridge"])
    d = make_design(data, "log10_k", "base", cfg)
    s = cross_validate([d], specs, cfg, plan, n_jobs=1).summary().set_index("model")
    assert s.loc["gpr", "R2_mean"] > 0.7
    assert s.loc["gpr", "RMSE_mean"] < 1.5 * NOISE_LOG10_K
    assert s.loc["gpr", "RMSE_mean"] < s.loc["ridge", "RMSE_mean"]     # nonlinearity is real
    assert 0.8 < s.loc["gpr", "coverage95_mean"] <= 1.0


def test_load_increases_wear_convexly(data, gpr_wear):
    est, cols = gpr_wear
    grid = np.linspace(2, 10, 9)
    pd_load = partial_dependence(est, data, cols, "Load_N", grid)
    c = curvature(grid, pd_load)
    assert c["rise"] > 0.15                         # planted: +0.44 over the range (+ curvature)
    assert c["sagitta_rel"] < 0                     # planted +0.10 xL^2: convex
    hi_c = partial_dependence(est, data, cols, "Load_N", grid, carbon=2.0)
    lo_c = partial_dependence(est, data, cols, "Load_N", grid, carbon=0.0)
    assert (hi_c[-1] - hi_c[0]) < (lo_c[-1] - lo_c[0])   # planted carbon x load interaction


def test_carbon_reduces_wear(data, gpr_wear):
    est, cols = gpr_wear
    pd_c = partial_dependence(est, data, cols, "Carbon_pct", np.array([0.0, 0.5, 1.0, 2.0]))
    assert pd_c[-1] - pd_c[0] < -0.2               # planted ~ -0.33 (incl. hardness)
    assert np.all(np.diff(pd_c) < 0.02)


def test_cof_steady_state_effects(data, gpr_cof):
    est, cols = gpr_cof
    f = partial_dependence(est, data, cols, "Freq_Hz", np.linspace(2, 10, 5))
    c = partial_dependence(est, data, cols, "Carbon_pct", np.array([0.0, 2.0]))
    assert f[-1] - f[0] > 0.03                      # planted +0.07
    assert c[-1] - c[0] < -0.03                     # planted -0.052
    assert est.length_scales_["Hardness_HV"] > 10   # COF does not depend on hardness


def test_running_in_is_recovered(data, cfg):
    """COF(t) model predicts the planted exponential running-in (high early, then steady)."""
    from ml.models.registry import build_estimator, get_spec
    from ml.evaluate import thin_rows
    d = make_design(data, "COF_t", "base", cfg)
    keep = thin_rows(d.time, 30)
    est = build_estimator(get_spec("xgb"), {"n_estimators": 200, "max_depth": 3}, 0)
    est.fit(d.X[keep], d.y[keep])
    early = est.predict(d.X[d.time == 5]).mean()
    late = est.predict(d.X[d.time == 400]).mean()
    assert early - late > 0.05                      # planted amplitude ~0.15 decaying over ~45 s


def test_archard_deviation_is_detected(data):
    a = archard_check(data).set_index("term")
    load = a.loc["load exponent (log10 F)"]
    assert load["estimate"] > 1.1 and not load["consistent_with_archard"]   # k rises with load


def test_trajectory_curves_are_recovered(data, cfg, tmp_path):
    from ml.trajectory import run_trajectory
    c = {**cfg, "trajectory": {**cfg["trajectory"], "models": ["gpr"], "feature_sets": ["base"]}}
    plan = plan_from_data(data.cv_runs, c)
    (tmp_path / "figures").mkdir()
    res = run_trajectory(data, c, plan, tmp_path / "trajectory", n_jobs=1)
    cm = res["curve_metrics"].set_index("model")
    assert cm.loc["gpr", "R2_V"] > 0.8
    vd = res["summary"].set_index(["target", "model"]).loc[("log10_Vdot_steady", "gpr")]
    assert vd["R2_mean"] > 0.5


def test_trajectory_does_nothing_without_columns(data, cfg, tmp_path):
    import dataclasses
    from ml.trajectory import run_trajectory
    d2 = dataclasses.replace(data, runs=data.runs.drop(
        columns=["Vdot_steady_mm3_per_s", "dV_run_mm3", "tau_trans_s"]))
    assert run_trajectory(d2, cfg, plan_from_data(d2.cv_runs, cfg), tmp_path / "t", 1) is None
    assert not (tmp_path / "t").exists()


def test_active_learning_beats_random(data, cfg, tmp_path):
    from ml.active_learning import run_active_learning
    c = {**cfg, "active_learning": {**cfg["active_learning"], "targets": ["COF_ss_mean"],
                                    "repeats": 4, "budget": 36}}
    res = run_active_learning(data, c, tmp_path, n_jobs=1)
    s = res["targets"]["COF_ss_mean"]["summary"].set_index("strategy")
    assert s.loc["variance", "rmse_at_budget"] <= 1.05 * s.loc["random", "rmse_at_budget"]
    curves = pd.read_csv(tmp_path / "al_curves_COF_ss_mean.csv")
    v = curves[curves["strategy"] == "variance"].groupby("N")["rmse_remaining"].mean()
    assert v.iloc[-1] < v.iloc[0]                   # the model improves as conditions are added
