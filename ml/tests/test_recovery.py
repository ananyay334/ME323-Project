"""The pipeline recovers the effects planted in the synthetic data."""
import numpy as np

from ml.evaluate import cross_validate, make_design, select_best
from ml.models.registry import select_models
from ml.splits import SplitPlan
from ml.trajectory import physics_route


def test_best_endpoint_model_r2_on_log10_k(data, cfg):
    plan = SplitPlan(data.runs, cfg)
    specs, _ = select_models(cfg, ["gpr", "svr", "rf", "xgb", "ridge"])
    designs = [make_design(data, "log10_k", fs, cfg) for fs in ("base", "physics")]
    s = cross_validate(designs, specs, cfg, plan, n_jobs=-1).summary()
    best = select_best(s)
    assert best.iloc[0]["model"] != "ridge"
    assert best.iloc[0]["R2_mean"] > 0.6


def test_physics_route_reconstructs_checkpoints(data, cfg):
    plan = SplitPlan(data.runs, cfg)
    specs, _ = select_models(cfg, ["gpr", "rf"])
    c = {**cfg, "features": {**cfg["features"], "use": ["base"]}}
    res, params = physics_route(data, c, plan, specs, n_jobs=-1)
    assert not res.failures and len(res.predictions)
    assert (res.predictions["model"] == "physics_route").all()
    assert np.isfinite(res.predictions["y_pred"]).all()
    assert res.summary()["R2_mean"].iloc[0] > 0.5
    assert {"Vdot_steady", "dV_run", "tau_trans"} <= set(params.columns)
