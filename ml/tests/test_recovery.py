"""The pipeline recovers the effects planted in the synthetic data."""
import json

import pytest

from ml.evaluate import cross_validate, select_best
from ml.interpret import ridge_exponents
from ml.models.registry import select_models
from ml.splits import SplitPlan


def test_ridge_exponents_recover_planted_effects(data, cfg, synth_dir):
    planted = json.loads((synth_dir / "planted_effects.json").read_text())["effects"]
    ex = ridge_exponents(data, cfg, SplitPlan(data.groups, cfg), seed=0).set_index(["target", "feature"])
    for f, v in [("ln_L", 0.3), ("ln_f", 0.2), ("ln_t", -0.25)]:
        assert planted["log10_k"][f] == v
        assert ex.loc[("log10_k", f), "exponent"] == pytest.approx(v, abs=0.1)
        assert ex.loc[("log10_k", f), "ci_low"] < ex.loc[("log10_k", f), "ci_high"]


def test_best_model_r2_on_log10_k(data, cfg):
    specs, _ = select_models(cfg, ["ann", "gpr", "svr", "xgb", "ridge"])
    s = cross_validate(data, specs, cfg, SplitPlan(data.groups, cfg), n_jobs=-1).summary()
    best = select_best(s).set_index("target")
    assert best.loc["log10_k", "model"] != "ridge"
    assert best.loc["log10_k", "R2_mean"] > 0.6
