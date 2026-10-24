"""Registry contract: the DOE's models only; every available model fits on every regime."""
import numpy as np
import pytest
from sklearn.base import clone

from ml.evaluate import make_design
from ml.models.registry import build_estimator, registry, select_models
from ml.tuning import grid_candidates


def test_registry_holds_exactly_the_doe_models():
    reg = registry()
    assert set(reg) == {"gpr", "tabpfn", "svr", "rf", "xgb", "ridge"}
    assert not reg["ridge"].selectable and all(reg[m].selectable for m in reg if m != "ridge")


def test_grids(cfg):
    m = cfg["models"]
    assert len(grid_candidates(m["svr"]["grid"])) == 48
    assert len(grid_candidates(m["rf"]["grid"])) == 8
    assert len(grid_candidates(m["xgb"]["grid"])) == 16
    assert grid_candidates(m["xgb"]["grid"], first_only=True) == [
        {"learning_rate": 0.02, "max_depth": 2, "min_child_weight": 3, "n_estimators": 200}]
    assert all(d <= 3 for d in m["rf"]["grid"]["max_depth"] + m["xgb"]["grid"]["max_depth"])
    assert max(m["xgb"]["grid"]["learning_rate"]) <= 0.05
    assert min(m["rf"]["grid"]["min_samples_leaf"]) >= 3


@pytest.mark.parametrize("target", ["log10_k", "log10_dV_run", "log10_V_t"])
@pytest.mark.parametrize("name", ["gpr", "tabpfn", "svr", "rf", "xgb", "ridge"])
def test_contract(name, target, cfg, data):
    spec = registry()[name]
    if spec.missing():
        pytest.skip(f"optional dependency missing: {spec.missing()}")
    d = make_design(data, target, "physics", cfg)
    params = {**(cfg["models"][name].get("params") or {}),
              **(grid_candidates(cfg["models"][name].get("grid"), True)[0])}
    est = build_estimator(spec, params, 0)
    clone(est)
    n = int(0.75 * len(d.y))
    est.fit(d.X.iloc[:n], d.y[:n])
    p = est.predict(d.X.iloc[n:])
    assert p.shape == (len(d.y) - n,) and np.isfinite(p).all()
    if spec.has_std:
        assert (est.predict_std(d.X.iloc[n:]) > 0).all()


def test_missing_tabpfn_is_skipped_not_fatal(cfg, monkeypatch):
    from ml.models import registry as reg
    monkeypatch.setattr(reg, "has_module", lambda m: m != "tabpfn")
    specs, skipped = select_models(cfg)
    assert "tabpfn" in skipped and "tabpfn" not in {s.name for s in specs}
    assert {"gpr", "svr", "rf", "xgb", "ridge"} <= {s.name for s in specs}


def test_fast_gpr_likelihood_matches_sklearn(cfg, data):
    """FastLML reproduces sklearn's LML and gradient; the fit reaches sklearn's optimum."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from ml.models import gpr as G
    d = make_design(data, "log10_k", "physics", cfg)
    Z = ((d.X - d.X.mean()) / d.X.std()).fillna(0).to_numpy()
    y = d.y
    k = G.make_kernel(Z.shape[1])
    ref = GaussianProcessRegressor(kernel=k, normalize_y=True, optimizer=None).fit(Z, y)
    f = G.FastLML(Z, (y - y.mean()) / y.std())
    rng = np.random.default_rng(0)
    for _ in range(10):
        th = rng.uniform(k.bounds[:, 0], k.bounds[:, 1])
        a, ga = ref.log_marginal_likelihood(th, eval_gradient=True)
        b, gb = f(th)
        assert abs(a - b) <= 1e-6 * max(1.0, abs(a))
        assert np.linalg.norm(ga - gb) <= 1e-3 * np.linalg.norm(ga)
    fast = G.GPRModel(n_restarts_optimizer=2, random_state=1).fit(d.X, y)
    slow = G.GPRModel(n_restarts_optimizer=2, random_state=1, kernel=G.make_kernel(Z.shape[1])).fit(d.X, y)
    assert fast.gp_.log_marginal_likelihood_value_ >= slow.gp_.log_marginal_likelihood_value_ - 1e-3
