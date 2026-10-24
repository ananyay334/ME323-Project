"""Registry contract: exactly the DOE's models; every available model fits and predicts."""
import numpy as np
import pytest
from sklearn.base import clone

from ml.evaluate import fit_full, predict_target
from ml.models.ann import ANNEnsemble, n_weights
from ml.models.registry import build_estimator, registry, select_models
from ml.splits import SplitPlan
from ml.tuning import grid_candidates


def test_registry_holds_exactly_the_doe_models():
    reg = registry()
    assert set(reg) == {"ann", "gpr", "svr", "tabpfn", "xgb", "ridge"}
    assert not reg["ridge"].selectable and all(reg[m].selectable for m in reg if m != "ridge")
    assert reg["ann"].multi_output and reg["ann"].has_std and reg["gpr"].has_std


def test_grids(cfg, synth_dir):
    from ml.tests._util import synth_config
    m = synth_config(synth_dir, quick=False)["models"]
    assert len(grid_candidates(m["ann"]["grid"])) == 6
    assert len(grid_candidates(m["svr"]["grid"])) == 36
    assert len(grid_candidates(m["xgb"]["grid"])) == 16
    assert max(m["xgb"]["grid"]["max_depth"]) <= 2 and m["xgb"]["params"]["subsample"] == 0.8
    assert m["ann"]["params"]["n_members"] == 20 and cfg["models"]["ann"]["params"]["n_members"] == 3


@pytest.mark.parametrize("hidden", [3, 4])
def test_ann_has_at_most_30_weights(data, hidden):
    assert n_weights(3, hidden, 2) <= 30
    est = ANNEnsemble(hidden=hidden, n_members=2, max_epochs=50).fit(data.X, data.Y, data.groups)
    assert sum(w[0].size for w in est.weights_) == n_weights(3, hidden, 2) <= 30
    assert est.predict(data.X).shape == (len(data.X), 2)


@pytest.mark.parametrize("name", ["ann", "gpr", "svr", "tabpfn", "xgb", "ridge"])
def test_contract(name, cfg, data):
    spec = registry()[name]
    if spec.missing():
        pytest.skip(f"optional dependency missing: {spec.missing()}")
    params = {**(cfg["models"][name].get("params") or {}),
              **grid_candidates(cfg["models"][name].get("grid"), True)[0]}
    est = build_estimator(spec, params, 0)
    clone(est)
    n = 36
    for t in data.targets:
        j = data.targets.index(t)
        y = data.Y if spec.multi_output else data.Y[:, j]
        ok = np.isfinite(data.Y[:n, j]) if not spec.multi_output else np.ones(n, bool)
        est.fit(data.X.iloc[:n][ok], y[:n][ok])
        mu, sd = predict_target(est, spec, data.X.iloc[n:], data.targets, t)
        assert mu.shape == (len(data.X) - n,) and np.isfinite(mu).all()
        if spec.has_std:
            assert (sd > 0).all()


def test_missing_tabpfn_is_skipped_not_fatal(cfg, monkeypatch):
    from ml.models import registry as reg
    monkeypatch.setattr(reg, "has_module", lambda m: m != "tabpfn")
    specs, skipped = select_models(cfg)
    assert "tabpfn" in skipped and "tabpfn" not in {s.name for s in specs}
    assert {"ann", "gpr", "svr", "xgb", "ridge"} <= {s.name for s in specs}


def test_fit_full_ann_uses_both_targets(cfg, data):
    spec = registry()["ann"]
    est, params = fit_full(spec, data, cfg, SplitPlan(data.groups, cfg))
    assert est.n_outputs_ == 2 and set(params) == {"hidden", "weight_decay"}


def test_fast_gpr_likelihood_matches_sklearn(data):
    """FastLML reproduces sklearn's LML and gradient; the fit reaches sklearn's optimum."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from ml.models import gpr as G
    X, y = data.X, data.Y[:, 1]
    Z = ((X - X.mean()) / X.std()).to_numpy()
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
    fast = G.GPRModel(n_restarts_optimizer=2, random_state=1).fit(X, y)
    slow = G.GPRModel(n_restarts_optimizer=2, random_state=1, kernel=G.make_kernel(Z.shape[1])).fit(X, y)
    assert fast.gp_.log_marginal_likelihood_value_ >= slow.gp_.log_marginal_likelihood_value_ - 1e-3


def test_numpy_ann_gradients_match_finite_differences():
    """The numpy backend (used when torch is absent) has hand-written gradients: check them."""
    from ml.models.ann import _loss_and_grads
    rng = np.random.default_rng(0)
    M, n, d, H, o = 3, 11, 3, 4, 2
    Z, t = rng.normal(size=(n, d)), rng.normal(size=(n, o))
    mtr = (rng.random((M, n, o)) > 0.3).astype(float)
    ntr = mtr.sum((1, 2))
    params = [rng.normal(size=s) for s in ((M, d, H), (M, 1, H), (M, H, o), (M, 1, o))]
    _, _, grads = _loss_and_grads(params, Z, t, mtr, ntr)
    h = 1e-6
    for p, g in zip(params, grads):
        for idx in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(6)]:
            old = p[idx]
            p[idx] = old + h
            lp = _loss_and_grads(params, Z, t, mtr, ntr)[0].sum()
            p[idx] = old - h
            lm = _loss_and_grads(params, Z, t, mtr, ntr)[0].sum()
            p[idx] = old
            assert g[idx] == pytest.approx((lp - lm) / (2 * h), rel=1e-5, abs=1e-8)


def test_numpy_ann_fits_a_smooth_function():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(40, 3))
    Y = np.c_[np.tanh(X @ [0.8, -0.5, 0.3]), 0.2 * X[:, 0] ** 2]
    est = ANNEnsemble(hidden=4, n_members=3, weight_decay=1e-4, max_epochs=3000, backend="numpy").fit(X, Y)
    r2 = 1 - ((est.predict(X) - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    assert (r2 > 0.9).all()
