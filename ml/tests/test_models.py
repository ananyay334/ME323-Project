"""Registry contract: every available model builds, fits, predicts; missing deps are skipped."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ml.evaluate import make_design, make_joint_design, thin_rows
from ml.models import registry as reg
from ml.models.linear import DropCollinear
from ml.models.registry import build_estimator, registry, select_models
from ml.tuning import fit_estimator, sample_candidates

FAST = {"n_seeds": 2, "epochs": 30, "patience": 10}


def _params(cfg, name):
    p = dict((cfg["models"].get(name) or {}).get("params") or {})
    if name in ("mlp", "kan", "mt_mlp", "mt_kan"):
        p.update(FAST)
    if name == "rf":
        p["n_estimators"] = 30
    return p


@pytest.mark.parametrize("name", [n for n, s in registry().items() if s.regimes != ("joint",)])
def test_single_target_contract(name, cfg, data):
    spec = registry()[name]
    if spec.missing():
        pytest.skip(f"optional dependency missing: {spec.missing()}")
    target = "log10_k"
    d = make_design(data, target, "physics", cfg)
    if not spec.supports(d.regime, d.family, d.target):
        pytest.skip("model does not support this target")
    est = build_estimator(spec, _params(cfg, name), 0, {"collinear_threshold": 0.995})
    clone(est)                                                  # sklearn-compatible
    n = 70
    fit_estimator(spec, est, d.X.iloc[:n], d.y[:n], d.groups[:n])
    p = est.predict(d.X.iloc[n:])
    assert p.shape == (len(d.y) - n,) and np.isfinite(p).all()
    if spec.has_std:
        s = est.predict_std(d.X.iloc[n:])
        assert s.shape == p.shape and (s > 0).all()


@pytest.mark.parametrize("name", ["mt_mlp", "mt_kan"])
def test_joint_contract(name, cfg, data):
    spec = registry()[name]
    if spec.missing():
        pytest.skip("torch missing")
    jd = make_joint_design(data, "base", cfg)
    est = build_estimator(spec, _params(cfg, name), 0)
    runs = np.arange(60)
    mask = thin_rows(jd.rows["Time_s"].to_numpy(float), 10) & np.isin(jd.row_run, runs)
    rows = np.flatnonzero(mask)
    est.fit(jd.R.iloc[runs], jd.runs["y_wear"].to_numpy(float)[runs], jd.row_run[rows],
            jd.T.iloc[rows], jd.rows["y"].to_numpy(float)[rows], groups=jd.run_groups[runs])
    out = est.predict_both(jd.R.iloc[runs], jd.row_run[rows], jd.T.iloc[rows])
    assert out["wear"].shape == (60,) and out["cof"].shape == (len(rows),)
    assert np.isfinite(out["wear"]).all() and (out["cof_std"] > 0).all()


def test_mlp_size_limits_enforced(cfg, data):
    from ml.models.mlp import MLPEnsemble
    d = make_design(data, "log10_k", "base", cfg)
    with pytest.raises(ValueError):
        MLPEnsemble(hidden=(64,), n_seeds=1, epochs=2).fit(d.X, d.y)
    with pytest.raises(ValueError):
        MLPEnsemble(hidden=(8, 8, 8), n_seeds=1, epochs=2).fit(d.X, d.y)


def test_missing_optional_dependency_is_skipped(cfg, monkeypatch):
    monkeypatch.setattr(reg, "has_module", lambda m: m not in ("xgboost", "torch"))
    specs, skipped = select_models(cfg)
    names = {s.name for s in specs}
    assert "xgb" in skipped and "kan" in skipped and "mt_mlp" in skipped
    assert {"linear", "ridge", "rf", "svr", "gpr"} <= names


def test_drop_collinear_removes_speed(cfg, data):
    d = make_design(data, "log10_k", "physics", cfg)
    Z = (d.X - d.X.mean()) / d.X.std()
    dc = DropCollinear(0.995).fit(Z.fillna(0).to_numpy())
    kept = [d.X.columns[i] for i in dc.keep_]
    assert "Freq_Hz" in kept and "Speed_mps" not in kept      # exact multiple at fixed stroke
    assert "Load_N" in kept


def test_search_space_sampling():
    space = {"a": [1, 2], "b": [3, 4]}
    assert len(sample_candidates(space, 10, 0)) == 4            # full grid when small
    c = sample_candidates({"x": {"loguniform": [1e-3, 1e3]}, "y": [1, 2, 3]}, 7, 1)
    assert len(c) == 7 and all(1e-3 <= v["x"] <= 1e3 for v in c)
    assert sample_candidates({"x": {"loguniform": [1e-3, 1e3]}}, 5, 1) == \
        sample_candidates({"x": {"loguniform": [1e-3, 1e3]}}, 5, 1)
    assert sample_candidates({}, 5, 0) == [{}]


def test_time_thinning_keeps_running_in():
    t = np.tile(np.arange(1, 601, dtype=float), 3)
    m = thin_rows(t, 24)
    kept = np.unique(t[m])
    assert 15 <= len(kept) <= 24 and kept.min() <= 2 and kept.max() >= 590
    assert (kept < 60).sum() >= 6                                  # dense early (log spacing)


def test_torch_and_xgboost_coexist_in_one_process():
    """Regression: two OpenMP runtimes (torch, xgboost) used to segfault the parent process."""
    import os
    import subprocess
    import sys
    from ml.config import PROJECT_ROOT, has_module
    if not (has_module("torch") and has_module("xgboost")):
        pytest.skip("needs torch and xgboost")
    code = (
        "import ml, numpy as np, pandas as pd, torch\n"
        "from ml.models.registry import build_estimator, get_spec\n"
        "X = pd.DataFrame(np.random.default_rng(0).normal(size=(400, 6)), columns=list('abcdef'))\n"
        "y = X.a.to_numpy()\n"
        "xg = build_estimator(get_spec('xgb'), {'n_estimators': 30}, 0).fit(X, y)\n"
        "kan = build_estimator(get_spec('kan'), {'n_seeds': 1, 'epochs': 3}, 0).fit(X, y)\n"
        "Z = pd.DataFrame(np.random.default_rng(1).normal(size=(100000, 6)), columns=list('abcdef'))\n"
        "for _ in range(2):\n"
        "    xg.predict(Z); kan.predict(Z.iloc[:1000]); t = torch.randn(1500, 1500); (t @ t).sum()\n"
        "    xg.get_booster().set_param({'nthread': 8}); xg.get_booster().inplace_predict(Z.to_numpy())\n"
        "print('ok')\n")
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, (r.returncode, r.stderr[-2000:])
