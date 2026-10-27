"""ml.literature_model: paper data, feature mapping and the paper-slope prior."""
import numpy as np
import pandas as pd
import pytest

from ml import literature_model as lm


def test_paper_table():
    p = lm.load_paper()
    assert len(p) == 18 and set(p["Load_N"]) == {10, 15, 20} and set(p["Speed_m_s"]) == {1, 2, 3}
    r = p.set_index("Run").loc["P01"]
    assert r["Distance_m"] == 200 and r["k"] == pytest.approx(380 / (10 * 200))
    assert p["group"].nunique() == 9                       # the two track diameters share a group


def _ours(n=12, seed=0):
    rng = np.random.default_rng(seed)
    o = pd.DataFrame({"lnL": np.log(rng.uniform(10, 100, n)), "lnv": np.log(rng.uniform(0.08, 0.3, n)),
                      "lnt": np.log(rng.uniform(60, 120, n))})
    o["y"] = -2 + 0.3 * o["lnL"] - 0.2 * o["lnv"] + 0.1 * o["lnt"] + rng.normal(0, 0.01, n)
    o["group"] = [f"g{i}" for i in range(n)]
    return o


def test_prior_limits():
    o = _ours()
    ols = lm._ols(o[lm.FEATS].to_numpy(), o["y"].to_numpy())
    assert np.allclose(lm.prior_fit(o, "y", np.array([5.0, 5.0]), 0.0), ols)        # no shrinkage = our OLS
    fixed = lm.prior_fit(o, "y", np.array([1.0, -1.0]), np.inf)                       # slopes imposed
    assert fixed[1] == 1.0 and fixed[2] == -1.0
    mid = lm.prior_fit(o, "y", np.array([1.0, -1.0]), 10.0)
    assert ols[1] < mid[1] < 1.0                                                      # pulled toward the prior


def test_prior_tuning_rejects_a_wrong_prior():
    o = _ours(n=20)
    c, lam = lm.prior_tuned(o, "y", np.array([3.0, 3.0]))
    assert lam == 0.0 and c[1] == pytest.approx(0.3, abs=0.05)
