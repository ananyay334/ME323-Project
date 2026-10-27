import numpy as np
import pandas as pd

from tribo_final.evaluate.cv import protocol_splits
from tribo_final.evaluate.metrics import compute_metrics, one_se_tie
from tribo_final.models.ann import ANNEnsemble
from tribo_final.models.gpr import GPR
from tribo_final.models.hybrid import Hybrid
from tribo_final.models.physics import LinearPhysics


def frame(n_extra=20, seed=0):
    rng = np.random.default_rng(seed)
    ids = ["R04", "R14", "R20"] + [f"X{i:02d}" for i in range(n_extra)]
    L = np.r_[32, 32, 32, rng.uniform(10, 100, n_extra)]
    f = np.r_[7.5, 7.5, 7.5, rng.uniform(4, 15, n_extra)]
    t = np.r_[84, 84, 84, rng.uniform(60, 120, n_extra)]
    groups = ["R04", "R04", "R04"] + ids[3:]
    role = ["training"] * (3 + n_extra - 5) + ["test"] * 5
    return pd.DataFrame({"run_id": ids, "group": groups, "role": role, "lnL": np.log(L), "lnf": np.log(f),
                         "lnt": np.log(t), "L": L, "log10_LS": np.log10(L * 0.02 * f * t),
                         "y": 0.12 + 0.05 * L ** (-1 / 3) + rng.normal(0, 0.004, len(L))})


def test_cv_never_splits_replicates():
    df = frame()
    for prot in ("P1", "P2"):
        for tr, te in protocol_splits(df, prot):
            test_ids, train_ids = set(df.run_id.iloc[te]), set(df.run_id.iloc[tr])
            reps = {"R04", "R14", "R20"}
            assert not (reps & test_ids and reps & train_ids), "replicates in different folds"
            assert not set(df.group.iloc[te]) & set(df.group.iloc[tr])
    p1 = protocol_splits(df, "P1")
    assert len(p1) == df.group.nunique()
    (tr, te), = protocol_splits(df, "P3")
    assert set(df.role.iloc[te]) == {"test"} and set(df.role.iloc[tr]) == {"training"}


def test_hybrid_with_zero_residual_equals_physics(cfg):
    df = frame()
    df["y"] = 0.13 + 0.04 * df.L ** (-1 / 3)  # exactly the physics law
    phys = LinearPhysics("cof").fit(df, df.y.to_numpy())
    for res in (ANNEnsemble(cfg["models"]["ann"], seed=1, n_seeds=3, fixed={"hidden": 3, "alpha": 0.01}),
                GPR(cfg["models"]["gpr"], seed=1, n_restarts=0)):
        h = Hybrid(LinearPhysics("cof"), res).fit(df, df.y.to_numpy(), df.group.to_numpy())
        assert np.allclose(h.predict(df)[0], phys.predict(df)[0], atol=1e-9)


def test_ann_ensemble_deterministic(cfg):
    df = frame()
    c = dict(cfg["models"]["ann"], n_inner_seeds=2)
    p1 = ANNEnsemble(c, seed=5, n_seeds=4).fit(df, df.y.to_numpy(), df.group.to_numpy()).predict(df)
    p2 = ANNEnsemble(c, seed=5, n_seeds=4).fit(df, df.y.to_numpy(), df.group.to_numpy()).predict(df)
    assert np.array_equal(p1[0], p2[0]) and np.array_equal(p1[1], p2[1])


def test_physics_fits_recover_parameters():
    df = frame()
    df["y"] = 0.13 + 0.04 * df.L ** (-1 / 3)
    p = LinearPhysics("cof").fit(df, df.y.to_numpy()).params
    assert abs(p["mu0"] - 0.13) < 1e-9 and abs(p["c"] - 0.04) < 1e-9
    df["y"] = np.log10(2.9e-5) + df.log10_LS
    a = LinearPhysics("archard").fit(df, df.y.to_numpy()).params
    assert abs(a["log10_k"] - np.log10(2.9e-5)) < 1e-9


def test_metrics_and_tie_rule():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9])
    m = compute_metrics(y, y, np.full(9, 0.1), rep_sd=0.5)
    assert m["R2"] == 1 and m["RMSE"] == 0 and m["coverage95_pct"] == 100 and m["tercile_acc"] == 1
    m0 = compute_metrics(y, np.full(9, y.mean()))
    assert abs(m0["R2"]) < 1e-12
    assert one_se_tie(np.ones(9), np.ones(9))[0]
    assert not one_se_tie(np.full(9, 2.0) + np.linspace(0, 0.1, 9), np.ones(9))[0]
