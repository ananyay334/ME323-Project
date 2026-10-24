"""Top-up candidates satisfy every DOE constraint."""
import numpy as np
import pandas as pd

from ml.config import resolve_path
from ml.data import INPUTS, load_data
from ml.topup import candidate_grid, satisfies, select_topups


def test_constraint_function(cfg):
    ok = satisfies([100, 10, 50, 100, 100, 30], [15, 4, 12, 15, 15, 10], [60, 120, 100, 61, 62, 120], cfg)
    #               lft=90000 ok | lft 4800 | t>96     | odd t    | t>60     | ok
    assert list(ok) == [True, False, False, False, False, True]


def test_candidate_grid(cfg):
    c = candidate_grid(cfg)
    assert len(c) > 100 and satisfies(c["Load_N"], c["Freq_Hz"], c["Run_time_s"], cfg).all()
    assert (c["Run_time_s"] % 2 == 0).all()


def test_selected_topups_satisfy_constraints(cfg):
    df = select_topups(cfg, "synthetic", 4, log=lambda *a: None)
    assert list(df["Experiment_ID"]) == ["T01", "T02", "T03", "T04"]
    L, f, t = df["Load_N"].to_numpy(), df["Freq_Hz"].to_numpy(), df["Run_time_s"].to_numpy()
    assert satisfies(L, f, t, cfg).all()
    tmax = np.where(f <= 10, 120, 120 - 12 * (f - 10))
    assert (t <= tmax).all() and (L * f * t >= 7200).all() and (L * f * t <= 90000).all() and (t % 2 == 0).all()
    # distance >= 0.15 in standardised log space from existing runs and each other
    d = load_data(cfg, "synthetic").frame
    ref = d[d["Checkpoint"] == 2][list(INPUTS)].to_numpy(float)
    mu, sd = ref.mean(0), ref.std(0)
    sheet = pd.read_csv(resolve_path(cfg["data"]["run_sheet"]), dtype={"Face": str})
    ex = (np.log(sheet[["Load_N", "Freq_Hz", "Run_time_s"]].to_numpy(float)) - mu) / sd
    z = (np.log(np.c_[L, f, t]) - mu) / sd
    for i in range(len(z)):
        assert np.linalg.norm(ex - z[i], axis=1).min() >= 0.15
        for k in range(i):
            assert np.linalg.norm(z[i] - z[k]) >= 0.15
    # free, distinct positions on the plate
    used = {(r.Face, int(r.X_mm), int(r.Y_mm)) for r in sheet.dropna(subset=["Face"]).itertuples()}
    pos = list(zip(df["Face"], df["X_mm"], df["Y_mm"]))
    assert len(set(pos)) == len(pos) and not set(pos) & used
    assert set(df["Face"]) <= {"F1", "F2"} and set(df["X_mm"]) <= {8, 21, 34}
    assert set(df["Y_mm"]) <= set(range(3, 40, 3))
    for c in ("log10_k_cp1", "log10_k_cp2_sd", "COF_steady_cp2", "COF_steady_cp1_sd"):
        assert c in df and np.isfinite(df[c]).all()
