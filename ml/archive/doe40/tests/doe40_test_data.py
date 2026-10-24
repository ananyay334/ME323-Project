"""Synthetic schema, data rules and the four regimes."""
import numpy as np
import pandas as pd
import pytest

from ml.data import target_frame
from ml.evaluate import make_design
from ml.features import check_inputs, resolve_feature_set


def test_synthetic_schema(synth_dir):
    runs = pd.read_csv(synth_dir / "runs_targets.csv")
    wt = pd.read_csv(synth_dir / "wear_trajectory.csv")
    ts = pd.read_csv(synth_dir / "cof_timeseries.csv", usecols=["Experiment_ID", "Time_s", "COF"])
    sheet = pd.read_csv(synth_dir / "run_sheet.csv")
    need = {"Experiment_ID", "status", "Carbon_pct", "Hardness_HV", "Load_N", "Freq_Hz", "Speed_mps",
            "Hertz_p_mean_MPa", "Specific_Wear_Rate_mm3_per_Nm", "Wear_Volume_mm3", "Wear_detected",
            "Wear_Volume_LOD_mm3", "COF_ss_mean", "Vdot_steady_mm3_per_s", "dV_run_mm3", "tau_trans_s",
            "Specimen_ID", "Replicate_of", "Role", "Run_order"}
    assert need <= set(runs.columns)
    assert list(wt.columns) == ["Experiment_ID", "Time_s", "Wear_Volume_mm3", "Wear_Volume_LOD_mm3",
                                "Wear_detected", "Wear_method"]
    assert sorted(wt["Time_s"].unique()) == [0, 300, 600, 900, 1200, 1800, 2400]
    assert (ts.groupby("Experiment_ID").size() == 2400).all()
    assert len(sheet) == 40 and {"Role", "Run_order"} <= set(sheet.columns)
    r = runs[runs["Experiment_ID"].str.match(r"^R\d\d$")]
    assert np.allclose(r["Speed_mps"], 4 * 6.25e-3 * r["Freq_Hz"])          # V = 4·Δx·f
    assert 2 <= (r["Wear_detected"].astype(str) == "False").sum() <= 4      # ~3 below LOD


def test_regimes(data):
    k, info = target_frame(data, "log10_k")
    assert info["n_censored_excluded"] == info["n_censored"] > 0 and len(k) == 40 - info["n_censored"]
    assert len(target_frame(data, "COF_ss_mean")[0]) == 40
    tp, _ = target_frame(data, "log10_tau_trans")
    assert 30 <= len(tp) <= 40 and tp["Experiment_ID"].is_unique
    wc, _ = target_frame(data, "log10_V_t")
    assert len(wc) <= 240 and (wc["Time_s"] > 0).all()
    ct, _ = target_frame(data, "COF_t")
    assert len(ct) == 800 and set(ct["Time_s"]) == set(np.arange(120, 2401, 120))
    assert "Zdepth_um" not in ct and "TRIAL_2026-09-22" not in set(data.runs["Experiment_ID"])


def test_censoring_lod_half(data):
    ex, info = target_frame(data, "log10_V_t", "exclude")
    lh, _ = target_frame(data, "log10_V_t", "lod_half")
    assert len(lh) == len(ex) + info["n_censored"]
    c = lh[lh["censored"]]
    assert np.allclose(c["y"], np.log10(c["Wear_Volume_LOD_mm3"] / 2))


def test_feature_sets(data, cfg):
    assert resolve_feature_set("base", data, cfg) == ["Carbon_pct", "Load_N", "Freq_Hz", "Hardness_HV"]
    assert resolve_feature_set("physics", data, cfg)[-3:] == ["Speed_mps", "Hertz_p_mean_MPa", "PV"]
    assert list(make_design(data, "log10_V_t", "base", cfg).X.columns)[-1] == "log10_Time_s"
    assert list(make_design(data, "COF_t", "base", cfg).X.columns)[-1] == "Time_s"
    for bad in (["Time_s"], ["log10_Time_s"], ["Zdepth_um"], ["COF_ss_mean"]):
        with pytest.raises(ValueError):
            check_inputs(["Load_N"] + bad, "endpoint")
