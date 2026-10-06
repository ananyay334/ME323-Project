"""Schema of the (synthetic) input tables and the data-cleaning rules."""
import numpy as np
import pandas as pd
import pytest

from ml.data import load_data, target_frame
from ml.features import check_inputs, resolve_feature_set

from ml.tests._util import synth_config

RUN_COLUMNS = [
    "Experiment_ID", "status", "issues", "cof_flags", "wear_flags", "Replicate_of",
    "Carbon_pct", "Hardness_HV", "Load_N", "Freq_Hz", "Stroke_mm", "Speed_mps", "sliding_time_s",
    "Sliding_distance_m", "Hertz_p_mean_MPa", "Hertz_width_um", "Ambient_T_C", "RH_pct",
    "Initial_Ra_um", "Initial_Sa_um",
    "COF_ss_mean", "Specific_Wear_Rate_mm3_per_Nm", "Wear_Volume_mm3", "Wear_detected",
    "Wear_Volume_LOD_mm3", "Specific_Wear_Rate_LOD_mm3_per_Nm",
    "Vdot_steady_mm3_per_s", "dV_run_mm3", "tau_trans_s", "Specific_Wear_Rate_slope_mm3_per_Nm",
    "t_runin_s", "COF_runin_peak", "COF_at_60s", "COF_at_150s", "COF_at_300s", "COF_at_450s",
    "COF_at_600s"]
TS_COLUMNS = ["Experiment_ID", "Time_s", "COF", "COF_std", "Fz_N", "Zdepth_um", "Phase",
              "Carbon_pct", "Load_N", "Freq_Hz"]


def test_synthetic_schema(synth_dir):
    runs = pd.read_csv(synth_dir / "runs_targets.csv")
    ts = pd.read_csv(synth_dir / "cof_timeseries.csv")
    assert set(RUN_COLUMNS) <= set(runs.columns), set(RUN_COLUMNS) - set(runs.columns)
    assert set(TS_COLUMNS) <= set(ts.columns), set(TS_COLUMNS) - set(ts.columns)
    grid = runs[runs["Experiment_ID"].str.match(r"^R0\d\d$|^R100$")]
    assert len(grid) == 100
    assert grid.groupby(["Carbon_pct", "Load_N", "Freq_Hz"]).ngroups == 100
    reps = runs[runs["Replicate_of"].notna()]
    assert len(reps) >= 10 and reps["Replicate_of"].isin(grid["Experiment_ID"]).all()
    assert (ts.groupby("Experiment_ID").size().drop("TRIAL_2026-09-22") == 600).all()
    assert set(ts["Phase"].unique()) <= {"running_in", "steady"}


def test_cleaning_rules(data):
    ids = set(data.runs["Experiment_ID"])
    assert "R111" not in ids                                   # status == error
    assert not any(i.startswith("TRIAL") for i in ids)
    assert "Zdepth_um" not in data.ts.columns                  # diagnostic only
    assert set(data.holdout_ids) == {f"U0{i}" for i in range(1, 7)}
    assert not data.cv_runs["Experiment_ID"].isin(data.holdout_ids).any()


def test_partial_runs_used_only_for_available_targets(data):
    partial = set(data.cv_runs.loc[data.cv_runs["status"] == "partial", "Experiment_ID"])
    assert partial
    wear, _ = target_frame(data, "log10_k", "exclude")
    cof, _ = target_frame(data, "COF_ss_mean")
    assert partial <= set(cof["Experiment_ID"])               # COF still used
    assert not partial & set(wear["Experiment_ID"])            # no wear value -> not used


def test_censoring_modes(data):
    ex, info_ex = target_frame(data, "log10_k", "exclude")
    lh, info_lh = target_frame(data, "log10_k", "lod_half")
    assert info_ex["n_censored"] == 8 and not ex["censored"].any()
    assert len(lh) == len(ex) + info_ex["n_censored"]
    c = lh[lh["censored"]]
    expect = np.log10(c["Specific_Wear_Rate_LOD_mm3_per_Nm"] / 2)
    assert np.allclose(c["y"], expect)
    assert (c["y"] < c["y_lod"]).all()


def test_all_nan_and_absent_columns_dropped(synth_dir, tmp_path):
    runs = pd.read_csv(synth_dir / "runs_targets.csv")
    runs["RH_pct"] = np.nan
    runs = runs.drop(columns=["Ambient_T_C", "Replicate_of"])
    runs.to_csv(tmp_path / "runs.csv", index=False)
    cfg = synth_config(synth_dir, data={"synthetic": {"runs": str(tmp_path / "runs.csv")}})
    d = load_data(cfg, "synthetic")
    assert d.dropped_inputs["RH_pct"] == "all NaN"
    assert d.dropped_inputs["Ambient_T_C"] == "absent"
    assert any("RH_pct (all NaN)" in n for n in d.notes)
    assert "RH_pct" not in resolve_feature_set("physics", d, cfg)
    assert (d.runs["group"] == d.runs["Experiment_ID"]).all()   # no Replicate_of column


def test_missing_design_column_raises(synth_dir, tmp_path):
    runs = pd.read_csv(synth_dir / "runs_targets.csv").drop(columns=["Freq_Hz"])
    runs.to_csv(tmp_path / "runs.csv", index=False)
    cfg = synth_config(synth_dir, data={"synthetic": {"runs": str(tmp_path / "runs.csv")}})
    with pytest.raises(ValueError, match="Freq_Hz"):
        load_data(cfg, "synthetic")


def test_forbidden_inputs():
    check_inputs(["Carbon_pct", "Load_N", "Time_s"], "time")
    for bad in (["Zdepth_um"], ["Time_s"], ["Sliding_distance_m"], ["COF_ss_mean"]):
        with pytest.raises(ValueError):
            check_inputs(["Load_N"] + bad, "run")
    with pytest.raises(ValueError):
        check_inputs(["Load_N", "Zdepth_um"], "time")


def test_holdout_pulls_replicate_group(synth_dir, tmp_path):
    runs = pd.read_csv(synth_dir / "runs_targets.csv", dtype={"Replicate_of": str})
    rep = runs.dropna(subset=["Replicate_of"]).iloc[0]
    cfg = synth_config(synth_dir, data={"synthetic": {"holdout_ids": [rep["Replicate_of"]]}})
    d = load_data(cfg, "synthetic")
    assert {rep["Replicate_of"], rep["Experiment_ID"]} <= set(d.holdout_ids)
