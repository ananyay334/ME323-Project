"""Loading: train/test roles, log inputs, masked targets, censoring."""
import numpy as np
import pytest

from ml.data import load_data
from ml.tests._util import synth_config

TEST_RUNS = {f"R{i}" for i in range(24, 30)}


def test_train_rows(data):
    f = data.frame
    assert not set(f["Experiment_ID"]) & TEST_RUNS
    assert set(f["Experiment_ID"]) == {f"R{i:02d}" for i in range(24)}
    assert len(f) == 48
    np.testing.assert_allclose(f["ln_L"], np.log(f["Load_N"]))
    np.testing.assert_allclose(f["ln_t"], np.log(f["Time_s"]))
    assert list(data.X.columns) == ["ln_L", "ln_f", "ln_t"]


def test_censoring_masks_only_k(data, synth_dir):
    f = data.frame
    cens = f["censored_log10_k"]
    assert cens.sum() == 4 and f.loc[cens, "log10_k"].isna().all()
    assert f.loc[cens, "COF_steady"].notna().all()                         # kept for the other target
    lod = load_data(synth_config(synth_dir, censoring={"mode": "lod_half"}), "synthetic")
    g = lod.frame
    exp = g.loc[g["censored_log10_k"], "log10_k_lod"] - np.log10(2)
    np.testing.assert_allclose(g.loc[g["censored_log10_k"], "log10_k"], exp)


def test_test_split(cfg):
    t = load_data(cfg, "synthetic", split="test")
    assert set(t.frame["Experiment_ID"]) == TEST_RUNS and len(t.frame) == 12
    with pytest.raises(ValueError):
        load_data(cfg, "synthetic", split="all")
