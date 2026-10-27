"""Options taken from the TA's script: sigma-clipped levelling / profile, Savitzky-Golay profile, COF windows."""
import copy

import numpy as np
import pandas as pd
import pytest
from test_campaign import rough_track

from ml.integrated_model import cof_windows
from tribo_extract.config import load_config
from tribo_extract.wear import _band_plane, measure_track_band
from tribo_extract.wli import HeightMap

CFG = load_config()


def _cfg(**opts):
    c = copy.deepcopy(CFG)
    c["wear"]["band"] = {**c["wear"]["band"], **opts}
    return c


def test_defaults_unchanged():
    z, _ = rough_track(seed=1)
    hm = HeightMap(z=z, dx_um=1.875, dy_um=1.875, source="synthetic")
    a, _ = measure_track_band(hm, CFG)
    b, _ = measure_track_band(hm, _cfg(clip_sigma=None, profile_savgol_px=0))
    assert a["worn_area_um2"] == b["worn_area_um2"]


def test_clipped_plane_ignores_a_pit():
    ny, nx = 200, 300
    yy, xx = np.mgrid[0:ny, 0:nx]
    z = 0.01 * xx + 0.02 * yy
    z[20:60, 100:140] -= 30.0                                   # deep pit
    use = np.ones_like(z, bool)
    plain = _band_plane(z, use)
    robust = _band_plane(z, use, clip_sigma=2.0)
    truth = 0.01 * xx + 0.02 * yy
    assert np.abs(robust - truth).max() < 0.05 < np.abs(plain - truth).max()


def test_pit_next_to_track_hurts_less_with_clipping():
    z, rows = rough_track(seed=2)
    z[60:200, 400:520] -= 25.0                                  # pit in the unworn surface
    hm = HeightMap(z=z, dx_um=1.875, dy_um=1.875, source="synthetic")
    true = rows * 1.875 * 1.0
    plain, _ = measure_track_band(hm, CFG)
    clipped, _ = measure_track_band(hm, _cfg(clip_sigma=2.0))
    assert abs(clipped["worn_area_um2"] - true) < abs(plain["worn_area_um2"] - true)


def test_cof_windows(tmp_path):
    t = np.arange(0.5, 60, 0.5)
    cof = np.where(t <= 10.5, 0.10, np.where(t >= 49.5, 0.20, 0.15))
    p = tmp_path / "cyc.csv"
    pd.DataFrame({"Experiment_ID": "R99", "Step": 1, "Cycle": range(len(t)), "t_s": t, "COF_c": cof}).to_csv(p, index=False)
    w = cof_windows(p).iloc[0]
    assert w["cof_initial"] == pytest.approx(0.10) and w["cof_final10"] == pytest.approx(0.20)
