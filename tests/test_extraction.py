"""Regression tests for the target extraction.

Run from the project folder:   python -m pytest -q
They use the TA's practice run in data/raw/TRIAL_2026-09-22 as a fixture, plus
synthetic wear scars of known size painted into that screenshot (tests/synth.py).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth import add_groove
from tribo_extract import physics
from tribo_extract.cof import extract_cof
from tribo_extract.config import PROJECT_ROOT, load_config
from tribo_extract.pipeline import RUN_SHEET_COLUMNS, process_all
from tribo_extract.tribometer_csv import read_tribometer_csv
from tribo_extract.wear import measure_wear_scar, subtract_baseline
from tribo_extract.wli import read_wli_screenshot

TRIAL = PROJECT_ROOT / "data" / "raw" / "TRIAL_2026-09-22"
CSV = TRIAL / "Trail_Tribo.csv"
PNG = TRIAL / "Run-2026-09-23-001048_WLI_10x.png"
CBAR = (45.74, 52.04)
FOV = [899.98, 562.49]
CFG = load_config()


def _has_ocr():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------- CSV / COF
def test_csv_parsing():
    run = read_tribometer_csv(CSV, CFG["csv"]["columns"])
    assert run.meta["Type"] == "Reciprocating"
    assert abs(run.sample_rate_hz - 100) < 1e-6
    assert set(run.data["recipe_step"].unique()) == {1, 2}
    assert np.all(np.diff(run.data["t_s"]) > 0)          # continuous clock


def test_cof_trial():
    run = read_tribometer_csv(CSV, CFG["csv"]["columns"])
    res, ts, _ = extract_cof(run, CFG, load_N=2.0)
    assert res["sliding_step"] == 2
    assert abs(res["COF_ss_mean"] - 0.107) < 0.003
    assert 1.0 < res["t_runin_s"] < 2.0
    assert list(ts["Time_s"]) == [1.0, 2.0, 3.0]


# ----------------------------------------------------------------- screenshot
@pytest.mark.skipif(not _has_ocr(), reason="tesseract not installed")
def test_colourbar_ocr():
    hm = read_wli_screenshot(PNG, CFG)
    assert (hm.zmin, hm.zmax) == CBAR
    assert hm.objective == "10x"


def test_screenshot_decode_and_no_false_wear():
    hm = read_wli_screenshot(PNG, CFG, *CBAR)
    assert hm.masked_frac < 0.05                           # only the overlay lines are masked
    assert CBAR[0] < np.nanmedian(hm.z) < CBAR[1]
    for ang in (None, 0, 45, 90, 135):
        res, _ = measure_wear_scar(hm, CFG, angle_deg=ang)
        assert not res["wear_detected"], ang


@pytest.mark.parametrize("ang,W,D,off", [(10, 220, 1.0, 40), (60, 250, 1.5, 30), (120, 200, 0.8, 0),
                                         (90, 80, 0.4, 0), (0, 400, 2.5, 0)])
def test_single_scan_groove_area(tmp_path, ang, W, D, off):
    out = tmp_path / "worn.png"
    true_area = add_groove(PNG, out, *CBAR, FOV, W, D, ang, off)
    hm = read_wli_screenshot(out, CFG, *CBAR)
    res, _ = measure_wear_scar(hm, CFG, angle_deg=ang)
    assert res["wear_detected"]
    assert abs(res["worn_area_um2"] / true_area - 1) < 0.10


@pytest.mark.parametrize("ang,W,D", [(2, 120, 0.3), (45, 100, 0.25), (90, 300, 2.0)])
def test_baseline_difference(tmp_path, ang, W, D):
    out = tmp_path / "worn.png"
    true_area = add_groove(PNG, out, *CBAR, FOV, W, D, ang, 0)
    base = read_wli_screenshot(PNG, CFG, *CBAR)
    worn = read_wli_screenshot(out, CFG, *CBAR)
    worn.z = np.roll(np.roll(worn.z, 5, 0), -9, 1)          # simulate re-positioning error
    diff, info = subtract_baseline(worn, base)
    assert info["baseline_shift_px"] == [5, -9]
    res, _ = measure_wear_scar(diff, CFG)
    assert abs(res["worn_area_um2"] / true_area - 1) < 0.03


# ----------------------------------------------------------------- physics
def test_physics():
    assert physics.sliding_distance_m(4, 6.25, 600, "full") == pytest.approx(30.0)
    assert physics.sliding_distance_m(4, 6.25, 600, "amplitude") == pytest.approx(60.0)
    assert physics.mean_speed_mps(10, 6.25, "full") == pytest.approx(0.125)
    assert physics.specific_wear_rate(0.01, 10, 50) == pytest.approx(2e-5)
    h = physics.hertz_contact(10, 6, 210, 0.3, 190, 0.3)
    assert 100 < h["hertz_width_um"] < 140


# ----------------------------------------------------------------- end to end
def test_pipeline_end_to_end(tmp_path):
    raw = tmp_path / "raw" / "R001"
    raw.mkdir(parents=True)
    shutil.copy(CSV, raw / "run.csv")
    shutil.copy(PNG, raw / "scan_t0s.png")
    areas = {}
    for t, D in [(300, 0.6), (600, 1.0)]:
        areas[t] = add_groove(PNG, raw / f"scan_t{t}s.png", *CBAR, FOV, 220, D, 10, 0)
    pd.DataFrame({"file": ["scan_t0s.png", "scan_t300s.png", "scan_t600s.png"],
                  "cbar_min_um": [CBAR[0]] * 3, "cbar_max_um": [CBAR[1]] * 3}) \
        .to_csv(raw / "wli_scans.csv", index=False)
    sheet = pd.DataFrame([{"Experiment_ID": "R001", "Carbon_pct": 0.5, "Load_N": 2, "Freq_Hz": 4,
                           "Stroke_mm": 6.25}], columns=RUN_SHEET_COLUMNS)
    sheet.to_csv(tmp_path / "run_sheet.csv", index=False)
    cfg = load_config(overrides={"paths": {"raw_dir": str(tmp_path / "raw"),
                                           "run_sheet": str(tmp_path / "run_sheet.csv"),
                                           "processed_dir": str(tmp_path / "processed"),
                                           "reports_dir": str(tmp_path / "reports")}})
    runs = process_all(cfg, make_plots=False, log=lambda *a: None)
    r = runs.iloc[0]
    assert r["Wear_method"] == "baseline_difference"
    V_true = areas[600] * 1e-6 * 6.25
    assert r["Wear_Volume_mm3"] == pytest.approx(V_true, rel=0.03)
    S = physics.sliding_distance_m(4, 6.25, r["sliding_time_s"])
    assert r["Specific_Wear_Rate_mm3_per_Nm"] == pytest.approx(r["Wear_Volume_mm3"] / (2 * S))
    assert r["n_checkpoints"] == 3
    ts = pd.read_csv(tmp_path / "processed" / "cof_timeseries.csv")
    assert {"Experiment_ID", "Time_s", "COF", "Carbon_pct", "Load_N", "Freq_Hz"} <= set(ts.columns)
