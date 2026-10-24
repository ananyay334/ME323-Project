"""Tests for the real-campaign file formats and protocol (Oct 2026 data).

* BCR-F height maps, .lrx recipes, location suffixes R04a / R11_a / R01a_13;
* one continuous sliding step split at t_run/2, a static loading step with an Fx offset;
* run parameters read from the run's own files and matched to the DOE by value;
* rough-surface ("band") wear measurement;
* the real files in testdata/ when present (skipped otherwise).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from synth import tribo_csv
from tribo_extract.config import PROJECT_ROOT, load_config
from tribo_extract.cof import extract_cof_cycles, sliding_segments
from tribo_extract.height_io import read_bcrf, read_height
from tribo_extract.layout import classify_run
from tribo_extract.pipeline import RUN_SHEET_COLUMNS, match_doe, process_all, run_parameters
from tribo_extract.recipe import recipe_params
from tribo_extract.tribometer_csv import read_tribometer_csv
from tribo_extract.wear import measure_track_band
from tribo_extract.wli import HeightMap

CFG = load_config()
TESTDATA = PROJECT_ROOT / "testdata"


# ----------------------------------------------------------------- writers
def write_bcrf(path, z, px_um=1.875):
    """Unicode BCR-F like the RTEC export: UTF-16 header (2048 chars) + float32 data."""
    ny, nx = z.shape
    hdr = (f"fileformat = bcrf_unicode\n%\nheadersize = 2048\nxpixels = {nx}\nypixels = {ny}\n"
           f"zmin = {np.nanmin(z):.5f}\nintelmode = 1\nxlength = {nx * px_um * 1000:.4f}\n"
           f"ylength = {ny * px_um * 1000:.4f}\nxoffset = 0\nyoffset = 0\nvoidpixels = 3.402823E+38\n"
           "bit2nm = 1\n xunit = nm\n yunit = nm\n zunit = um\nforcecurve = 0\n")
    raw = hdr.ljust(2048).encode("utf-16-le")
    data = np.where(np.isfinite(z), z, 3.402823e38).astype("<f4").tobytes()
    Path(path).write_bytes(raw + data)


def write_lrx(path, load, minutes, freq):
    Path(path).write_text(
        '<?xml version="1.0" standalone="yes"?>\n<TribologyRecipeSummary>\n'
        f'  <Steps StepKey="0"><StepDetail>1&gt; Fz:{load} N,  t:0.03333 min\nNone\n</StepDetail></Steps>\n'
        f'  <Steps StepKey="1"><StepDetail>2&gt; Fz:{load} N,  t:{minutes} min\nR - Continuous\n'
        f'Frequency (Hz): {freq}\nAcceleration (s): 1\nDeceleration (s): 1\nNone\n</StepDetail></Steps>\n'
        '</TribologyRecipeSummary>\n')


def rough_track(ny=600, nx=960, depth=1.0, ridge=2.5, seed=0):
    """Rough surface (Sq 3 um) with a flattened 1 um-deep band (rows 230-450) and two ridges."""
    rng = np.random.default_rng(seed)
    tex = ndi.gaussian_filter(rng.normal(size=(ny, nx)), 6)
    z = tex / tex.std() * 3.0
    y = np.arange(ny)[:, None]
    c0, c1 = 230, 450
    z[c0:c1] = -depth + 0.2 * z[c0:c1] / 3.0
    rid = ridge * (np.exp(-((y - (c0 - 30)) ** 2) / (2 * 16 ** 2)) + np.exp(-((y - (c1 + 30)) ** 2) / (2 * 16 ** 2)))
    return z + rid * (1 - ((y >= c0) & (y < c1))), c1 - c0


# ----------------------------------------------------------------- readers / layout
def test_bcrf_roundtrip(tmp_path):
    z = np.random.default_rng(1).normal(size=(40, 64))
    z[3, 5] = np.nan
    write_bcrf(tmp_path / "R04a.bcrf", z, px_um=1.875)
    hm = read_height(tmp_path / "R04a.bcrf", CFG)
    assert hm.info["read_method"] == "bcrf" and hm.z.shape == (40, 64)
    assert hm.dx_um == pytest.approx(1.875) and hm.dy_um == pytest.approx(1.875)
    assert np.isnan(hm.z[3, 5]) and np.allclose(hm.z[np.isfinite(z)], z[np.isfinite(z)], atol=1e-6)


def test_location_suffix_variants(tmp_path):
    d = tmp_path / "R01"
    d.mkdir()
    for n in ("R01_13.bcrf", "R01a_13.bcrf", "R01_b.bcrf", "R01c.bcrf", "R01.csv", "R01.lrx",
              "R01a_13.info", "R01.Raw", "R01.rtec", "R01.enc"):
        (d / n).write_bytes(b"x")
    by_role, scans, issues = classify_run(d, CFG["layout"])
    assert {p.name for p in by_role["ignore"]} == {"R01.Raw", "R01.rtec", "R01.enc"}
    assert [p.name for p in by_role["recipe"]] == ["R01.lrx"]
    assert dict(zip(scans["key"], scans["location"])) == {"R01_13": "m", "R01a_13": "a", "R01_b": "b",
                                                         "R01c": "c"}
    assert set(scans["checkpoint"]) == {"cp2"} and not issues          # one scan per location


def test_recipe_params(tmp_path):
    write_lrx(tmp_path / "R06.lrx", 24.5, 1.93333, 4.5)
    p = recipe_params(tmp_path / "R06.lrx")
    assert (p["Load_N"], p["Freq_Hz"], p["Run_time_s"]) == (24.5, 4.5, 116.0)


# ----------------------------------------------------------------- COF protocol
def _single_step_csv(path, mu=0.15, fz=32.0, f=7.5, slide_s=84.0, fs=100, offset=1.0, seed=0):
    """Real protocol: static loading step (constant Fx offset) then ONE continuous sliding step."""
    rng = np.random.default_rng(seed)
    rows = []
    t = np.arange(int(2 * fs)) / fs
    for ti in t:
        rows.append((1, ti, 1, fz, offset / fz, offset, 0.5))
    t = np.arange(int(slide_s * fs)) / fs
    ft = f * np.clip(np.minimum(t, slide_s - t), 0, 1)
    phi = 2 * np.pi * np.cumsum(ft) / fs
    fx = mu * fz * np.sign(np.cos(phi)) * (ft > 0) + offset + rng.normal(0, 0.05, len(t))
    for ti, x in zip(t, fx):
        rows.append((2, ti, 2, fz + rng.normal(0, 0.2), abs(x) / fz, x, 0.5))
    lines = ["Recipe,Averaging,LogFrequency\rfake.Rx,10,1000",
             "Step, Timestamp, RecipeStep, DAQ.Fz (N),DAQ.COF (Fx),DAQ.Fx (N),XYZ.Z Depth (mm),"]
    lines += [f"{a},{b + 0.01:.3f},{c},{d:.4f},{e:.5f},{g:.4f},{h}," for a, b, c, d, e, g, h in rows]
    Path(path).write_text("\n".join(lines + ["", "TotalTime", "00:01:30.0"]) + "\n")
    return phi[-1] / (2 * np.pi)


def test_static_step_with_offset_is_not_sliding(tmp_path):
    _single_step_csv(tmp_path / "R04.csv", offset=1.0)
    run = read_tribometer_csv(tmp_path / "R04.csv", CFG["csv"]["columns"])
    segs = sliding_segments(run.data)
    assert len(segs) == 1 and set(run.data["recipe_step"].to_numpy()[segs[0]]) == {2}


def test_single_step_split_at_half(tmp_path):
    n_true = _single_step_csv(tmp_path / "R04.csv", mu=0.15, f=7.5, slide_s=84.0)
    run = read_tribometer_csv(tmp_path / "R04.csv", CFG["csv"]["columns"])
    cps, cycles, info = extract_cof_cycles(run, CFG, load_N=32, freq_hz=7.5, stroke_mm=5)
    assert "single_step_split_at_half" in info["flags"] and [c["Checkpoint"] for c in cps] == [1, 2]
    assert cps[0]["Step_duration_s"] == pytest.approx(42, abs=0.1)
    assert abs(cps[1]["Cycles_cum"] - n_true) <= 2                     # whole run, ramps included
    for c in cps:
        assert c["COF_steady"] == pytest.approx(0.15, rel=0.03)        # offset of 1 N cancels


# ----------------------------------------------------------------- parameters from files
def test_parameters_from_files_and_doe_matching(tmp_path):
    """Folder names do not have to follow the DOE: parameters come from the files and the DOE
    row (role, position) is matched by value; replicates = identical parameters."""
    sheet = pd.DataFrame([
        {"Experiment_ID": "R04", "Load_N": 32, "Freq_Hz": 7.5, "Run_time_s": 84, "Role": "center_replicate",
         "Face": "F1", "Include": "Y"},
        {"Experiment_ID": "R24", "Load_N": 23, "Freq_Hz": 5, "Run_time_s": 64, "Role": "test", "Include": "Y"},
    ], columns=RUN_SHEET_COLUMNS)
    # the folder called R09 actually contains the R24 test condition
    d = tmp_path / "raw" / "R09"
    d.mkdir(parents=True)
    write_lrx(d / "R09.lrx", 23, 1.06667, 5)
    p, src, _ = run_parameters(d, CFG)
    assert src == "recipe_lrx" and p == {"Load_N": 23.0, "Freq_Hz": 5.0, "Run_time_s": 64.0}
    row, doe_id = match_doe("R09", p, sheet)
    assert doe_id == "R24" and row["Role"] == "test"
    # CSV-only fallback (no recipe): measured Fz, reversal frequency, sliding time
    e = tmp_path / "raw" / "R30"
    e.mkdir()
    _single_step_csv(e / "R30.csv", fz=32.0, f=7.5, slide_s=84.0)
    p2, src2, _ = run_parameters(e, CFG)
    assert src2 == "csv_measured" and p2 == {"Load_N": 32.0, "Freq_Hz": 7.5, "Run_time_s": 84}
    assert match_doe("R30", p2, sheet)[1] == "R04"


def test_process_all_replicates_from_files(tmp_path):
    raw = tmp_path / "raw"
    z, _ = rough_track()
    for rid in ("R04", "R07", "R11"):                                   # R07/R11: same condition as R04
        d = raw / rid
        d.mkdir(parents=True)
        _single_step_csv(d / f"{rid}.csv", seed=int(rid[1:]))
        write_lrx(d / f"{rid}.lrx", 32, 1.4, 7.5)
        write_bcrf(d / f"{rid}.bcrf", z)
    pd.DataFrame([{"Experiment_ID": "R04", "Load_N": 32, "Freq_Hz": 7.5, "Run_time_s": 84,
                   "Role": "center_replicate", "Include": "Y"}],
                 columns=RUN_SHEET_COLUMNS).to_csv(tmp_path / "run_sheet.csv", index=False)
    cfg = load_config(overrides={"paths": {"raw_dir": str(raw), "run_sheet": str(tmp_path / "run_sheet.csv"),
                                           "processed_dir": str(tmp_path / "processed"),
                                           "reports_dir": str(tmp_path / "reports")}})
    runs = process_all(cfg, make_plots=False, log=lambda *a: None)
    cp2 = runs[runs["Checkpoint"] == 2].set_index("Experiment_ID")
    assert cp2.loc["R07", "Replicate_of"] == "R04" and cp2.loc["R11", "Replicate_of"] == "R04"
    assert (cp2["Role"] == "center_replicate").all() and (cp2["Params_source"] == "recipe_lrx").all()
    assert cp2["Track_width_um"].notna().all()
    assert runs[runs["Checkpoint"] == 1]["Groove_volume_mm3"].isna().all()   # scanned at the end only


# ----------------------------------------------------------------- rough-surface wear
def test_band_method_unbiased():
    eg, ew = [], []
    for seed in range(6):
        z, rows = rough_track(seed=seed)
        res, _ = measure_track_band(HeightMap(z=z, dx_um=1.875, dy_um=1.875, source="synthetic"), CFG)
        assert res["wear_detected"] and res["reference_ok"]
        eg.append(res["worn_area_um2"] / (rows * 1.875 * 1.0) - 1)
        ew.append(res["track_width_um"] / (rows * 1.875) - 1)
    assert abs(np.median(eg)) < 0.2 and abs(np.median(ew)) < 0.1


def test_band_method_no_track_on_unworn_surface():
    z = ndi.gaussian_filter(np.random.default_rng(3).normal(size=(600, 960)), 6) * 20
    res, _ = measure_track_band(HeightMap(z=z, dx_um=1.875, dy_um=1.875, source="synthetic"), CFG)
    assert not res["wear_detected"]


def test_band_method_flags_one_sided_reference():
    z, _ = rough_track()
    res, _ = measure_track_band(HeightMap(z=z[180:], dx_um=1.875, dy_um=1.875, source="synthetic"), CFG)
    assert not res["reference_ok"] and "reference_one_sided" in res["wear_flags"]


# ----------------------------------------------------------------- real campaign files
@pytest.mark.skipif(not (TESTDATA / "R04").exists(), reason="testdata/R04 not present")
def test_real_R04():
    p, src, _ = run_parameters(TESTDATA / "R04", CFG)
    assert src == "recipe_lrx" and p == {"Load_N": 32.0, "Freq_Hz": 7.5, "Run_time_s": 84.0}
    run = read_tribometer_csv(TESTDATA / "R04" / "R04.csv", CFG["csv"]["columns"])
    cps, _, _ = extract_cof_cycles(run, CFG, load_N=32, freq_hz=7.5, stroke_mm=5)
    assert 0.13 < cps[1]["COF_steady"] < 0.17 and 615 < cps[1]["Cycles_cum"] < 635
    hm = read_bcrf(TESTDATA / "R04" / "R04.bcrf")
    assert hm.z.shape == (600, 960) and hm.dx_um == pytest.approx(1.875, rel=1e-3)
    res, _ = measure_track_band(hm, CFG)
    assert res["wear_detected"] and 350 < res["track_width_um"] < 470
