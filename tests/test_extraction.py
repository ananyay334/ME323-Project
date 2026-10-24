"""Regression tests for the target extraction.

Run from the project folder:   python -m pytest -q
They use the TA's practice run in data/raw/TRIAL_2026-09-22 as a fixture, synthetic wear
scars of known size painted into that screenshot, and synthetic 1000 Hz CSVs / height maps
of the 304 SS DOE (tests/synth.py).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth import add_groove, height_map, tribo_csv
from tribo_extract import physics
from tribo_extract.cof import extract_cof, extract_cof_cycles
from tribo_extract.config import PROJECT_ROOT, load_config
from tribo_extract.height_io import read_height
from tribo_extract.layout import classify_run, organize
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


# ----------------------------------------------------------------- per-cycle COF
@pytest.mark.parametrize("position", [False, True])
def test_cof_cycles(tmp_path, position):
    """Square-wave Fx (mu = 0.15) + 0.29 N offset + noise + reversal spikes + 1 s ramps."""
    true = tribo_csv(tmp_path / "run.csv", mu=0.15, freq=10.0, step_s=60.0, position=position)
    run = read_tribometer_csv(tmp_path / "run.csv", CFG["csv"]["columns"])
    cps, cycles, info = extract_cof_cycles(run, CFG, load_N=20.0, freq_hz=10.0)
    assert [c["Checkpoint"] for c in cps] == [1, 2]
    for c, n in zip(cps, true):
        assert c["COF_steady"] == pytest.approx(0.15, rel=0.02)
        assert abs(c["Cycles_step"] - n) <= 1
    assert cps[1]["Cycles_cum"] == cps[0]["Cycles_step"] + cps[1]["Cycles_step"]
    assert cps[1]["Distance_m"] == pytest.approx(4 * cps[1]["Stroke_amp_mm"] * cps[1]["Cycles_cum"] / 1000)
    assert ("stroke_nominal" in info["flags"]) == (not position)
    if position:
        assert cps[0]["Stroke_amp_mm"] == pytest.approx(5.0, rel=0.01)
    assert cycles["t_s"].min() >= 1.0 and set(cycles["Step"]) == {1, 2}


# ----------------------------------------------------------------- file layout
def _sheet(path, rows):
    pd.DataFrame(rows, columns=RUN_SHEET_COLUMNS).to_csv(path, index=False)


def _cfg(tmp_path, wear_method="groove"):
    """Config on tmp folders; the synthetic height maps are smooth -> groove method by default."""
    return load_config(overrides={"paths": {"raw_dir": str(tmp_path / "raw"),
                                            "run_sheet": str(tmp_path / "run_sheet.csv"),
                                            "processed_dir": str(tmp_path / "processed"),
                                            "reports_dir": str(tmp_path / "reports")},
                                  "wear": {"method": wear_method}})


def test_organize(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    for n in ("R14_a.bcrf", "R00.csv", "32_7.5.csv", "10_10.csv", "notes.txt"):
        (dump / n).write_bytes(n.encode())
    rows = [{"Experiment_ID": "R00", "Load_N": 10, "Freq_Hz": 10, "Include": "Y"},
                   {"Experiment_ID": "R04", "Load_N": 32, "Freq_Hz": 7.5, "Include": "Y"},
                   {"Experiment_ID": "R14", "Load_N": 32, "Freq_Hz": 7.5, "Replicate_of": "R04", "Include": "Y"},
                   {"Experiment_ID": "R20", "Load_N": 32, "Freq_Hz": 7.5, "Replicate_of": "R04", "Include": "Y"}]
    _sheet(tmp_path / "run_sheet.csv", rows)
    cfg = _cfg(tmp_path)
    copied, un = organize(dump, cfg, log=lambda *a: None)
    raw = tmp_path / "raw"
    assert (raw / "R14" / "R14_a.bcrf").exists() and (raw / "R00" / "R00.csv").exists()
    assert (raw / "R00" / "10_10.csv").exists()                     # unique (load, freq)
    un = pd.read_csv(raw / "_unassigned.csv")
    amb = un.set_index("file").loc["32_7.5.csv"]
    assert amb["reason"] == "ambiguous_load_freq" and set(amb["candidates"].split(";")) == {"R04", "R14", "R20"}
    assert "notes.txt" in set(un["file"])
    assert not any((raw / r / "32_7.5.csv").exists() for r in ("R04", "R14", "R20"))


def test_checkpoint_ladder(tmp_path):
    lay, t0 = CFG["layout"], 1.7e9
    d = tmp_path / "R02"
    for sub in ("base", "cp1", "cp2"):                               # rule 2: sub-folder names
        (d / sub).mkdir(parents=True)
        (d / sub / "R02_b.bcrf").write_bytes(sub.encode())
    for n in ("R02_c_base.bcrf", "R02_c_t60s.bcrf", "R02_c_cp2.bcrf"):    # rule 1: name tokens
        (d / n).write_bytes(n.encode())
    (d / "R02.bcrf").write_bytes(b"m")                                  # rule 4: one scan -> cp2
    _, scans, issues = classify_run(d, lay, t_run=120)
    got = {(r.location, r.checkpoint, r.checkpoint_source) for r in scans.itertuples()}
    assert got == {("c", "base", "filename"), ("c", "cp1", "filename"), ("c", "cp2", "filename"),
                   ("b", "base", "subfolder"), ("b", "cp1", "subfolder"), ("b", "cp2", "subfolder"),
                   ("m", "cp2", "mtime")} and not issues
    e = tmp_path / "R05"                                               # rule 4: 3 scans by mtime
    e.mkdir()
    for n, dt in (("R05_b (3).bcrf", 30), ("R05_b (1).bcrf", 10), ("R05_b (2).bcrf", 20)):
        (e / n).write_bytes(b"x")
        os.utime(e / n, (t0 + dt, t0 + dt))
    _, scans, _ = classify_run(e, lay, t_run=60)
    assert dict(zip(scans["key"], scans["checkpoint"])) == {"R05_b (1)": "base", "R05_b (2)": "cp1",
                                                            "R05_b (3)": "cp2"}
    assert set(scans["checkpoint_source"]) == {"mtime"}


# ----------------------------------------------------------------- RAW reader
def _part0_raw():
    raw = PROJECT_ROOT / "data" / "raw"
    return next((p for p in sorted(raw.rglob("*.raw"))), None) if raw.exists() else None


@pytest.mark.skipif(_part0_raw() is None, reason="no Part-0 RAW sample in data/raw yet")
def test_raw_reader_part0_sample():
    p = _part0_raw()
    hm = read_height(p, CFG)
    assert hm.z.ndim == 2 and min(hm.z.shape) >= 8 and np.isfinite(hm.z).mean() > 0.5
    if "preview_r" in hm.info and np.isfinite(hm.info["preview_r"]):
        assert hm.info["preview_r"] > 0.9
    else:
        assert hm.info["validated"], hm.info


@pytest.mark.parametrize("with_meta", [False, True])
def test_raw_reader_synthetic(tmp_path, with_meta):
    """Headerless/unknown float32 RAW: metadata ladder step, else brute force, recovers the map."""
    z, _ = height_map((150, 240), width_um=120, depth_um=1.2, angle_deg=4, seed=3)
    p = tmp_path / "R07_a.raw"
    with open(p, "wb") as fh:
        fh.write(b"\x00" * 512)                                     # unknown header
        fh.write((z * 1e-6).astype("<f4").tobytes())                # metres
    if with_meta:
        (tmp_path / "R07_a.txt").write_text("Width = 240\nHeight = 150\nData type = float32\n"
                                            "Header size = 512\nZ scale (um) = 1e6\nPixel size (um) = 3.0\n")
    hm = read_height(p, CFG, verbose=False)
    assert hm.z.shape == z.shape
    assert hm.info["read_method"] == ("metadata" if with_meta else "brute_force")
    assert np.corrcoef(hm.z.ravel(), z.ravel())[0, 1] > 0.999
    assert np.nanmax(hm.z) - np.nanmin(hm.z) == pytest.approx(np.ptp(z), rel=1e-3)   # um
    assert hm.info["validated"]


# ----------------------------------------------------------------- end to end
def test_pipeline_end_to_end(tmp_path):
    """4 locations x (base, cp1, cp2) height maps with known grooves + a 1000 Hz CSV."""
    run = tmp_path / "raw" / "R02"
    run.mkdir(parents=True)
    tribo_csv(run / "R02.csv", mu=0.15, fz=20.0, freq=10.0, step_s=60.0)
    W, D = 150.0, {"base": 0.0, "cp1": 1.0, "cp2": 1.6}
    true = {}
    for li, loc in enumerate(["", "_a", "_b", "_c"]):
        for k, (cp, depth) in enumerate(D.items()):
            z, area = height_map(width_um=W if depth else 0.0, depth_um=depth, angle_deg=3,
                                 offset_um=10 * li, seed=li, shift_px=(2 * k, -3 * k))
            np.savetxt(run / f"R02{loc}_{cp}_height.csv", z, delimiter=",", fmt="%.5f")
            true[cp] = area
    _sheet(tmp_path / "run_sheet.csv", [{"Experiment_ID": "R02", "Load_N": 20, "Freq_Hz": 10,
                                         "Run_time_s": 120, "Role": "corner", "Include": "Y",
                                         "Ball_scar_d1_um": 400, "Ball_scar_d2_um": 420}])
    cfg = _cfg(tmp_path)
    runs = process_all(cfg, make_plots=False, log=lambda *a: None)
    assert list(runs["Checkpoint"]) == [1, 2] and (runs["status"] == "ok").all()
    ell = cfg["wear"]["track_length_mm"]
    for _, r in runs.iterrows():
        V_true = true[f"cp{r['Checkpoint']}"] * 1e-6 * ell
        assert r["Groove_volume_mm3"] == pytest.approx(V_true, rel=0.05)
        assert r["n_locations"] == 4 and r["Wear_detected"]
        assert r["k_mm3_per_Nm"] == pytest.approx(r["Groove_volume_mm3"] / (20 * r["Distance_m"]))
        assert r["Net_volume_mm3"] == pytest.approx(r["Groove_volume_mm3"] - r["Pileup_volume_mm3"])
        assert r["COF_steady"] == pytest.approx(0.15, rel=0.02)
    assert runs.loc[1, "Distance_m"] == pytest.approx(physics.sliding_distance_m(10, 5.0, 118, "amplitude"),
                                                      rel=0.01)       # 1 s ramps: N = f (t - 1) per step
    d = 0.41
    assert runs.loc[1, "V_ball_mm3"] == pytest.approx(np.pi * d ** 4 / (64 * 3.0))
    out = tmp_path / "processed"
    assert {p.name for p in out.iterdir()} == {"runs_checkpoints.csv", "cof_cycles.csv", "wear_locations.csv"}
    locs = pd.read_csv(out / "wear_locations.csv")
    assert len(locs) == 8 and set(locs["wear_method"]) == {"baseline_difference"}
    cyc = pd.read_csv(out / "cof_cycles.csv")
    assert list(cyc.columns) == ["Experiment_ID", "Step", "Cycle", "t_s", "COF_c", "Fz_N", "Halfrange_N", "Amp_mm"]
