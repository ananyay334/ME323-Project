"""Real-data acceptance tests (skipped when testdata/ is absent). Uses the extraction cache when present."""

import numpy as np
import pandas as pd
import pytest

from tribo_final.io.layout import find_runs, load_doe
from tribo_final.io.lrx import parse_lrx


@pytest.fixture(scope="module")
def real(cfg, paths):
    if not paths.testdata.exists():
        pytest.skip("testdata/ not found")
    from tribo_final.extract.dataset import build_dataset, extract_all

    paths.make()
    d = build_dataset(extract_all(cfg, paths), cfg, paths)
    return d


def test_run_folders_and_parameters(paths):
    if not paths.testdata.exists():
        pytest.skip("testdata/ not found")
    runs = find_runs(paths.testdata)
    assert len(runs) == 23
    doe = load_doe(paths.inputs / "run_sheet.csv").set_index("Experiment_ID")
    for r in runs:
        p = parse_lrx(r.lrx)
        row = doe.loc[r.folder_id]
        assert (p["Load_N"], p["Freq_Hz"], p["Run_time_s"]) == (row.Load_N, row.Freq_Hz, row.Run_time_s), r.folder_id
        assert len(r.scans) == 4 and sorted(s.location for s in r.scans) == ["a", "b", "c", "m"]


def test_r04(real):
    r = real["runs"].set_index("run_id").loc["R04"]
    assert (r.Load_N, r.Freq_Hz, r.Run_time_s) == (32, 7.5, 84)
    assert abs(r.cycles - 626) <= 10
    assert abs(r.COF_end - 0.155) <= 0.015
    assert abs(r.track_width_um - 414) <= 60


def test_r01_static_step(real):
    r = real["runs"].set_index("run_id").loc["R01"]
    assert abs(r.COF_instrument_static - 0.049) < 0.005


def test_cycles_within_2pct(real):
    assert (real["runs"].cycles_ratio_ft - 1).abs().max() <= 0.02


def test_wear_status(real):
    runs = real["runs"]
    det = runs[runs.wear_status == "detected"].run_id.tolist()
    low = sorted(runs[runs.wear_status == "below_LOD"].run_id)
    nm = sorted(runs[runs.wear_status == "not_measurable"].run_id)
    assert abs(len(det) - 15) <= 1
    assert low == ["R04", "R06", "R07", "R09", "R10", "R29"]
    # documented deviation: with the per-row high-pass, R19 location b has a two-sided reference (README 6.2)
    assert "R12" in nm and set(nm) <= {"R12", "R19"}


def test_wear_status_anchor_implementation(real):
    """With the earlier 2-D box high-pass the reference wear status is reproduced exactly."""
    v = real["wear_variants"].set_index("run_id")
    assert (v.box__status == "detected").sum() == 15
    assert sorted(v.index[v.box__status == "below_LOD"]) == ["R04", "R06", "R07", "R09", "R10", "R29"]
    assert sorted(v.index[v.box__status == "not_measurable"]) == ["R12", "R19"]


def test_replicate_noise_floor(real):
    from tribo_final.evaluate.interpret import noise_floor
    from tribo_final.models.registry import TARGETS, target_frame

    runs = real["runs"]
    for k in ("R04", "R14", "R20"):
        assert runs.set_index("run_id").at[k, "replicate_group"] == "R04"
    w = noise_floor(target_frame(runs, TARGETS["log10_V"]))
    assert w["replicate_runs"] == "R14,R20"  # R04 is below the LOD
    assert w["replicate_sd"] < 0.2
    c = noise_floor(target_frame(runs, TARGETS["COF_end"]))
    assert c["replicate_sd"] < 0.01
