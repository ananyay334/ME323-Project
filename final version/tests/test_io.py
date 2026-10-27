import numpy as np
import pandas as pd

from tribo_final.io.bcrf import read_bcrf, write_bcrf
from tribo_final.io.csv_reader import read_friction_csv
from tribo_final.io.layout import match_doe, replicate_groups, scan_location
from tribo_final.io.lrx import parse_lrx

from conftest import ROOT, write_rtec_csv

LRX = """<?xml version="1.0" standalone="yes"?>
<TribologyRecipeSummary>
  <Steps StepKey="0" StepName="" StepType="0" LoopFromStep="0">
    <StepDetail>1&gt; Fz:32 N,  t:0.03333 min
None
</StepDetail>
  </Steps>
  <Steps StepKey="1" StepName="" StepType="0" LoopFromStep="0">
    <StepDetail>2&gt; Fz:32 N,  t:1.4 min
R - Continuous
Frequency (Hz): 7.5
Acceleration (s): 1
Deceleration (s): 1
None
</StepDetail>
  </Steps>
</TribologyRecipeSummary>
"""


def test_lrx_parsing(tmp_path):
    p = tmp_path / "R04.lrx"
    p.write_text(LRX)
    r = parse_lrx(p)
    assert r["Load_N"] == 32 and r["Freq_Hz"] == 7.5 and r["Run_time_s"] == 84
    assert r["static_s"] == 0.03333 * 60
    assert [s["sliding"] for s in r["steps"]] == [False, True]


def test_bcrf_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    z = rng.normal(size=(60, 96)).astype(np.float32).astype(float)
    z[3, 7] = np.nan
    write_bcrf(tmp_path / "x.bcrf", z, pixel_um=1.875)
    hm = read_bcrf(tmp_path / "x.bcrf")
    assert hm.z.shape == (60, 96)
    assert np.isnan(hm.z[3, 7])
    assert np.allclose(hm.z[np.isfinite(z)], z[np.isfinite(z)])
    assert abs(hm.pixel_um - 1.875) < 1e-9


def test_location_naming():
    assert scan_location("R04.bcrf") == "m"
    assert scan_location("R04a.bcrf") == "a"
    assert scan_location("R04_a.bcrf") == "a"
    assert scan_location("R01a_13.bcrf") == "a"
    assert scan_location("R01_13.bcrf") == "m"
    assert scan_location("R11_c.bcrf") == "c"
    assert scan_location("R01b_13.bcrf") == "b"


def test_doe_matching_by_value(cfg):
    doe = pd.read_csv(ROOT / "inputs" / "run_sheet.csv")
    doe["Replicate_of"] = doe["Replicate_of"].fillna("").astype(str)
    # a folder named R09 holding the R24 condition (23 N, 5 Hz, 64 s) maps to R24
    assert match_doe({"Load_N": 23, "Freq_Hz": 5, "Run_time_s": 64}, doe, "R09", cfg["doe_match"]) == "R24"
    # replicates: the folder name breaks the tie between identical rows
    for rid in ("R04", "R14", "R20"):
        assert match_doe({"Load_N": 32, "Freq_Hz": 7.5, "Run_time_s": 84}, doe, rid, cfg["doe_match"]) == rid
    assert match_doe({"Load_N": 33.5, "Freq_Hz": 7.5, "Run_time_s": 84}, doe, "R04", cfg["doe_match"]) is None


def test_replicate_groups():
    df = pd.DataFrame({"run_id": ["R04", "R05", "R14", "R20"], "Load_N": [32, 10, 32, 32], "Freq_Hz": [7.5, 15, 7.5, 7.5],
                       "Run_time_s": [84, 60, 84, 84]})
    g = replicate_groups(df)
    assert list(g) == ["R04", "R05", "R04", "R04"]


def test_csv_reader_quirks(tmp_path):
    p = tmp_path / "R99.csv"
    write_rtec_csv(p, with_freq=True)
    log = read_friction_csv(p)
    assert log.fs == 100
    assert log.meta["Averaging"] == "10"
    assert log.test_date.day == 13
    d = log.data
    assert set(d.recipe_step) == {1, 2}
    assert np.all(np.diff(d.t) > 0)  # continuous clock across the restarting Timestamp
    assert "freq" in d.columns and len(d) == 200 + 3000
