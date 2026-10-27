import numpy as np
import pytest

from tribo_final.extract.cof import extract_cof, find_sliding_step, measure_params
from tribo_final.io.csv_reader import read_friction_csv

from conftest import write_rtec_csv


@pytest.mark.parametrize("with_freq", [False, True])
def test_static_excluded_offset_cancels_cycles_counted(tmp_path, cfg, with_freq):
    p = tmp_path / "R98.csv"
    true_cycles = write_rtec_csv(p, f_hz=5.0, slide_s=30.0, mu=0.15, offset=0.3, with_freq=with_freq)
    log = read_friction_csv(p)
    step = find_sliding_step(log.data, cfg["cof"])
    assert (log.data.recipe_step.iloc[step.start:step.stop] == 2).all(), "static step counted as sliding"
    res, cyc = extract_cof(log, cfg["cof"], 5.0, {"Freq_Hz": 5.0, "Run_time_s": 30.0})
    for k in ("COF_first10s", "COF_end", "COF_whole"):
        assert abs(res[k] / 0.15 - 1) < 0.03, (k, res[k])
    assert abs(res["cycles"] - true_cycles) <= 2
    assert abs(res["Fx_offset_N"] - 0.3) < 0.05
    # in the static step the instrument channel shows only the sensor offset
    assert res["COF_instrument_static"] == pytest.approx(0.3 / 20, rel=0.05)
    assert cyc.t_s.min() >= 1.0  # ramps trimmed


def test_measure_params_without_lrx(tmp_path, cfg):
    p = tmp_path / "R97.csv"
    write_rtec_csv(p, f_hz=6.0, slide_s=40.0, fz=24.5)
    m = measure_params(read_friction_csv(p), cfg["cof"])
    assert m["Load_N"] == 24.5
    assert abs(m["Run_time_s"] - 40) <= 1
    assert abs(m["Freq_Hz"] - 6.0) / 6.0 < 0.05
