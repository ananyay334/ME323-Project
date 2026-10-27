"""ml.maxwear_model.peak_wear_table: most-worn valid location, censoring, k formula."""
import numpy as np
import pandas as pd
import pytest

from ml.maxwear_model import peak_wear_table


def test_peak_wear_table():
    runs = pd.DataFrame({"Experiment_ID": ["A", "B", "C"], "Load_N": [10.0, 20.0, 100.0],
                         "Distance_m": [10.0, 10.0, 10.0], "k_mm3_per_Nm": [np.nan, 1e-5, np.nan]})
    loc = pd.DataFrame([
        # run A: biggest groove is one-sided (ignored); valid max = 300 > LOD 200 -> detected
        ("A", "a", 900.0, 100.0, "reference_one_sided"), ("A", "b", 300.0, 200.0, ""), ("A", "c", 50.0, 200.0, ""),
        # run B: valid max 150 < its LOD 250 -> below_LOD
        ("B", "a", 150.0, 250.0, ""), ("B", "b", 20.0, 100.0, np.nan), ("B", "m", np.nan, np.nan, "no_track_found"),
        # run C: only one-sided scans -> not measurable
        ("C", "a", 5000.0, 300.0, "track_touches_fov_edge;reference_one_sided"),
    ], columns=["Experiment_ID", "Location", "Groove_area_um2", "Area_LOD_um2", "wear_flags"])
    loc["Checkpoint"] = 2
    t = peak_wear_table(runs, loc).set_index("Experiment_ID")
    assert t.loc["A", "max_location"] == "b" and t.loc["A", "wear_status_max"] == "detected"
    assert t.loc["A", "k_max"] == pytest.approx(300e-6 * 10 / (10 * 10))          # area * 10 mm / (F S)
    assert t.loc["B", "wear_status_max"] == "below_LOD" and np.isnan(t.loc["B", "k_max"])
    assert t.loc["B", "log10_kmax_imp"] == pytest.approx(np.log10(250e-6 * 10 / (20 * 10) / 2))
    assert t.loc["C", "wear_status_max"] == "not_measurable" and t.loc["C", "n_valid_locations"] == 0
    assert t.loc["B", "log10_kavg"] == pytest.approx(-5.0)
