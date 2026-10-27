"""ml.campaign_model: data loading, leak-free leave-one-group-out CV, physics-model recovery."""
import numpy as np
import pandas as pd
import pytest

from ml import campaign_model as cm


def _table(tmp_path, seed=0, n=18):
    rng = np.random.default_rng(seed)
    L = np.round(np.geomspace(10, 100, n), 1)
    day = np.tile([0, 1], n // 2)
    rows = []
    for i, (l, k) in enumerate(zip(L, day)):
        rid = f"R{i + 1:02d}"
        cof = 0.12 + 0.06 * l ** (-1 / 3) - 0.015 * k + rng.normal(0, 0.002)
        w = 100 * l ** 0.33 * (0.8 if k else 1.0) * 10 ** rng.normal(0, 0.02)
        for cp in (1, 2):
            rows.append({"Experiment_ID": rid, "Checkpoint": cp, "status": "ok",
                         "Test_date": f"2026-10-{13 + 3 * k}T10:{i:02d}:00", "Load_N": l,
                         "Freq_Hz": rng.choice([4, 7.5, 15]), "Run_time_s": rng.choice([60, 90, 120]),
                         "COF_steady": cof, "Track_width_um": w, "Replicate_of": ""})
    df = pd.DataFrame(rows)
    df.loc[df["Experiment_ID"] == "R03", "Replicate_of"] = "R01"          # one replicate pair
    p = tmp_path / "runs_checkpoints.csv"
    df.to_csv(p, index=False)
    return p


def test_load_runs_one_row_per_run_with_day_and_groups(tmp_path):
    d = cm.load_runs(_table(tmp_path))
    assert len(d) == 18 and set(d["day"]) == {0.0, 1.0}
    assert d.set_index("Experiment_ID").loc["R03", "group"] == "R01"
    assert np.allclose(d["L_m13"], d["Load_N"] ** (-1 / 3))


def test_logo_never_uses_the_held_out_group(tmp_path):
    d = cm.load_runs(_table(tmp_path))
    p = cm.logo_predict(d, "COF", [], "mean")
    for g in d["group"].unique():
        te = (d["group"] == g).to_numpy()
        assert np.allclose(p[te], d.loc[~te, "COF"].mean())


def test_run_recovers_the_physics_coefficients(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MODELS", ["linear", "ridge"])
    out = tmp_path / "out"
    ctx = cm.run(_table(tmp_path), tmp_path / "missing.csv", out, quick=True, n_boot=300, n_jobs=1)
    c = ctx["coefs"].set_index(["target", "term"])["estimate"]
    assert c[("COF", "16 Oct offset")] == pytest.approx(-0.015, abs=0.004)
    assert c[("log10_w", "b: load exponent")] == pytest.approx(0.33, abs=0.06)
    assert c[("log10_w", "16 Oct width factor")] == pytest.approx(0.8, abs=0.06)
    cv = ctx["cv"]
    phys = cv[(cv["target"] == "COF") & (cv["inputs"] == cm.PHYSICS["COF"]) & (cv["model"] == "linear")].iloc[0]
    assert phys["R2"] > 0.8 and phys["tied_with_best"] == "yes"
    assert "L^-1/3 + plate roughness" not in set(cv["inputs"])          # skipped: no roughness file
    for f in ("report.md", "cv_metrics.csv", "coefficients.csv", "figures/cof_vs_load_by_day.png",
              "figures/width_vs_load_by_day.png", "figures/parity_physics_loo.png"):
        assert (out / f).exists(), f
