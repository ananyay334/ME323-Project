"""CLI: the test set stays sealed without --final, and --final refuses to rerun."""
import pandas as pd
import pytest
import yaml

from ml import data as data_mod
from ml import train
from ml.config import DEFAULT_CONFIG

TEST_RUNS = {f"R{i}" for i in range(24, 30)}


def _config(synth_dir, tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"]["synthetic"].update({"checkpoints": str(synth_dir / "runs_checkpoints.csv"),
                                     "cof_cycles": str(synth_dir / "cof_cycles.csv"),
                                     "topup_candidates": str(synth_dir / "topup_candidates.csv")})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _spy(monkeypatch):
    calls = []
    real = data_mod.load_data

    def spy(cfg, source, split="train"):
        calls.append(split)
        return real(cfg, source, split)
    monkeypatch.setattr(train, "load_data", spy)
    return calls


def test_train_never_loads_test_rows(synth_dir, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    out = train.main(["--config", str(_config(synth_dir, tmp_path)), "--data", "synthetic", "--quick",
                      "--models", "ann", "gpr", "ridge", "--n-jobs", "2", "--out", str(tmp_path / "res")])
    assert set(calls) == {"train"}
    for f in ["metrics.csv", "predictions.csv", "report.md", "config_snapshot.yaml", "best_models.csv",
              "figures/parity_cv.png", "figures/gpr_surface_log10_k.png", "figures/cof_replicates.png",
              "interpretation/exponents.csv"]:
        assert (out / f).exists(), f
    assert not set(pd.read_csv(out / "predictions.csv")["Experiment_ID"]) & TEST_RUNS
    assert not (out / "test_metrics.csv").exists()
    rep = (out / "report.md").read_text()
    assert "Noise floor" in rep and "Which factor controls" in rep and "Test set" not in rep


def test_final_scores_once_and_refuses_rerun(synth_dir, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    args = ["--config", str(_config(synth_dir, tmp_path)), "--data", "synthetic", "--quick",
            "--models", "gpr", "ridge", "--n-jobs", "1", "--no-interpret", "--out", str(tmp_path / "res"), "--final"]
    out = train.main(args)
    assert out == tmp_path / "res" / "final_synthetic" and calls == ["train", "test"]
    tm = pd.read_csv(out / "test_metrics.csv")
    assert set(tm["target"]) == {"log10_k", "COF_steady"} and set(tm["model"]) == {"gpr"}
    tp = pd.read_csv(out / "test_predictions.csv")
    assert set(tp["Experiment_ID"]) == TEST_RUNS
    assert "Test set" in (out / "report.md").read_text() and (out / "figures" / "parity_test.png").exists()
    with pytest.raises(SystemExit) as e:
        train.main(args)
    assert e.value.code == 4
    assert train.main(args + ["--force"]) == out


def test_real_data_without_usable_rows_exits_cleanly(tmp_path):
    bad = tmp_path / "runs.csv"
    pd.DataFrame({"Experiment_ID": ["R00"], "Role": ["completed"], "status": ["error"], "Load_N": [10],
                  "Freq_Hz": [10], "Time_s": [60], "Checkpoint": [1]}).to_csv(bad, index=False)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"]["real"]["checkpoints"] = str(bad)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as e:
        train.main(["--config", str(tmp_path / "c.yaml"), "--data", "real", "--quick", "--out", str(tmp_path / "r")])
    assert e.value.code == 2
