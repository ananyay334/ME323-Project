"""End-to-end CLI smoke test on synthetic data (reduced model / regime set)."""
import pandas as pd
import pytest
import yaml

from ml import train
from ml.config import DEFAULT_CONFIG


def _config(synth_dir, tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"]["synthetic"].update({"runs": str(synth_dir / "runs_targets.csv"),
                                     "timeseries": str(synth_dir / "cof_timeseries.csv"),
                                     "wear_trajectory": str(synth_dir / "wear_trajectory.csv")})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_train_cli_end_to_end(synth_dir, tmp_path):
    out = train.main(["--config", str(_config(synth_dir, tmp_path)), "--data", "synthetic", "--quick",
                      "--targets", "endpoint", "trajectory_params", "wear_checkpoints",
                      "--models", "gpr", "rf", "ridge", "--n-jobs", "2",
                      "--out", str(tmp_path / "results")])
    for f in ["metrics.csv", "predictions.csv", "report.md", "config_snapshot.yaml",
              "metrics_folds.csv", "best_models.csv"]:
        assert (out / f).exists(), f
    m = pd.read_csv(out / "metrics.csv")
    assert {"gkf5", "loao"} == set(m["split"])
    assert "physics_route" in set(m["model"])
    assert set(pd.read_csv(out / "best_models.csv")["model"]) <= {"gpr", "rf"}
    for fig in ["parity_best.png", "gpr_surface_log10_k.png", "gpr_lengthscales.png",
                "loao_per_grade.png", "trajectory_reconstruction.png", "gpr_calibration.png"]:
        assert (out / "figures" / fig).exists(), fig
    report = (out / "report.md").read_text()
    assert "Noise floor" in report and "Leave-one-alloy-out" in report
    assert not (out / "failures.csv").exists()


def test_targets_resolution():
    cfg = {"regimes": {"endpoint": ["log10_k", "log10_V"], "cof_time": ["COF_t"]}}
    assert train.resolve_targets(cfg, ["endpoint"]) == ["log10_k", "log10_V"]
    assert train.resolve_targets(cfg, ["COF_t", "endpoint"]) == ["COF_t", "log10_k", "log10_V"]
    with pytest.raises(SystemExit):
        train.resolve_targets(cfg, ["nope"])


def test_real_data_without_usable_runs_exits_cleanly(tmp_path):
    bad = tmp_path / "runs.csv"
    pd.DataFrame({"Experiment_ID": ["TRIAL_x"], "status": ["partial"], "Load_N": [2]}).to_csv(bad, index=False)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["data"]["real"]["runs"] = str(bad)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as e:
        train.main(["--config", str(tmp_path / "c.yaml"), "--data", "real", "--quick",
                    "--out", str(tmp_path / "res")])
    assert e.value.code == 2
