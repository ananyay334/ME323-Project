"""End-to-end CLI smoke test on synthetic data (reduced model set)."""
import pandas as pd
import pytest
import yaml

from ml.config import DEFAULT_CONFIG
from ml import train


def _config_file(synth_dir, tmp_path):
    with open(DEFAULT_CONFIG) as fh:
        raw = yaml.safe_load(fh)
    raw["data"]["synthetic"].update({"runs": str(synth_dir / "runs_targets.csv"),
                                     "timeseries": str(synth_dir / "cof_timeseries.csv"),
                                     "checkpoints": str(synth_dir / "wear_checkpoints.csv")})
    raw["quick"]["active_learning"] = {"repeats": 2, "budget": 22}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_train_cli_end_to_end(synth_dir, tmp_path):
    cfg_path = _config_file(synth_dir, tmp_path)
    out = train.main(["--config", str(cfg_path), "--data", "synthetic", "--quick",
                      "--models", "ridge", "gpr", "kan", "mt_mlp", "--n-jobs", "2",
                      "--out", str(tmp_path / "results")])
    for f in ["metrics.csv", "metrics_folds.csv", "predictions.csv", "report.md",
              "config_snapshot.yaml", "environment.json", "holdout_predictions.csv",
              "final_selection.csv", "run.log"]:
        assert (out / f).exists(), f
    m = pd.read_csv(out / "metrics.csv")
    assert {"log10_k", "COF_ss_mean", "COF_t", "COF_t@ss"} <= set(m["target"])
    assert {"base", "physics"} == set(m["feature_set"])
    assert {"mt_mlp", "kan", "gpr", "ridge"} <= set(m["model"])
    assert any(out.joinpath("figures").glob("gpr_map_*.png"))
    assert any(out.joinpath("figures").glob("al_learning_curves_*.png"))
    assert any(out.joinpath("models").glob("*.joblib"))
    h = pd.read_csv(out / "holdout_predictions.csv")
    assert set(h["Experiment_ID"]) == {f"U0{i}" for i in range(1, 7)}
    report = (out / "report.md").read_text()
    for section in ["## 1. Data", "## 3. Cross-validated results", "## 5. Unseen operating points",
                    "## 6. Interpretation", "## 7. Extension A", "## 8. Extension B"]:
        assert section in report
    assert not (out / "failures.csv").exists()


def test_targets_resolution():
    cfg = {"targets": {"wear": ["log10_k", "log10_V"], "cof": ["COF_ss_mean", "COF_t"]}}
    assert train.resolve_targets(cfg, ["wear"]) == ["log10_k", "log10_V"]
    assert train.resolve_targets(cfg, ["COF_t", "wear"]) == ["COF_t", "log10_k", "log10_V"]
    with pytest.raises(SystemExit):
        train.resolve_targets(cfg, ["nope"])


def test_real_data_without_usable_runs_exits_cleanly(synth_dir, tmp_path):
    bad = tmp_path / "runs.csv"
    pd.DataFrame({"Experiment_ID": ["TRIAL_x"], "status": ["partial"], "Load_N": [2]}).to_csv(bad, index=False)
    with open(DEFAULT_CONFIG) as fh:
        raw = yaml.safe_load(fh)
    raw["data"]["real"].update({"runs": str(bad), "timeseries": str(tmp_path / "none.csv")})
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as e:
        train.main(["--config", str(tmp_path / "c.yaml"), "--data", "real", "--quick",
                    "--out", str(tmp_path / "res")])
    assert e.value.code == 2
    assert not (tmp_path / "res").exists() or not any((tmp_path / "res").iterdir())
