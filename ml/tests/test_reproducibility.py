"""Deterministic results with a fixed seed."""
import pandas as pd

from ml.evaluate import cross_validate, make_design
from ml.models.registry import select_models
from ml.splits import SplitPlan
from ml.synthetic import generate


def test_synthetic_is_deterministic(synth_dir, tmp_path):
    generate(tmp_path, seed=323, verbose=False)
    for f in ("runs_targets.csv", "wear_trajectory.csv", "run_sheet.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(synth_dir / f), pd.read_csv(tmp_path / f))


def test_split_plan_is_deterministic(data, cfg):
    a, b = SplitPlan(data.runs, cfg), SplitPlan(data.runs, cfg)
    assert a.gkf5 == b.gkf5 and a.offsets == b.offsets and a.loao == b.loao


def test_cv_predictions_are_reproducible(data, cfg):
    """Same seed -> identical predictions, serial or parallel."""
    plan = SplitPlan(data.runs, cfg)
    designs = [make_design(data, "log10_k", "base", cfg)]
    specs, _ = select_models(cfg, ["gpr", "svr", "rf", "xgb"])
    a = cross_validate(designs, specs, cfg, plan, n_jobs=1).predictions
    b = cross_validate(designs, specs, cfg, plan, n_jobs=2).predictions
    key = ["model", "split", "repeat", "fold", "Experiment_ID"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + ["y_pred", "y_std"]], b[key + ["y_pred", "y_std"]])
