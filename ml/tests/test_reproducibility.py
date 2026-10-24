"""Deterministic results with a fixed seed."""
import numpy as np
import pandas as pd

from ml.evaluate import cross_validate
from ml.models.ann import ANNEnsemble
from ml.models.registry import select_models
from ml.splits import SplitPlan
from ml.synthetic import generate


def test_synthetic_is_deterministic(synth_dir, tmp_path):
    generate(tmp_path, seed=323, verbose=False)
    for f in ("runs_checkpoints.csv", "cof_cycles.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(synth_dir / f), pd.read_csv(tmp_path / f))


def test_ann_is_deterministic(data):
    kw = dict(hidden=4, n_members=3, max_epochs=400, patience=50)
    a = ANNEnsemble(random_state=7, **kw).fit(data.X, data.Y, data.groups)
    b = ANNEnsemble(random_state=7, **kw).fit(data.X, data.Y, data.groups)
    c = ANNEnsemble(random_state=8, **kw).fit(data.X, data.Y, data.groups)
    np.testing.assert_array_equal(a.predict(data.X), b.predict(data.X))
    np.testing.assert_array_equal(a.predict_std(data.X), b.predict_std(data.X))
    assert not np.allclose(a.predict(data.X), c.predict(data.X))


def test_cv_predictions_are_reproducible(data, cfg):
    """Same seed -> identical out-of-fold predictions, serial or parallel."""
    plan = SplitPlan(data.groups, cfg)
    specs, _ = select_models(cfg, ["ann", "gpr", "svr", "xgb"])
    a = cross_validate(data, specs, cfg, plan, n_jobs=1).predictions
    b = cross_validate(data, specs, cfg, plan, n_jobs=2).predictions
    key = ["model", "target", "repeat", "fold", "Experiment_ID", "Checkpoint"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + ["y_pred", "y_std"]], b[key + ["y_pred", "y_std"]])
