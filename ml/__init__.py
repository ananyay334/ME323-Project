"""ml - modelling framework for the ME 323 triboinformatics project.

Reads only the ML-ready tables written by ``tribo_extract`` (``data/processed/``) or
their synthetic stand-ins (``data/synthetic/``), and trains, evaluates and reports
models for steady-state COF, COF(t) and the specific wear rate.

Modules
-------
config           configuration loading, seeding, logging, run snapshots
data             load + clean the two input tables, censoring, holdout
features         feature sets, derived features, leakage guards
splits           group definitions and deterministic (nested) group K-fold
metrics          regression / interval metrics, run-level aggregation of COF(t)
models           model registry and model implementations
evaluate         nested group CV, joint (multi-task) CV, final refit + holdout test
interpret        SHAP, partial dependence, KAN functions, GPR maps, calibration
active_learning  retrospective active-learning DOE simulation (Extension A)
trajectory       time-resolved wear trajectory parameters (Extension B)
synthetic        synthetic tables in the exact pipeline schema
report           auto-generated report.md
train            CLI entry point
"""
import os as _os

# torch and xgboost each ship their own OpenMP runtime (libomp). On macOS, two runtimes
# running multi-threaded in one process segfault. The framework parallelises across CV
# tasks (joblib) and runs every model single-threaded, so OpenMP is capped at one thread
# before either library can be imported. Set OMP_NUM_THREADS yourself to override.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
