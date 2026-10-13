"""ml - modelling framework for the ME 323 triboinformatics project (40-run DOE).

Reads only the ML-ready tables written by ``tribo_extract`` (``data/processed/``) or their
synthetic stand-ins (``data/synthetic/``).

Modules
-------
config      configuration loading, seeding, logging, run snapshots
data        load + clean the input tables; the four regimes; censoring
features    feature sets, derived features, leakage guards
splits      groups, gkf5 (repeated stratified group K-fold) and leave-one-alloy-out
metrics     regression / interval metrics, replicate noise floor
tuning      fixed grids, inner-CV tuning
models      model registry (GPR, TabPFN, SVR, RF, XGBoost, Ridge)
evaluate    nested CV over gkf5 + LOAO, model selection
trajectory  physics route: eq.-1 parameters -> V_w(t) on the checkpoint rows
interpret   report figures
synthetic   synthetic tables in the exact pipeline schema
report      auto-generated report.md
train       CLI entry point
archive/    excluded by the DOE (MLP, KAN, multi-task, PySR, active learning); not imported
"""
import os as _os

# torch and xgboost each ship their own OpenMP runtime (libomp). On macOS, two runtimes
# running multi-threaded in one process segfault. The framework parallelises across CV
# tasks (joblib) and runs every model single-threaded, so OpenMP is capped at one thread
# before either library can be imported. Set OMP_NUM_THREADS yourself to override.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
