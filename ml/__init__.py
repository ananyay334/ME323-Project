"""ml - modelling framework for the ME 323 project (304 stainless steel, dry reciprocating).

Reads only ``runs_checkpoints.csv`` (+ ``cof_cycles.csv`` for one figure) written by
``tribo_extract`` (``data/processed/``) or the synthetic stand-in (``data/synthetic/``).

Modules
-------
config      configuration loading, seeding, logging, run snapshots
data        load the run x checkpoint rows (train / sealed test), inputs, targets, censoring
splits      run groups, 5-fold GroupKFold (groups shuffled by seed), inner GroupKFold
metrics     regression / interval metrics, replicate noise floor
tuning      fixed grids, inner-CV tuning
models      model registry (ANN ensemble, GPR, SVR, TabPFN, XGBoost, Ridge)
evaluate    nested CV, group-permutation importance, model selection, full refits
interpret   figures, GPR length scales, Ridge exponents
synthetic   synthetic runs_checkpoints.csv in the exact pipeline schema
report      auto-generated report.md
train       CLI entry point (``--final`` scores the test runs once)
topup       top-up run selection (GPR sigma + ANN ensemble spread)
archive/    excluded by the DOE (not imported, not tested, not registered)
"""
import os as _os

# torch and xgboost each ship their own OpenMP runtime (libomp). On macOS, two runtimes
# running multi-threaded in one process segfault. The framework parallelises across CV
# tasks (joblib) and runs every model single-threaded, so OpenMP is capped at one thread
# before either library can be imported. Set OMP_NUM_THREADS yourself to override.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
