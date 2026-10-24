# ml/archive — code that the current 304 SS DOE does not use

This code is not imported, not tested and not in the registry. It is kept for reference only.

| file | was |
|---|---|
| `models/mlp.py`, `models/_torch.py` | MLP seed ensemble and shared torch training code (100-run DOE) |
| `models/kan.py` | Kolmogorov-Arnold network |
| `models/multitask.py` | shared-trunk COF(t) + wear network (MLP / KAN) |
| `models/symbolic.py` | PySR symbolic regression |
| `active_learning.py` | retrospective active-learning DOE simulation (and its CLI) |
| `tests/old_test_*.py` | the 100-run test suite (prefixed `old_` so pytest does not collect it) |
| `doe40/features.py` | 40-run carbon/specimen feature sets (Carbon_pct, Hardness, Hertz, PV) |
| `doe40/splits.py` | StratifiedGroupKFold by specimen (gkf5) and leave-one-alloy-out |
| `doe40/data.py` | the four 40-run regimes, incl. trajectory parameters (eq.-1 V̇_steady, ΔV_run, τ_trans) |
| `doe40/trajectory.py` | physics route: eq.-1 parameters → V_w(t) on the checkpoint rows |
| `doe40/interpret.py` | 40-run figures (LOAO per grade, trajectory reconstructions) |
| `doe40/synthetic.py` | 40-run synthetic generator |
| `doe40/tests/doe40_test_*.py` | the 40-run test suite (prefix keeps it out of pytest collection) |

Two pieces are not in this folder:
- The "unseen operating points" holdout refit was removed during the 40-run migration and is
  only in git history (commit f628851).
- The random forest (`build_rf` in `ml/models/trees.py`) was dropped in the 304 SS migration
  (commit d590b3f has it).

These modules use package-relative imports from their old layout. Restoring one means moving
it back into `ml/` and re-registering it in `ml/models/registry.py`.
