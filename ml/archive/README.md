# ml/archive — excluded by the 40-run DOE (not imported, not tested, not in the registry)

Kept for reference only. The new DOE excludes deep ANNs/MLPs, and the 40-run plan has no
active-learning or symbolic-regression step.

| file | was |
|---|---|
| `models/mlp.py`, `models/_torch.py` | MLP seed ensemble and shared torch training code |
| `models/kan.py` | Kolmogorov-Arnold network |
| `models/multitask.py` | shared-trunk COF(t) + wear network (MLP / KAN) |
| `models/symbolic.py` | PySR symbolic regression |
| `active_learning.py` | retrospective active-learning DOE simulation (and its CLI) |
| `tests/old_test_*.py` | the 100-run test suite (prefixed `old_` so pytest does not collect it) |

These modules use package-relative imports from the old layout; restoring one means
moving it back into `ml/` and re-registering it in `ml/models/registry.py`.
