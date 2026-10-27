"""ME 323 (IIT Bombay) tribology project: COF and wear of 304 stainless steel, final pipeline."""

import multiprocessing.resource_tracker as _resource_tracker

__version__ = "1.0.0"

# Python 3.13 added ResourceTracker.__del__, which fails with a harmless "ChildProcessError: No child processes"
# traceback when each joblib (loky 1.4) worker exits. Every worker imports this package, so the no-op below keeps
# the console clean; it changes no results.
if hasattr(_resource_tracker.ResourceTracker, "__del__"):
    _resource_tracker.ResourceTracker.__del__ = lambda self: None
