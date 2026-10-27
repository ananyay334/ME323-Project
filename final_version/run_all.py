"""Reproduce everything: python run_all.py  (or one stage: python run_all.py --stage figures).

Stages: extract -> dataset -> models -> evaluate -> compare -> figures -> report -> readme.
Each stage reads the outputs of the earlier ones from outputs/, so a single stage can be rerun alone.
"""

from __future__ import annotations

import os

# one BLAS / OpenMP thread per process: the parallelism comes from joblib workers, which inherit this
# (without it every worker starts its own thread pool and an 8-core laptop runs ~40 busy threads)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tribo_final.config import get_paths, load_config  # noqa: E402
from tribo_final.utils import LOG, quiet_warnings, set_seeds, setup_logging  # noqa: E402

STAGES = ["extract", "dataset", "models", "evaluate", "compare", "figures", "report", "readme"]


def write_environment(paths) -> None:
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML", "joblib", "reportlab", "pytest",
            "xgboost", "tabpfn", "torch"]
    vers = {}
    for p in pkgs:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "packages": vers, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    (paths.outputs / "environment.json").write_text(json.dumps(env, indent=2))


def run_stage(name: str, cfg: dict, paths, force: bool = False) -> None:
    from tribo_final import stages

    t0 = time.time()
    LOG.info("=== stage %s ===", name)
    if name == "extract":
        stages.stage_extract(cfg, paths, force=force)
    elif name == "dataset":
        stages.stage_dataset(cfg, paths)
    elif name == "models":
        stages.stage_models(cfg, paths)
    elif name == "evaluate":
        stages.stage_evaluate(cfg, paths)
    elif name == "compare":
        from tribo_final.compare import run_all_comparisons

        run_all_comparisons(cfg, paths)
    elif name == "figures":
        from tribo_final.plots.figures import make_all_figures

        make_all_figures(cfg, paths)
    elif name == "report":
        from tribo_final.report.pdf import build_pdf
        from tribo_final.report.slides import build_slides

        build_slides(cfg, paths)
        build_pdf(cfg, paths)
    elif name == "readme":
        from tribo_final.report.readme import build_readme

        build_readme(cfg, paths)
    LOG.info("=== stage %s done in %.1f s ===", name, time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, action="append", help="run only this stage (repeatable)")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, help="run this stage and every later one")
    ap.add_argument("--config", default=None, help="config file (default: config.yaml next to this script)")
    ap.add_argument("--force-extract", action="store_true", help="ignore the per-run extraction cache")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    cold = not paths.cache.exists() or not any(paths.cache.iterdir())
    paths.make()
    setup_logging(paths.logs / "run.log")
    quiet_warnings()
    set_seeds(cfg["seed"])
    if args.stage:
        todo = [s for s in STAGES if s in args.stage]
    elif args.from_stage:
        todo = STAGES[STAGES.index(args.from_stage):]
    else:
        todo = STAGES
    LOG.info("run_all: stages %s; testdata = %s", ", ".join(todo), paths.testdata)
    if not paths.testdata.exists() and ("extract" in todo or "dataset" in todo):
        LOG.error("testdata folder not found: %s (set paths.testdata in config.yaml)", paths.testdata)
        return 1
    write_environment(paths)
    t0 = time.time()
    for s in todo:
        run_stage(s, cfg, paths, force=args.force_extract)
    minutes = (time.time() - t0) / 60
    LOG.info("run_all finished in %.1f min", minutes)
    if todo == STAGES and cold:  # a full run without any cache: record its runtime and refresh the README with it
        (paths.logs / "last_full_run.json").write_text(json.dumps({"minutes": round(minutes, 1),
                                                                    "finished": time.strftime("%Y-%m-%d %H:%M")}))
        run_stage("readme", cfg, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
