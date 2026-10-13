"""Train, evaluate and report everything in one command (40-run DOE).

    python -m ml.train --config ml/configs/default.yaml [--data synthetic|real]
                       [--targets endpoint ...] [--models gpr svr ...] [--quick]

Writes ``ml/results/<timestamp>_<data>[_quick]/`` with metrics.csv, predictions.csv,
figures/, the config snapshot and report.md. Switching between synthetic and real data
needs no code changes: ``--data real`` reads ``data/processed/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import ML_DIR, load_config, make_run_dir, save_snapshot, set_global_seeds, setup_logging
from .data import TARGETS, load_data, target_frame
from .evaluate import cross_validate, make_design, select_best
from .features import check_physics_consistency
from .metrics import replicate_noise_floor
from .models.registry import select_models
from .splits import SplitPlan

log = logging.getLogger("ml.train")


def resolve_targets(cfg: dict, requested: list[str] | None) -> list[str]:
    """Regimes and/or target names -> ordered list of target names."""
    reg = cfg["regimes"]
    out = []
    for r in requested or list(reg):
        if r in reg:
            out += reg[r]
        elif r in TARGETS:
            out.append(r)
        else:
            raise SystemExit(f"unknown regime or target {r!r}; regimes: {list(reg)}, "
                             f"targets: {list(TARGETS)}")
    return list(dict.fromkeys(out))


def readme_assumptions() -> list[str]:
    """One-line assumptions listed under '## Assumptions' in ml/README.md."""
    path = ML_DIR / "README.md"
    if not path.exists():
        return []
    lines, on = [], False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("## "):
            on = ln.strip().lower() == "## assumptions"
            continue
        if on and ln.startswith("- "):
            lines.append(ln[2:].strip())
    return lines


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ME 323 ML framework (40-run DOE): nested group CV "
                                             "(gkf5 + LOAO), physics route, report")
    ap.add_argument("--config", default=None, help="YAML config (default ml/configs/default.yaml)")
    ap.add_argument("--data", choices=["real", "synthetic"], default=None)
    ap.add_argument("--targets", nargs="+", default=None, help="regimes and/or target names")
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these registry models")
    ap.add_argument("--quick", action="store_true", help="one gkf5 repeat, inner 3 folds, first grid value")
    ap.add_argument("--n-jobs", type=int, default=None, help="parallel workers (-1 = all cores)")
    ap.add_argument("--out", default=None, help="results base folder (default ml/results)")
    ap.add_argument("--no-interpret", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    a = parse_args(argv)
    t_start = time.perf_counter()
    cfg = load_config(a.config, quick=a.quick)
    if a.n_jobs is not None:
        cfg["n_jobs"] = a.n_jobs
    source = a.data or cfg["data"]["source"]
    setup_logging()
    set_global_seeds(cfg["seed"])
    n_jobs = int(cfg["n_jobs"])
    try:
        data = load_data(cfg, source)
    except (FileNotFoundError, ValueError) as exc:
        log.error("cannot load %s data: %s", source, exc)
        raise SystemExit(2) from exc
    out = make_run_dir(Path(a.out) if a.out else None, f"{source}{'_quick' if a.quick else ''}")
    setup_logging(out / "run.log")
    for n in data.notes:
        log.info("data: %s", n)
    save_snapshot(cfg, out, ["python", "-m", "ml.train"] + (list(argv) if argv is not None
                                                            else sys.argv[1:]))
    log.info("results -> %s", out)
    for msg in check_physics_consistency(data.runs):
        data.note(msg, logging.WARNING)

    targets = resolve_targets(cfg, a.targets)
    specs, skipped = select_models(cfg, a.models)
    plan = SplitPlan(data.runs, cfg)
    log.info("gkf5 seeds %s (offsets %s); LOAO folds %s", plan.seeds, plan.offsets, plan.loao_labels)
    mode = cfg["censoring"]["mode"]
    designs, target_info = [], {}
    for t in targets:
        for fs in cfg["features"]["use"]:
            d = make_design(data, t, fs, cfg, mode)
            if d is None:
                log.warning("target %s: no usable rows -> skipped", t)
                break
            designs.append(d)
            target_info[t] = d.info
    log.info("targets %s | models %s | feature sets %s", list(target_info),
             [s.name for s in specs], cfg["features"]["use"])
    cv = cross_validate(designs, specs, cfg, plan, n_jobs, "CV")

    route_params, route_choices = pd.DataFrame(), None
    if cfg.get("physics_route", {}).get("enabled", True) and "log10_V_t" in target_info:
        from .trajectory import physics_route
        route, route_params = physics_route(data, cfg, plan, specs, n_jobs, mode)
        cv = cv.extend(route)
        if len(route.params):
            cnt: dict = {}
            for js in route.params["params"]:
                for p, v in json.loads(js).items():
                    cnt.setdefault(p, Counter())[v["model"]] += 1
            route_choices = {p: dict(c) for p, c in cnt.items()}
    summary = cv.summary()
    if summary.empty:
        log.error("no model produced results (failures: %d) - see run.log", len(cv.failures))
        pd.DataFrame(cv.failures).to_csv(out / "failures.csv", index=False)
        raise SystemExit(3)
    best = select_best(summary)
    noise = {}
    for t in cfg["regimes"]["endpoint"]:
        if t in target_info:
            noise[t] = replicate_noise_floor(target_frame(data, t, mode)[0], cfg["data"]["replicate_pairs"])

    summary.to_csv(out / "metrics.csv", index=False)
    cv.fold_metrics.to_csv(out / "metrics_folds.csv", index=False)
    cv.predictions.to_csv(out / "predictions.csv", index=False)
    cv.params.to_csv(out / "hyperparameters.csv", index=False)
    best.to_csv(out / "best_models.csv", index=False)
    if len(route_params):
        route_params.to_csv(out / "physics_route_params.csv", index=False)
    if cv.failures:
        pd.DataFrame(cv.failures).to_csv(out / "failures.csv", index=False)

    interp = None
    if cfg["interpret"].get("enabled", True) and not a.no_interpret:
        from .interpret import run_interpretation
        interp = run_interpretation(data, cfg, cv.predictions, summary, best, route_params, out)

    from .report import write_report
    seconds = time.perf_counter() - t_start
    report = write_report(out, {
        "cfg": cfg, "data": data, "summary": summary, "best": best, "plan": plan,
        "target_info": target_info, "models_run": [s.name for s in specs], "skipped": skipped,
        "failures": cv.failures, "noise": noise, "interp": interp, "route_choices": route_choices,
        "assumptions": readme_assumptions(), "seconds": seconds})
    log.info("selected models (lowest gkf5 RMSE, Ridge excluded):\n%s",
             best.to_string(index=False))
    log.info("done in %.1f min -> %s", seconds / 60, report)
    return out


if __name__ == "__main__":
    main()
