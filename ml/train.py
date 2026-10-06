"""Train, evaluate and report everything in one command.

    python -m ml.train --config ml/configs/default.yaml [--data synthetic|real]
                       [--targets wear cof] [--models gpr kan ...] [--quick]

Writes ``ml/results/<timestamp>_<data>[_quick]/`` with metrics.csv, predictions.csv,
figures/, models/, the config snapshot and report.md. Switching between synthetic and
real data needs no code changes: ``--data real`` reads ``data/processed/``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .config import (load_config, make_run_dir, save_snapshot, set_global_seeds, setup_logging)
from .data import TARGETS, load_data
from .evaluate import (JOINT_COF, JOINT_WEAR, CVResult, cross_validate, cross_validate_joint,
                       final_refit, make_design, make_joint_design, select_final)
from .features import check_physics_consistency
from .models.registry import select_models
from .splits import plan_from_data

log = logging.getLogger("ml.train")


def resolve_targets(cfg: dict, requested: list[str] | None) -> list[str]:
    """Families (``wear``, ``cof``) and/or target names -> ordered list of target names."""
    fam = cfg["targets"]
    if not requested:
        requested = list(fam)
    out = []
    for r in requested:
        if r in fam:
            out += fam[r]
        elif r in TARGETS and TARGETS[r].family != "trajectory":
            out.append(r)
        else:
            raise SystemExit(f"unknown target or family {r!r}; families: {list(fam)}, "
                             f"targets: {[t for t in TARGETS if TARGETS[t].family != 'trajectory']}")
    return list(dict.fromkeys(out))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ME 323 ML framework: nested group CV, final test, "
                                             "interpretation, active learning, trajectory, report")
    ap.add_argument("--config", default=None, help="YAML config (default ml/configs/default.yaml)")
    ap.add_argument("--data", choices=["real", "synthetic"], default=None,
                    help="data source (default: data.source in the config)")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="target families (wear, cof) and/or names (log10_k, COF_ss_mean, ...)")
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these registry models")
    ap.add_argument("--quick", action="store_true", help="reduced search: smoke test (< 5 min)")
    ap.add_argument("--n-jobs", type=int, default=None, help="parallel workers (-1 = all cores)")
    ap.add_argument("--out", default=None, help="results base folder (default ml/results)")
    ap.add_argument("--no-interpret", action="store_true")
    ap.add_argument("--no-al", action="store_true", help="skip the active-learning simulation")
    ap.add_argument("--no-trajectory", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    a = parse_args(argv)
    t_start = time.perf_counter()
    cfg = load_config(a.config, quick=a.quick)
    if a.n_jobs is not None:
        cfg["n_jobs"] = a.n_jobs
    source = a.data or cfg["data"]["source"]
    cfg["data"]["source_used"] = source
    setup_logging()
    set_global_seeds(cfg["seed"])
    n_jobs = int(cfg["n_jobs"])

    # ---- data (before creating the results folder) ---------------------------
    try:
        data = load_data(cfg, source)
    except (FileNotFoundError, ValueError) as exc:
        log.error("cannot load %s data: %s", source, exc)
        raise SystemExit(2) from exc
    out = make_run_dir(Path(a.out) if a.out else None, f"{source}{'_quick' if a.quick else ''}")
    setup_logging(out / "run.log")
    for n in data.notes:
        log.info("data: %s", n)
    save_snapshot(cfg, out, (["python", "-m", "ml.train"] + list(argv)) if argv is not None
                  else ["python", "-m", "ml.train"] + sys.argv[1:])
    log.info("results -> %s", out)
    for msg in check_physics_consistency(data.runs):
        data.note(msg, logging.WARNING)
    targets = resolve_targets(cfg, a.targets)
    specs, skipped = select_models(cfg, a.models)
    single = [s for s in specs if s.regimes != ("joint",)]
    joint = [s for s in specs if s.regimes == ("joint",)]
    plan = plan_from_data(data.cv_runs, cfg)
    main_mode = cfg["censoring"]["mode"]
    fsets = cfg["features"]["use"]
    log.info("targets %s | models %s | feature sets %s", targets, [s.name for s in specs], fsets)

    # ---- nested group CV --------------------------------------------------
    designs, target_info = [], {}
    for t in targets:
        for fs in fsets:
            d = make_design(data, t, fs, cfg, main_mode)
            if d is None:
                log.warning("target %s: no usable rows -> skipped", t)
                break
            designs.append(d)
            target_info[t] = d.info
    cv = cross_validate(designs, single, cfg, plan, n_jobs, "CV")
    jt = tuple(t for t in (JOINT_WEAR, JOINT_COF) if t in targets)
    if joint and jt:
        jds = [j for j in (make_joint_design(data, fs, cfg, main_mode) for fs in fsets) if j is not None]
        cv = cv.extend(cross_validate_joint(jds, joint, cfg, plan, n_jobs, jt))
    sens_modes = [m for m in cfg["censoring"].get("sensitivity", []) if m != main_mode]
    sens_names = set(cfg["censoring"].get("sensitivity_models", []))
    sens_specs = [s for s in single if s.name in sens_names]
    for mode in sens_modes:
        sd = [make_design(data, t, fs, cfg, mode) for t in targets if TARGETS[t].censorable
              for fs in fsets]
        sd = [d for d in sd if d is not None]
        if sd and sens_specs:
            cv = cv.extend(cross_validate(sd, sens_specs, cfg, plan, n_jobs, f"CV censoring={mode}"))
    summary = cv.summary()
    if summary.empty:
        log.error("no model produced results (models: %s; failures: %d) - see run.log",
                  [s.name for s in specs], len(cv.failures))
        pd.DataFrame(cv.failures).to_csv(out / "failures.csv", index=False)
        raise SystemExit(3)
    summary.to_csv(out / "metrics.csv", index=False)
    cv.fold_metrics.to_csv(out / "metrics_folds.csv", index=False)
    cv.predictions.to_csv(out / "predictions.csv", index=False)
    if len(cv.predictions_time):        # ~1e6 rows: fast gzip, 6 significant digits
        cv.predictions_time.to_csv(out / "predictions_timeseries.csv.gz", index=False,
                                   float_format="%.6g",
                                   compression={"method": "gzip", "compresslevel": 1})
    if len(cv.params):
        cv.params.to_csv(out / "hyperparameters.csv", index=False)

    # ---- final refit + unseen operating points ------------------------------
    sel = select_final(summary, cfg, main_mode)
    sel.to_csv(out / "final_selection.csv", index=False)
    final = final_refit(data, cfg, plan, sel, cv, out / "models", n_jobs, main_mode)
    if len(final.predictions):
        final.predictions.to_csv(out / "holdout_predictions.csv", index=False)
    if len(final.predictions_time):
        final.predictions_time.to_csv(out / "holdout_predictions_timeseries.csv.gz", index=False,
                                      float_format="%.6g",
                                      compression={"method": "gzip", "compresslevel": 1})
    if len(final.metrics):
        final.metrics.to_csv(out / "holdout_metrics.csv", index=False)
    failures = cv.failures + final.failures

    # ---- interpretation / extensions ----------------------------------------
    interp = al = traj = None
    if cfg["interpret"].get("enabled", True) and not a.no_interpret:
        from .interpret import run_interpretation
        interp = run_interpretation(data, cfg, cv, summary, final, out)
    if cfg["trajectory"].get("enabled", True) and not a.no_trajectory:
        from .trajectory import run_trajectory
        traj = run_trajectory(data, cfg, plan, out / "trajectory", n_jobs, a.models)
        if traj and "cv" in traj:
            failures += traj["cv"].failures
    if cfg["active_learning"].get("enabled", True) and not a.no_al:
        from .active_learning import run_active_learning
        al = run_active_learning(data, cfg, out / "active_learning", n_jobs, out / "figures")
    if failures:
        pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)

    # ---- report ---------------------------------------------------------------
    from .report import write_report
    seconds = time.perf_counter() - t_start
    report = write_report(out, {
        "cfg": cfg, "data": data, "summary": summary, "cv_predictions": cv.predictions,
        "targets": targets, "target_info": target_info, "models_run": [s.name for s in specs],
        "skipped": skipped, "failures": failures, "final": final, "interp": interp, "al": al,
        "trajectory": traj, "seconds": seconds})
    best = (summary[summary["censor_mode"] == main_mode].sort_values("RMSE_mean")
            .groupby("target").head(1)[["target", "model", "feature_set", "R2_mean", "RMSE_mean"]])
    log.info("best models by outer-fold RMSE:\n%s", best.to_string(index=False))
    log.info("done in %.1f min -> %s", seconds / 60, report)
    return out


if __name__ == "__main__":
    main()
