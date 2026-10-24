"""Train, cross-validate and report (304 SS DOE); ``--final`` scores the sealed test runs once.

    python -m ml.train [--config ml/configs/default.yaml] [--data synthetic|real]
                       [--targets log10_k COF_steady] [--models ann gpr ...] [--quick]
    python -m ml.train --final [--force]

Without ``--final`` only the training runs are loaded (R00-R23 + top-ups); results go to
``ml/results/<timestamp>_<data>[_quick]/``. ``--final`` additionally refits the selected
model per target on all training rows, loads the test runs (R24-R29) for the first time,
scores them once and writes ``ml/results/final/`` (synthetic: ``final_synthetic/``). It
refuses to run if that folder exists, unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, RESULTS_DIR, load_config, make_run_dir, save_snapshot, set_global_seeds, \
    setup_logging
from .data import TARGETS, load_data
from .evaluate import cross_validate, fit_full, predict_target, select_best
from .metrics import interval_metrics, regression_metrics, replicate_noise_floor
from .models.registry import get_spec, select_models
from .splits import SplitPlan

log = logging.getLogger("ml.train")


def readme_assumptions() -> list[str]:
    """One-line assumptions listed under '## Assumptions' in the project README.md."""
    path = PROJECT_ROOT / "README.md"
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
    ap = argparse.ArgumentParser(description="ME 323 ML (304 SS DOE): nested group CV, report, "
                                             "sealed test set (--final)")
    ap.add_argument("--config", default=None, help="YAML config (default ml/configs/default.yaml)")
    ap.add_argument("--data", choices=["real", "synthetic"], default=None)
    ap.add_argument("--targets", nargs="+", default=None, choices=list(TARGETS))
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these registry models")
    ap.add_argument("--quick", action="store_true",
                    help="one outer repeat, inner 3 folds, first grid value, ANN ensemble of 3")
    ap.add_argument("--n-jobs", type=int, default=None, help="parallel workers (-1 = all cores)")
    ap.add_argument("--out", default=None, help="results base folder (default ml/results)")
    ap.add_argument("--no-interpret", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="refit the selected models on all training rows and score the test runs once")
    ap.add_argument("--force", action="store_true", help="with --final: overwrite an existing final folder")
    return ap.parse_args(argv)


def final_dir(cfg: dict, base: Path, source: str) -> Path:
    return base / (cfg["final"]["dir"] + ("" if source == "real" else f"_{source}"))


def score_test(cfg: dict, source: str, train, best: pd.DataFrame, plan: SplitPlan, out: Path) -> dict:
    """Refit the selected model per target on all training rows; score the test runs once."""
    test = load_data(cfg, source, split="test")
    log.info("test set unsealed: %d rows from %d runs", len(test.frame), test.frame["Experiment_ID"].nunique())
    fitted, rows, preds = {}, [], []
    for b in best.itertuples():
        spec = get_spec(b.model)
        key = b.model if spec.multi_output else (b.model, b.target)
        if key not in fitted:
            fitted[key] = fit_full(spec, train, cfg, plan, b.target)
        est, params = fitted[key]
        mu, sd = predict_target(est, spec, test.X, train.targets, b.target)
        y = test.frame[b.target].to_numpy(float)
        cens = test.frame[f"censored_{b.target}"].to_numpy(bool)
        ok = np.isfinite(y) & ~cens
        m = regression_metrics(y[ok], mu[ok])
        if sd is not None:
            m.update(interval_metrics(y[ok], mu[ok], sd[ok]))
        rows.append({"target": b.target, "model": b.model, "params": str(params), **m})
        p = test.frame[["Experiment_ID", "Checkpoint", "Time_s"]].copy()
        p["target"], p["model"], p["y_true"], p["y_pred"] = b.target, b.model, y, mu
        p["y_std"] = sd if sd is not None else np.nan
        p["censored"] = cens
        preds.append(p)
    metrics = pd.DataFrame(rows)
    pred = pd.concat(preds, ignore_index=True)
    metrics.to_csv(out / "test_metrics.csv", index=False)
    pred.to_csv(out / "test_predictions.csv", index=False)
    from .interpret import parity_figure
    fig = parity_figure(pred, metrics.rename(columns={"R2": "R2_mean"}), out / "figures" / "parity_test.png",
                        title="test runs R24-R29 (scored once)")
    return {"metrics": metrics, "predictions": pred, "figure": fig, "n_runs": test.frame["Experiment_ID"].nunique()}


def topup_check(cfg: dict, source: str, data) -> pd.DataFrame:
    """Pre-registered top-up predictions (data/topup_candidates.csv) vs their measurements."""
    from .config import data_paths
    p = data_paths(cfg, source)["topup_candidates"]
    if not p.exists():
        return pd.DataFrame()
    cand = pd.read_csv(p, dtype={"Experiment_ID": str})
    meas = data.frame[data.frame["Experiment_ID"].isin(cand["Experiment_ID"])]
    rows = []
    for r in meas.itertuples(index=False):
        c = cand[cand["Experiment_ID"] == r.Experiment_ID].iloc[0]
        for t in data.targets:
            col = f"{t}_cp{int(r.Checkpoint)}"
            if col in c:
                rows.append({"Experiment_ID": r.Experiment_ID, "Checkpoint": int(r.Checkpoint), "target": t,
                             "predicted": c[col], "pred_sd": c.get(f"{col}_sd", np.nan),
                             "measured": getattr(r, t)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> Path:
    a = parse_args(argv)
    t_start = time.perf_counter()
    cfg = load_config(a.config, quick=a.quick)
    if a.n_jobs is not None:
        cfg["n_jobs"] = a.n_jobs
    if a.targets:
        cfg["targets"] = list(a.targets)
    source = a.data or cfg["data"]["source"]
    setup_logging()
    set_global_seeds(cfg["seed"])
    n_jobs = int(cfg["n_jobs"])
    base = Path(a.out) if a.out else RESULTS_DIR
    if a.final:
        out = final_dir(cfg, base, source)
        if out.exists():
            if not a.force:
                log.error("%s exists: the test set has already been scored. Pass --force to overwrite.", out)
                raise SystemExit(4)
            shutil.rmtree(out)
    try:
        data = load_data(cfg, source, split="train")
    except (FileNotFoundError, ValueError) as exc:
        log.error("cannot load %s data: %s", source, exc)
        raise SystemExit(2) from exc
    if a.final:
        (out / "figures").mkdir(parents=True)
    else:
        out = make_run_dir(base, f"{source}{'_quick' if a.quick else ''}")
    setup_logging(out / "run.log")
    for n in data.notes:
        log.info("data: %s", n)
    save_snapshot(cfg, out, ["python", "-m", "ml.train"] + (list(argv) if argv is not None else sys.argv[1:]))
    log.info("results -> %s", out)

    specs, skipped = select_models(cfg, a.models)
    plan = SplitPlan(data.groups, cfg)
    log.info("targets %s | models %s | %d outer repeat(s) (seeds %s)", data.targets,
             [s.name for s in specs], len(plan.seeds), plan.seeds)
    cv = cross_validate(data, specs, cfg, plan, n_jobs, "CV")
    summary = cv.summary()
    if summary.empty:
        log.error("no model produced results (failures: %d) - see run.log", len(cv.failures))
        pd.DataFrame(cv.failures).to_csv(out / "failures.csv", index=False)
        raise SystemExit(3)
    best = select_best(summary)
    noise = {t: replicate_noise_floor(data.frame, t, cfg["data"]["replicates"]) for t in data.targets}

    summary.to_csv(out / "metrics.csv", index=False)
    cv.fold_metrics.to_csv(out / "metrics_folds.csv", index=False)
    cv.predictions.to_csv(out / "predictions.csv", index=False)
    cv.params.to_csv(out / "hyperparameters.csv", index=False)
    best.to_csv(out / "best_models.csv", index=False)
    if len(cv.importance):
        cv.importance.to_csv(out / "importance_folds.csv", index=False)
    if cv.failures:
        pd.DataFrame(cv.failures).to_csv(out / "failures.csv", index=False)

    interp = None
    if cfg["interpret"].get("enabled", True) and not a.no_interpret:
        from .interpret import run_interpretation
        interp = run_interpretation(data, cfg, cv, summary, best, plan, out, source)
    final = score_test(cfg, source, data, best, plan, out) if a.final else None

    from .report import write_report
    seconds = time.perf_counter() - t_start
    report = write_report(out, {
        "cfg": cfg, "data": data, "summary": summary, "best": best, "plan": plan,
        "models_run": [s.name for s in specs], "skipped": skipped, "failures": cv.failures,
        "noise": noise, "interp": interp, "final": final, "topup": topup_check(cfg, source, data),
        "assumptions": readme_assumptions(), "seconds": seconds})
    log.info("selected models (lowest CV RMSE, Ridge excluded):\n%s", best.to_string(index=False))
    log.info("done in %.1f min -> %s", seconds / 60, report)
    return out


if __name__ == "__main__":
    main()
