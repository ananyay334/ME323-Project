"""Extension A: retrospective simulation of an active-learning-guided DOE.

    python -m ml.active_learning --config ml/configs/default.yaml [--data synthetic|real] [--quick]

Candidate pool = the grid conditions of the CV runs (replicates of a condition are
averaged; censored or missing values are excluded under the main censoring mode).
Each repetition:

1. seed with ``n_seed`` space-filling conditions: many Latin-hypercube draws on the
   min-max-scaled design variables, each snapped greedily to the nearest unused pool
   condition, keeping the draw with the largest minimum pairwise distance (maximin);
2. fit the surrogate (GPR, anisotropic Matérn-5/2 + white noise, normalize_y) on the
   acquired conditions and score it on the conditions *not yet acquired*;
3. acquire one condition: ``variance`` = argmax latent sigma(x); ``ucb`` = argmax
   mu + kappa sigma; ``random`` = uniform (baseline);
4. stop at ``budget`` or when max latent sigma over the remaining pool < ``sigma_stop``
   (the stopping point is recorded; the curve is continued to the budget so that
   strategies can be compared at equal N).

The reference is the full-grid model: group 5-fold CV of the same GPR over the whole
pool. ``N_AL`` = the first N at which the mean error over repetitions is within
``tolerance`` of that reference (per repetition: the first N from which the error
stays within it). The latent sigma excludes the fitted white noise:
re-measuring a condition cannot remove noise, so it should not drive acquisition.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import plotting as P
from .config import (load_config, make_run_dir, save_snapshot, set_global_seeds, setup_logging,
                     stable_seed)
from .data import TARGETS, TriboData, load_data, target_frame
from .evaluate import run_parallel
from .models.gpr import GPRModel
from .splits import assign_group_folds

log = logging.getLogger("ml.active_learning")


def condition_pool(data: TriboData, target: str, cfg: dict, split: str = "cv") -> pd.DataFrame:
    """One row per condition (replicate group) with the design variables and mean target."""
    feats = cfg["active_learning"]["features"]
    df, _ = target_frame(data, target, cfg["censoring"]["mode"], split)
    if df.empty:
        return df
    df = df[~df["censored"].astype(bool)]
    if split == "cv":
        grid = cfg["data"].get("grid") or {}
        on = np.ones(len(df), bool)
        for col, levels in grid.items():
            if col in df:
                on &= np.isclose(df[col].to_numpy(float)[:, None], np.asarray(levels, float)).any(1)
        df = df[on]
    pool = df.groupby("group").agg({**{f: "first" for f in feats}, "y": "mean"}).reset_index()
    return pool.sort_values(feats).reset_index(drop=True)


def space_filling_seed(Xs: np.ndarray, n: int, rng: np.random.Generator,
                       n_candidates: int = 50) -> np.ndarray:
    """Maximin Latin-hypercube design snapped to distinct pool points."""
    best, best_score = None, -np.inf
    n = min(n, len(Xs))
    for _ in range(n_candidates):
        lhs = qmc.LatinHypercube(d=Xs.shape[1], seed=rng).random(n)
        free = np.ones(len(Xs), bool)
        chosen = []
        for p in lhs:
            d = np.where(free, ((Xs - p) ** 2).sum(1), np.inf)
            j = int(np.argmin(d))
            chosen.append(j)
            free[j] = False
        C = Xs[chosen]
        dist = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))
        score = dist[np.triu_indices(len(C), 1)].min() if len(C) > 1 else 0.0
        if score > best_score:
            best, best_score = np.array(chosen), score
    return best


def _scores(y, mu) -> tuple[float, float]:
    if len(y) == 0:
        return np.nan, np.nan
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    ss = float(((y - y.mean()) ** 2).sum())
    return rmse, (1 - float(((y - mu) ** 2).sum()) / ss) if ss > 0 and len(y) > 2 else np.nan


def simulate(Xs: np.ndarray, y: np.ndarray, strategy: str, rep: int, acfg: dict, sigma_stop: float,
             base_seed: int, X_ho: np.ndarray | None = None, y_ho: np.ndarray | None = None) -> dict:
    """One active-learning (or random) run; returns the per-step learning curve."""
    rng = np.random.default_rng(stable_seed(base_seed, "al", rep))
    n_seed, budget = int(acfg["n_seed"]), min(int(acfg["budget"]), len(Xs))
    if strategy == "random" and acfg.get("random_seed_design", "shared") == "random":
        acquired = list(rng.choice(len(Xs), size=n_seed, replace=False))
    else:
        acquired = list(space_filling_seed(Xs, n_seed, rng, int(acfg.get("lhs_candidates", 50))))
    pick_rng = np.random.default_rng(stable_seed(base_seed, "pick", strategy, rep))
    kernel, rows, stop_n = None, [], None
    cols = [f"x{i}" for i in range(Xs.shape[1])]
    while True:
        gp = GPRModel(n_restarts_optimizer=int(acfg.get("gpr_restarts", 1)) if kernel is None else 0,
                      random_state=stable_seed(base_seed, rep, len(acquired)), kernel=kernel)
        gp.fit(pd.DataFrame(Xs[acquired], columns=cols), y[acquired])
        kernel = gp.gp_.kernel_
        rem = np.setdiff1d(np.arange(len(Xs)), acquired)
        mu_r = gp.predict(pd.DataFrame(Xs[rem], columns=cols)) if len(rem) else np.array([])
        lat_r = gp.predict_latent_std(pd.DataFrame(Xs[rem], columns=cols)) if len(rem) else np.array([])
        rmse, r2 = _scores(y[rem], mu_r)
        row = {"strategy": strategy, "rep": rep, "N": len(acquired), "rmse_remaining": rmse,
               "r2_remaining": r2, "max_latent_sigma": float(lat_r.max()) if len(rem) else 0.0,
               "noise_sd": float(np.sqrt(gp.noise_var_))}
        if X_ho is not None and len(X_ho):
            row["rmse_holdout"] = _scores(y_ho, gp.predict(pd.DataFrame(X_ho, columns=cols)))[0]
        rows.append(row)
        if stop_n is None and len(rem) and row["max_latent_sigma"] < sigma_stop:
            stop_n = len(acquired)
        if len(acquired) >= budget or not len(rem):
            break
        if strategy == "variance":
            nxt = rem[int(np.argmax(lat_r))]
        elif strategy == "ucb":
            nxt = rem[int(np.argmax(mu_r + float(acfg.get("kappa", 2.0)) * lat_r))]
        elif strategy == "random":
            nxt = int(pick_rng.choice(rem))
        else:
            raise ValueError(f"unknown acquisition strategy {strategy!r}")
        acquired.append(int(nxt))
    return {"curve": pd.DataFrame(rows), "stop_N": stop_n if stop_n is not None else np.nan,
            "strategy": strategy, "rep": rep, "order": acquired}


def full_grid_reference(Xs: np.ndarray, y: np.ndarray, seed: int, n_folds: int = 5,
                        X_ho=None, y_ho=None) -> dict:
    """Group K-fold CV RMSE of the GPR trained on (folds of) the whole pool."""
    cols = [f"x{i}" for i in range(Xs.shape[1])]
    fm = assign_group_folds(np.arange(len(Xs)).astype(str), n_folds, stable_seed(seed, "al_ref"))
    fold = np.array([fm[str(i)] for i in range(len(Xs))])
    pred = np.empty(len(Xs))
    for f in range(n_folds):
        te = fold == f
        gp = GPRModel(n_restarts_optimizer=2, random_state=seed).fit(
            pd.DataFrame(Xs[~te], columns=cols), y[~te])
        pred[te] = gp.predict(pd.DataFrame(Xs[te], columns=cols))
    out = {"rmse_full_cv": _scores(y, pred)[0], "r2_full_cv": _scores(y, pred)[1], "n_pool": len(Xs)}
    if X_ho is not None and len(X_ho):
        gp = GPRModel(n_restarts_optimizer=2, random_state=seed).fit(pd.DataFrame(Xs, columns=cols), y)
        out["rmse_full_holdout"] = _scores(y_ho, gp.predict(pd.DataFrame(X_ho, columns=cols)))[0]
    return out


def _sim_job(Xs, y, strategy, rep, acfg, sigma_stop, seed, X_ho, y_ho):
    return simulate(Xs, y, strategy, rep, acfg, sigma_stop, seed, X_ho, y_ho)


def summarize_al(curves: pd.DataFrame, stops: pd.DataFrame, ref: dict, tol: float) -> pd.DataFrame:
    """N_AL per strategy: from the mean curve and per repetition (median)."""
    thr = ref["rmse_full_cv"] * (1 + tol)
    rows = []
    for s, c in curves.groupby("strategy"):
        mean = c.groupby("N")["rmse_remaining"].mean()
        hit = mean[mean <= thr]
        per = []                       # first N from which the curve *stays* below threshold
        for _, cr in c.groupby("rep"):
            above = cr.sort_values("N")["rmse_remaining"].to_numpy() > thr
            Ns = cr.sort_values("N")["N"].to_numpy()
            last_above = np.flatnonzero(above)
            if not len(last_above):
                per.append(float(Ns[0]))
            elif last_above[-1] + 1 < len(Ns):
                per.append(float(Ns[last_above[-1] + 1]))
            else:
                per.append(np.nan)
        st = stops[stops["strategy"] == s]["stop_N"]
        last = c[c["N"] == c["N"].max()]
        rows.append({"strategy": s, "N_AL_mean_curve": float(hit.index.min()) if len(hit) else np.nan,
                     "N_AL_sustained_median_rep": (float(np.nanmedian(per)) if np.isfinite(per).any()
                                               else np.nan),
                     "frac_reps_reaching": float(np.mean(np.isfinite(per))),
                     "N_stop_median": float(np.nanmedian(st)) if np.isfinite(st).any() else np.nan,
                     "rmse_at_budget": float(last["rmse_remaining"].mean()),
                     "rmse_threshold": thr, "rmse_full_cv": ref["rmse_full_cv"],
                     "N_budget": int(c["N"].max())})
    return pd.DataFrame(rows)


@P.styled
def learning_curve_figure(curves: pd.DataFrame, ref: dict, tol: float, target: str, out: Path):
    fig, ax = P.new_fig(w=5.6, h=3.8)
    a = ax[0, 0]
    for s, c in curves.groupby("strategy"):
        g = c.groupby("N")["rmse_remaining"]
        m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        col = P.SLOTS[P.STRATEGY_SLOT.get(s, 3)]
        a.fill_between(m.index, lo, hi, color=col, alpha=0.1, lw=0)
        a.plot(m.index, m, color=col, label=s)
    a.axhline(ref["rmse_full_cv"], color=P.INK2, lw=1)
    a.axhspan(ref["rmse_full_cv"], ref["rmse_full_cv"] * (1 + tol), color=P.GRID, alpha=0.6, lw=0)
    a.text(a.get_xlim()[1], ref["rmse_full_cv"], f" full grid ({ref['n_pool']}), +{100 * tol:.0f} %",
           va="bottom", ha="right", color=P.INK2, fontsize=8)
    a.set_xlabel("conditions acquired (N)")
    a.set_ylabel("RMSE, not-yet-acquired conditions")
    a.set_title(f"Active learning, {TARGETS[target].label}\nmean and 10-90 % band over "
                f"{curves['rep'].nunique()} repetitions", fontsize=9.5)
    a.legend(title="acquisition", loc="upper right")
    return P.save(fig, out)


def run_active_learning(data: TriboData, cfg: dict, out_dir: Path, n_jobs: int = 1,
                        fig_dir: Path | None = None) -> dict:
    """Run the simulation for every configured target; CSVs to ``out_dir``, figures to ``fig_dir``."""
    acfg = cfg["active_learning"]
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir) if fig_dir else out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"targets": {}}
    feats = acfg["features"]
    for target in acfg["targets"]:
        pool = condition_pool(data, target, cfg, "cv")
        if len(pool) < int(acfg["n_seed"]) + 5:
            log.warning("active learning %s skipped: only %d pool conditions", target, len(pool))
            continue
        lo, hi = pool[feats].min().to_numpy(float), pool[feats].max().to_numpy(float)
        span = np.where(hi > lo, hi - lo, 1.0)
        Xs = (pool[feats].to_numpy(float) - lo) / span
        y = pool["y"].to_numpy(float)
        ho = condition_pool(data, target, cfg, "holdout")
        X_ho = (ho[feats].to_numpy(float) - lo) / span if len(ho) else None
        y_ho = ho["y"].to_numpy(float) if len(ho) else None
        ref = full_grid_reference(Xs, y, cfg["seed"], 5, X_ho, y_ho)
        sig = acfg.get("sigma_stop", {})
        sigma_stop = float(sig.get(target, 0.0) if isinstance(sig, dict) else sig)
        jobs = [(Xs, y, s, r, acfg, sigma_stop, stable_seed(cfg["seed"], target), X_ho, y_ho)
                for s in acfg["strategies"] for r in range(int(acfg["repeats"]))]
        log.info("active learning %s: pool %d conditions, %d simulations", target, len(pool), len(jobs))
        sims = run_parallel(_sim_job, jobs, n_jobs, f"AL {target}")
        curves = pd.concat([s["curve"] for s in sims], ignore_index=True).sort_values(
            ["strategy", "rep", "N"])
        stops = pd.DataFrame([{"strategy": s["strategy"], "rep": s["rep"], "stop_N": s["stop_N"]}
                              for s in sims])
        tol = float(acfg.get("tolerance", 0.1))
        summ = summarize_al(curves, stops, ref, tol).assign(target=target, n_pool=len(pool),
                                                            sigma_stop=sigma_stop)
        curves.assign(target=target).to_csv(out_dir / f"al_curves_{target}.csv", index=False)
        fig = learning_curve_figure(curves, ref, tol, target,
                                    fig_dir / f"al_learning_curves_{target}.png")
        results["targets"][target] = {"summary": summ, "reference": ref, "figure": fig,
                                      "n_pool": len(pool)}
    if results["targets"]:
        pd.concat([v["summary"] for v in results["targets"].values()]).to_csv(
            out_dir / "al_summary.csv", index=False)
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Retrospective active-learning DOE simulation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", choices=["real", "synthetic"], default=None)
    ap.add_argument("--quick", action="store_true", help="few repetitions, smaller budget")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--out", default=None, help="results base folder (default ml/results)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, quick=a.quick)
    source = a.data or cfg["data"]["source"]
    out = make_run_dir(Path(a.out) if a.out else None, f"al_{source}{'_quick' if a.quick else ''}")
    setup_logging(out / "run.log")
    save_snapshot(cfg, out, sys.argv if argv is None else argv)
    set_global_seeds(cfg["seed"])
    t0 = time.perf_counter()
    data = load_data(cfg, source)
    res = run_active_learning(data, cfg, out, a.n_jobs if a.n_jobs is not None else cfg["n_jobs"],
                              out / "figures")
    from .report import active_learning_section
    (out / "report.md").write_text("# Active-learning DOE simulation\n\n"
                                   + active_learning_section(res, cfg, out), encoding="utf-8")
    log.info("active learning done in %.0f s -> %s", time.perf_counter() - t0, out)


if __name__ == "__main__":
    main()
