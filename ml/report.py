"""Auto-generated ``report.md`` for a results folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COF_T_RUN, TARGETS

ASSUMPTIONS = [
    "Model selection uses only held-out outer-fold (group K-fold) error; hyperparameters are "
    "chosen in the inner loop; nothing is selected on training error.",
    "Groups = `Replicate_of` (followed to the root of a chain) else `Experiment_ID`; folds are "
    "assigned per group and stratified by carbon level, identically for every target and model.",
    "Censored wear (`Wear_detected == False`) is excluded by default; the LOD/2 sensitivity run "
    "is scored on the detected runs only, so both modes are compared on the same runs.",
    "COF(t) models are fitted on a thinned set of seconds per run (half log-, half "
    "evenly spaced) and scored on every second; `COF_t@ss` averages the predicted curve over "
    "the extraction's steady phase.",
    "Predictive std: GPR = posterior predictive incl. fitted noise; MLP/KAN/multi-task = "
    "ensemble spread + mean validation MSE. The run-level `COF_t@ss` std is the mean per-second "
    "std (conservative). Holdout intervals without a model std use the 95 % quantile of that "
    "model's absolute out-of-fold errors.",
    "Partial dependence / maps set carbon, load and frequency and recompute speed, Hertz "
    "pressure and PV with `tribo_extract.physics`; hardness follows the carbon level's mean.",
    "Active learning uses the design variables (carbon, load, frequency) on the nominal grid; "
    "replicates of a condition are averaged; acquisition and the stop rule use the latent "
    "(noise-free) GPR std.",
]


def fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        if a != 0 and (a < 1e-3 or a >= 1e5):
            return f"{v:.{nd - 1}e}"
        return f"{v:.{nd}f}" if a < 10 else f"{v:.{max(nd - 2, 1)}f}"
    return str(v)


def pm(m, s, nd=3) -> str:
    return f"{fmt(m, nd)} ± {fmt(s, nd)}" if np.isfinite(s) else fmt(m, nd)


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd) if not isinstance(r[c], str) else r[c]
                                     for c in cols) + " |")
    return "\n".join(out) + "\n"


def rel(path, out_dir: Path) -> str:
    try:
        return str(Path(path).relative_to(out_dir))
    except ValueError:
        return str(path)


def img(path, out_dir: Path, alt: str = "") -> str:
    return f"![{alt}]({rel(path, out_dir)})\n" if path else ""


# ---------------------------------------------------------------------------
def results_table(summary: pd.DataFrame, target: str, censor_mode: str) -> pd.DataFrame:
    s = summary[(summary["target"] == target) & (summary["censor_mode"] == censor_mode)] \
        .sort_values("RMSE_mean")
    if s.empty:
        return s
    rows = []
    best = s.iloc[0]
    se_best = best["RMSE_std"] / np.sqrt(best["n_folds"]) if best["n_folds"] > 1 else 0.0
    for i, r in enumerate(s.itertuples()):
        tie = i > 0 and r.RMSE_mean <= best["RMSE_mean"] + se_best
        d = {"model": f"**{r.model}**" if i == 0 else (f"{r.model} ≈" if tie else r.model),
             "features": r.feature_set,
             "R²": pm(r.R2_mean, r.R2_std), "RMSE": pm(r.RMSE_mean, r.RMSE_std),
             "MAE": pm(r.MAE_mean, r.MAE_std)}
        if target in TARGETS and TARGETS[target].is_log10:
            d["MdAPE (linear) %"] = fmt(getattr(r, "MdAPE_lin_pct_mean", np.nan), 1)
            d["× factor"] = fmt(getattr(r, "factor_err_mean", np.nan), 2)
        if "coverage95_mean" in s:
            d["95 % cov."] = fmt(getattr(r, "coverage95_mean", np.nan), 2)
        if "R2_oof" in s:
            d["R² pooled OOF"] = fmt(getattr(r, "R2_oof", np.nan), 3)
        d["folds"] = int(r.n_folds)
        rows.append(d)
    return pd.DataFrame(rows)


def sensitivity_table(summary: pd.DataFrame, cv_pred: pd.DataFrame, target: str, main: str,
                      modes: list[str]) -> pd.DataFrame:
    from .metrics import censored_consistency
    s = summary[summary["target"] == target]
    models = s[s["censor_mode"].isin(modes)][["model", "feature_set"]].drop_duplicates()
    rows = []
    for m, fs in models.itertuples(index=False):
        r = {"model": m, "features": fs}
        for mode in [main] + modes:
            x = s[(s["model"] == m) & (s["feature_set"] == fs) & (s["censor_mode"] == mode)]
            r[f"RMSE ({mode})"] = pm(x["RMSE_mean"].iloc[0], x["RMSE_std"].iloc[0]) if len(x) else "–"
            r[f"R² ({mode})"] = fmt(x["R2_mean"].iloc[0]) if len(x) else "–"
        for mode in modes:
            p = cv_pred[(cv_pred["target"] == target) & (cv_pred["model"] == m)
                        & (cv_pred["feature_set"] == fs) & (cv_pred["censor_mode"] == mode)]
            r[f"censored predicted < LOD ({mode})"] = fmt(censored_consistency(p), 2)
        rows.append(r)
    return pd.DataFrame(rows)


def holdout_tables(final, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = final.metrics[final.metrics["target"] == target] if len(final.metrics) else pd.DataFrame()
    mt = pd.DataFrame([{"model": r.model, "features": r.feature_set, "n": int(r.n),
                        "R²": fmt(r.R2), "RMSE": fmt(r.RMSE), "MAE": fmt(r.MAE),
                        "inside 95 % PI": fmt(getattr(r, "pi_coverage", np.nan), 2)}
                       for r in m.itertuples()]) if len(m) else pd.DataFrame()
    p = final.predictions[final.predictions["target"] == target] if len(final.predictions) else None
    if p is None or p.empty:
        return mt, pd.DataFrame()
    log10 = target in TARGETS and TARGETS[target].is_log10
    rows = []
    for r in p.sort_values(["Experiment_ID", "model"]).itertuples():
        d = {"run": r.Experiment_ID, "model": f"{r.model} ({r.feature_set})",
             "measured": fmt(r.y_true), "predicted": fmt(r.y_pred),
             "95 % PI": f"[{fmt(getattr(r, 'pi_lo', np.nan))}, {fmt(getattr(r, 'pi_hi', np.nan))}]"}
        if log10:
            d["k measured"] = fmt(10 ** r.y_true)
            d["k predicted"] = fmt(10 ** r.y_pred)
        rows.append(d)
    return mt, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def active_learning_section(al: dict | None, cfg: dict, out_dir: Path) -> str:
    if not al or not al.get("targets"):
        return "_Active-learning simulation not run (disabled or too few conditions)._\n\n"
    a = cfg["active_learning"]
    txt = [f"Pool = grid conditions of the CV runs; seed {a['n_seed']} maximin-LHS conditions; "
           f"budget {a['budget']}; {a['repeats']} repetitions per strategy; UCB κ = {a['kappa']}; "
           f"tolerance {100 * a['tolerance']:.0f} % of the full-grid group-CV RMSE; random baseline "
           f"seed design: `{a.get('random_seed_design', 'shared')}`.\n"]
    for t, v in al["targets"].items():
        s = v["summary"]
        ref = v["reference"]
        txt.append(f"\n**{t}** (pool {v['n_pool']} conditions; full-grid GPR group-CV RMSE "
                   f"{fmt(ref['rmse_full_cv'])}, R² {fmt(ref['r2_full_cv'])}"
                   + (f"; on the unseen points {fmt(ref['rmse_full_holdout'])}" if
                      "rmse_full_holdout" in ref else "") + ")\n\n")
        tab = s[["strategy", "N_AL_mean_curve", "N_AL_sustained_median_rep", "frac_reps_reaching",
                 "N_stop_median", "rmse_at_budget"]].rename(columns={
                     "N_AL_mean_curve": "N_AL (mean curve)",
                     "N_AL_sustained_median_rep": "N_AL (median rep., sustained)",
                     "frac_reps_reaching": "reps reaching tol.",
                     "N_stop_median": f"N at σ-stop (ε = {fmt(s['sigma_stop'].iloc[0])})",
                     "rmse_at_budget": f"RMSE at N = {int(s['N_budget'].iloc[0])}"})
        txt.append(md_table(tab))
        txt.append("\n" + img(v["figure"], out_dir, f"AL learning curves {t}"))
    txt.append("\n`N_AL (mean curve)` is the first N at which the mean error over repetitions "
               "is within tolerance of the full-grid model; `–` = not reached within the budget.\n")
    return "\n".join(txt) + "\n"


def trajectory_section(tr: dict | None, out_dir: Path) -> str:
    if tr is None:
        return ("_Skipped: no runs carry the eq.-4 parameters (`Vdot_steady_mm3_per_s`, "
                "`dV_run_mm3`, `tau_trans_s`), i.e. no run has >= 4 checkpoint scans._\n\n")
    txt = [f"{tr['n_runs']} CV runs carry eq.-4 parameters; measured curves from "
           f"**{tr['meas_source']}**. Fit-bound values excluded: {tr.get('n_at_bound', {})}.\n"]
    s = tr["summary"]
    if len(s):
        rows = [{"parameter": r.target, "model": r.model, "features": r.feature_set,
                 "R²": pm(r.R2_mean, r.R2_std), "RMSE (log10)": pm(r.RMSE_mean, r.RMSE_std)}
                for r in s.sort_values(["target", "RMSE_mean"]).itertuples()]
        txt.append("\nPer-parameter group CV (log10 scale):\n\n" + md_table(pd.DataFrame(rows)))
    if "curve_metrics" in tr:
        txt.append("\nReconstructed V(t) from out-of-fold parameter predictions vs measured "
                   "checkpoints (t > 0):\n\n" + md_table(tr["curve_metrics"]))
        txt.append("\n" + img(tr["figures"].get("curves"), out_dir, "trajectory curves"))
    tau = s[s["target"] == "log10_tau_trans"]["R2_mean"] if len(s) else pd.Series(dtype=float)
    if len(tau) and tau.max() < 0.2:
        txt.append("\n> **τ_trans is not predictable here** (best R² "
                   f"{fmt(tau.max())}). With checkpoints at 0/60/150/300/450/600 s only one scan "
                   "falls inside a running-in of tens of seconds, so the eq.-4 fit cannot resolve "
                   "τ_trans; the curve as a whole is still predicted (table above). Adding "
                   "checkpoints at ~15 and 30 s would make τ_trans identifiable.\n")
    return "\n".join(txt) + "\n"


def interpretation_section(ir: dict | None, out_dir: Path) -> str:
    if not ir:
        return "_Interpretation not run._\n\n"
    f = ir["figures"]
    txt = []
    if ir.get("pdp_shapes"):
        rows = []
        for (t, m, fs), sh in ir["pdp_shapes"].items():
            rows.append({"target": t, "model": f"{m} ({fs})",
                         "load: shape": sh["load"]["shape"],
                         "load: rise over range": fmt(sh["load"]["rise"]),
                         "load: R² of a line": fmt(sh["load"]["R2_linear"]),
                         "carbon: rise over range": fmt(sh["carbon"]["rise"]),
                         "frequency: rise": fmt(sh["freq"]["rise"])})
        txt.append("**Partial dependence** (physically consistent; rise = PD at the top of the "
                   "range minus the bottom, in target units; shape from the sagitta of the "
                   "chord: curved if |rel.| > 0.1 and a line explains < 98 %):\n\n" + md_table(pd.DataFrame(rows)))
        for k, p in f.items():
            if k.startswith("pdp_"):
                txt.append(img(p, out_dir, k))
    if ir.get("shap"):
        rows = [{"target": t, "model": f"{m} ({fs})",
                 "ranking (mean |SHAP|)": " > ".join(
                     f"{a} ({fmt(b, 3)})" for a, b in imp[["feature", "mean_abs_shap"]].values[:5])}
                for (t, m, fs), imp in ir["shap"].items()]
        txt.append("\n**SHAP** (on the final models; with correlated inputs such as carbon and "
                   "hardness the attribution is shared between them):\n\n" + md_table(pd.DataFrame(rows)))
        for k, p in f.items():
            if k.startswith("shap_"):
                txt.append(img(p, out_dir, k))
    if ir.get("gpr_lengthscales"):
        rows = [{"target": t, "features": fs, **{k: fmt(v, 2) for k, v in ls.items()}}
                for (t, fs), ls in ir["gpr_lengthscales"].items()]
        txt.append("\n**GPR ARD length scales** (standardised inputs; small = strong/curved "
                   "effect, very large = irrelevant):\n\n" + md_table(pd.DataFrame(rows)))
        for k, p in f.items():
            if k.startswith("gpr_map_"):
                txt.append(img(p, out_dir, k))
    kan = [k for k in f if k.startswith("kan_")]
    if kan:
        txt.append("\n**KAN learned univariate functions** (first layer; one line per hidden "
                   "node; Σ|φ| = mean absolute activation summed over nodes):\n")
        for k in kan:
            txt.append(img(f[k], out_dir, k))
    if ir.get("archard") is not None and len(ir["archard"]):
        a = ir["archard"]
        txt.append("\n**Archard check**: OLS of log10 V on log10 F and log10 S with carbon-level "
                   "offsets (detected CV runs). Archard's law predicts both exponents = 1. "
                   "With fixed test duration, S ∝ frequency, so the S exponent also carries any "
                   "frequency effect on k.\n\n" + md_table(a.drop(columns=["archard"])))
    if ir.get("symbolic") is not None:
        txt.append("\n**PySR equations** (log10 k, base features):\n\n"
                   + md_table(ir["symbolic"].head(8)))
    for n in ir.get("notes", []):
        txt.append(f"\n- note: {n}")
    return "\n".join(txt) + "\n"


# ---------------------------------------------------------------------------
def write_report(out_dir: Path, ctx: dict) -> Path:
    """Write ``report.md``; ``ctx`` is assembled by :mod:`ml.train`."""
    out_dir = Path(out_dir)
    cfg, data, summary = ctx["cfg"], ctx["data"], ctx["summary"]
    main = cfg["censoring"]["mode"]
    env = json.loads((out_dir / "environment.json").read_text()) if \
        (out_dir / "environment.json").exists() else {}
    L = [f"# ME 323 triboinformatics: model report\n",
         f"- results folder: `{out_dir.name}`  ·  data: **{data.source}**"
         f"{'  ·  **--quick** (reduced search; smoke test)' if cfg.get('quick_mode') else ''}",
         f"- command: `{' '.join(env.get('argv', []))}`",
         f"- git commit: `{env.get('git_commit')}`  ·  seed {cfg['seed']}  ·  run time "
         f"{ctx.get('seconds', 0) / 60:.1f} min  ·  {env.get('platform', '')}",
         f"- CV: outer {cfg['cv']['outer_folds']}-fold group K-fold × {cfg['cv']['repeats']} "
         f"repeat(s), inner {cfg['cv']['inner_folds']}-fold, ≤ {cfg['cv']['search_iter']} "
         f"candidates per model; feature sets {cfg['features']['use']}\n"]

    L.append("## 1. Data\n")
    cv_runs = data.cv_runs
    L.append(f"- {len(data.runs)} usable runs: {len(cv_runs)} for CV in "
             f"{cv_runs['group'].nunique()} groups "
             f"({int(cv_runs['Replicate_of'].notna().sum())} replicate runs), "
             f"{len(data.holdout_ids)} unseen holdout runs ({', '.join(data.holdout_ids) or 'none'}).")
    if data.dropped_inputs:
        L.append("- dropped input columns: " + ", ".join(f"`{c}` ({w})" for c, w in
                                                         data.dropped_inputs.items()))
    for t, info in ctx.get("target_info", {}).items():
        extra = ""
        if "n_censored" in info:
            extra = (f"; {info['n_censored']} censored (< LOD) → "
                     + ("excluded" if main == "exclude" else "imputed at LOD/2"))
        L.append(f"- `{t}`: {info.get('n_runs', 0)} runs"
                 + (f", {info.get('n_rows')} rows" if info.get("n_rows") != info.get("n_runs") else "")
                 + extra + (f"; {info['n_missing']} without a value" if info.get("n_missing") else ""))
    L.append("\n<details><summary>data-loading log</summary>\n")
    L += [f"- {n}" for n in data.notes]
    L.append("\n</details>")
    L.append("")
    L.append("## 2. Models\n")
    L.append("Ran: " + ", ".join(f"`{s}`" for s in ctx["models_run"]) + "\n")
    if ctx.get("skipped"):
        L.append("Skipped (optional dependency missing): " + ", ".join(
            f"`{k}` ({v})" for k, v in ctx["skipped"].items()) + "\n")
    if ctx.get("failures"):
        L.append("**Failures:**\n\n" + md_table(pd.DataFrame(ctx["failures"]).head(20)))

    L.append("## 3. Cross-validated results\n")
    L.append("Mean ± std over outer folds (held-out groups). The best row (lowest RMSE) is bold; "
             "`≈` marks models within one standard error (std/√folds) of it, i.e. not "
             "distinguishable from the best on this data. `95 % cov.` = fraction of held-out "
             "points inside the model's 95 % interval.\n")
    figs = ctx.get("interp", {}).get("figures", {}) if ctx.get("interp") else {}
    for t in ctx["targets"] + ([COF_T_RUN] if "COF_t" in ctx["targets"] else []):
        tab = results_table(summary, t, main)
        if tab.empty:
            continue
        label = TARGETS[t].label if t in TARGETS else "COF(t) averaged over the steady phase, per run"
        L.append(f"### {t}: {label}\n")
        if t == "COF_t":
            L.append("Scored per second on every second of the held-out runs.\n")
        if t == COF_T_RUN:
            L.append("Run-level score of the COF(t) models: predicted and measured COF(t) "
                     "averaged over the steady phase of each held-out run. Compare with "
                     "`COF_ss_mean`.\n")
        L.append(md_table(tab))
        L.append(img(figs.get(f"cv_rmse_{t}"), out_dir, f"CV RMSE {t}"))
    if figs.get("parity"):
        L.append(img(figs["parity"], out_dir, "parity"))
    for t in ctx["targets"]:
        if figs.get(f"calibration_{t}"):
            L.append(img(figs[f"calibration_{t}"], out_dir, f"calibration {t}"))
    if figs.get("cof_curves_cv"):
        L.append(img(figs["cof_curves_cv"], out_dir, "COF curves"))

    sens = [m for m in cfg["censoring"].get("sensitivity", []) if m != main]
    if sens:
        L.append("## 4. Censoring sensitivity\n")
        L.append(f"Wear targets refitted with censored runs imputed at LOD/2 ({', '.join(sens)}); "
                 "scores use the detected runs only, so the columns are comparable. The last "
                 "column is the fraction of censored runs whose out-of-fold prediction falls "
                 "below their detection limit (1 = consistent).\n")
        for t in [t for t in ctx["targets"] if TARGETS[t].censorable]:
            L.append(f"**{t}**\n\n" + md_table(sensitivity_table(summary, ctx["cv_predictions"], t,
                                                                 main, sens)))

    L.append("## 5. Unseen operating points (final test)\n")
    final = ctx.get("final")
    if final is None or final.predictions.empty:
        L.append("_No unseen operating points in this dataset: list their Experiment_IDs under "
                 f"`data.{data.source}.holdout_ids` in the config._\n")
    else:
        L.append("Selected models were refitted on all CV data (hyperparameters re-tuned by group "
                 "CV) and predict the held-out runs, which were not used anywhere before this "
                 "step.\n\nSelection:\n\n" + md_table(final.selections.rename(
                     columns={"RMSE_mean": "CV RMSE"})))
        for t in ctx["targets"] + [COF_T_RUN]:
            mt, pr = holdout_tables(final, t)
            if mt.empty:
                continue
            L.append(f"### {t}\n\n" + md_table(mt))
            if not pr.empty and t != "COF_t":
                L.append("\n<details><summary>per-run predictions</summary>\n\n" + md_table(pr)
                         + "\n</details>\n")
            L.append(img(figs.get(f"holdout_{t}"), out_dir, f"holdout {t}"))
        if figs.get("cof_curves_holdout"):
            L.append(img(figs["cof_curves_holdout"], out_dir, "holdout COF curves"))

    L.append("## 6. Interpretation\n")
    L.append(interpretation_section(ctx.get("interp"), out_dir))
    L.append("## 7. Extension A: active-learning DOE (retrospective)\n")
    L.append(active_learning_section(ctx.get("al"), cfg, out_dir))
    L.append("## 8. Extension B: wear trajectory\n")
    L.append(trajectory_section(ctx.get("trajectory"), out_dir))
    L.append("## 9. Method notes and assumptions\n")
    L += [f"- {a}" for a in ASSUMPTIONS]
    if data.source == "synthetic":
        L.append("- **Synthetic data**: effects are planted (see `data/synthetic/planted_effects."
                 "json`); numbers here validate the pipeline, not the material.")
    L.append("\nFiles: `metrics.csv` (this table), `metrics_folds.csv`, `predictions.csv`, "
             "`predictions_timeseries.csv.gz`, `hyperparameters.csv`, `holdout_*.csv`, "
             "`interpretation/`, `active_learning/`, `trajectory/`, `models/`, "
             "`config_snapshot.yaml`, `environment.json`, `run.log`.\n")
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
