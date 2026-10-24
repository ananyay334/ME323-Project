"""Auto-generated ``report.md`` for a results folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import INPUTS, TARGETS
from .models.registry import registry


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
        out.append("| " + " | ".join(r[c] if isinstance(r[c], str) else fmt(r[c], nd)
                                     for c in cols) + " |")
    return "\n".join(out) + "\n"


def img(path, out_dir: Path, alt: str = "") -> str:
    if not path:
        return ""
    try:
        rel = Path(path).relative_to(out_dir)
    except ValueError:
        rel = Path(path)
    return f"![{alt}]({rel})\n"


def target_table(summary: pd.DataFrame, target: str, best: pd.DataFrame) -> pd.DataFrame:
    """One row per model: CV mean ± SD over outer folds/repeats, interval coverage."""
    s = summary[summary["target"] == target].sort_values("RMSE_mean")
    b = best[best["target"] == target]
    bm = b["model"].iloc[0] if len(b) else None
    reg = registry()
    rows = []
    for r in s.itertuples():
        name = f"**{r.model}**" if r.model == bm else r.model
        if r.model in reg and not reg[r.model].selectable:
            name += " (reference)"
        rows.append({"model": name, "pooled R²": fmt(getattr(r, "R2_pooled", np.nan)),
                     "R²": pm(r.R2_mean, r.R2_std), "RMSE": pm(r.RMSE_mean, r.RMSE_std),
                     "MAE": pm(r.MAE_mean, r.MAE_std),
                     "95 % cov.": fmt(getattr(r, "coverage95_mean", np.nan), 2), "folds": r.n_folds})
    return pd.DataFrame(rows)


def factor_table(interp: dict, target: str, best_model: str | None) -> pd.DataFrame:
    """GPR length scale, permutation importance (best model) and Ridge exponent per input."""
    ls = interp.get("lengthscales", pd.DataFrame())
    imp = interp.get("importance", pd.DataFrame())
    ex = interp.get("exponents", pd.DataFrame())
    rows = []
    for f in INPUTS:
        row = {"input": f}
        if ls is not None and len(ls) and (ls["target"] == target).any() and f in ls:
            row["GPR length scale"] = fmt(float(ls.loc[ls["target"] == target, f].iloc[0]))
        if imp is not None and len(imp) and best_model:
            q = imp[(imp["target"] == target) & (imp["model"] == best_model) & (imp["feature"] == f)]
            row[f"perm. importance ΔRMSE ({best_model})"] = pm(q["mean"].iloc[0], q["std"].iloc[0]) if len(q) else "–"
        if ex is not None and len(ex):
            q = ex[(ex["target"] == target) & (ex["feature"] == f)]
            if len(q):
                q = q.iloc[0]
                row["Ridge exponent [95 % CI]"] = f"{fmt(q['exponent'])} [{fmt(q['ci_low'])}, {fmt(q['ci_high'])}]"
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, ctx: dict) -> Path:
    """Write ``report.md``; ``ctx`` is assembled by :mod:`ml.train`."""
    out_dir = Path(out_dir)
    cfg, data, summary, best = ctx["cfg"], ctx["data"], ctx["summary"], ctx["best"]
    env = json.loads((out_dir / "environment.json").read_text()) \
        if (out_dir / "environment.json").exists() else {}
    plan = ctx["plan"]
    interp = ctx.get("interp") or {}
    figs = interp.get("figures", {})
    f = data.frame
    L = ["# ME 323 — 304 stainless steel, dry reciprocating: model report\n",
         f"- results folder: `{out_dir.name}` · data: **{data.source}**"
         + (" · **--quick** (smoke test)" if cfg.get("quick_mode") else "")
         + (" · **--final**" if ctx.get("final") else ""),
         f"- command: `{' '.join(env.get('argv', []))}` · git `{env.get('git_commit')}` · seed "
         f"{cfg['seed']} · run time {ctx.get('seconds', 0) / 60:.1f} min",
         f"- validation: {plan.n_splits}-fold GroupKFold by run group (R04/R14/R20 = one group; both "
         f"checkpoints of a run together), groups shuffled by seed, {len(plan.seeds)} repeat(s) "
         f"(seeds {plan.seeds}); nested inner GroupKFold({plan.n_inner})"
         + ("; first grid value only" if cfg.get("grid_first_only") else "; full grids")
         + ". Test runs R24–R29 are not loaded" + (" until the final refit." if ctx.get("final") else ".") + "\n"]

    L.append("## 1. Data\n")
    L.append(f"- training rows: {len(f)} ({f['Experiment_ID'].nunique()} runs, {f['group'].nunique()} groups; "
             f"roles: " + ", ".join(f"{k} {v}" for k, v in f.drop_duplicates('Experiment_ID')['Role']
                                    .value_counts().items()) + ")")
    L.append("- inputs: ln(Load_N), ln(Freq_Hz), ln(Time_s) (nominal; standardised inside the folds)")
    rows = []
    for t in data.targets:
        rows.append({"target": t, "label": TARGETS[t], "rows": int(f[t].notna().sum()),
                     "censored (< LOD)": int(f[f"censored_{t}"].sum()),
                     "censoring": cfg["censoring"]["mode"] if t == "log10_k" else "–"})
    L.append("\n" + md_table(pd.DataFrame(rows)))
    L.append("<details><summary>data-loading log</summary>\n")
    L += [f"- {n}" for n in data.notes]
    L.append("\n</details>\n")

    L.append("## 2. Models\n")
    L.append("Ran: " + ", ".join(f"`{m}`" for m in ctx["models_run"]) + ".\n")
    for m, why in ctx.get("skipped", {}).items():
        L.append(f"> **⚠ `{m}` NOT RUN** — {why}.\n")
    if ctx.get("failures"):
        L.append("**Failures:**\n\n" + md_table(pd.DataFrame(ctx["failures"]).head(20)))

    L.append("## 3. Cross-validated results\n")
    L.append("Mean ± SD over outer folds × repeats. Bold = selected (lowest CV RMSE, Ridge excluded). "
             "`pooled R²` = R² of all out-of-fold predictions of a repeat together (mean over repeats); "
             "the fold-wise R² is unstable with 3–4 run groups per fold, so read the pooled one. "
             "`95 % cov.` = coverage of the central 95 % interval (GPR, ANN ensemble, TabPFN).\n")
    noise = ctx.get("noise", {})
    for t in data.targets:
        L.append(f"### {t} — {TARGETS[t]}\n")
        if t in noise:
            nf = noise[t]
            L.append(f"Noise floor (pooled SD of R04/R14/R20 over {nf['n_checkpoints']} checkpoint(s)): "
                     f"**{fmt(nf['noise_sd'])}** — compare with the RMSE column.\n")
        L.append(md_table(target_table(summary, t, best)))
    L.append(img(figs.get("parity"), out_dir, "parity (CV)"))

    L.append("## 4. Selected models\n")
    L.append(md_table(best.rename(columns={"R2_mean": "CV R²", "RMSE_mean": "CV RMSE"})))

    L.append("## 5. Which factor controls COF and wear\n")
    L.append("GPR ARD length scale (standardised inputs; small = strong or curved effect), group-permutation "
             "importance (increase in held-out RMSE when an input is permuted between run groups; first "
             "repeat, best model) and the Ridge exponent d(target)/d(ln x) with a group-bootstrap 95 % "
             f"interval ({cfg['interpret'].get('bootstrap', 1000)} resamples).\n")
    for t in data.targets:
        b = best[best["target"] == t]
        L.append(f"**{t}**\n\n" + md_table(factor_table(interp, t, b["model"].iloc[0] if len(b) else None)))
    for t in data.targets:
        L.append(img(figs.get(f"gpr_surface_{t}"), out_dir, f"GPR surface {t}"))
    L.append(img(figs.get("cof_replicates"), out_dir, "replicate COF"))

    tp = ctx.get("topup")
    L.append("## 6. Top-up runs: pre-registered prediction vs measurement\n")
    if tp is not None and len(tp):
        L.append("Predictions written by `python -m ml.topup` before the runs were made "
                 "(`topup_candidates.csv`):\n\n" + md_table(tp))
    else:
        L.append("_No measured top-up runs yet._\n")

    fin = ctx.get("final")
    if fin:
        L.append("## 7. Test set (R24–R29, scored once)\n")
        L.append(f"Selected model per target refitted on all {len(f)} training rows; {fin['n_runs']} test runs.\n")
        L.append(md_table(fin["metrics"][[c for c in ["target", "model", "R2", "RMSE", "MAE", "coverage95", "n"]
                                          if c in fin["metrics"]]]))
        L.append(img(fin.get("figure"), out_dir, "parity (test)"))

    L.append("\n## Assumptions\n\nFrom the project `README.md` → *Assumptions*.\n")
    L += [f"- {a}" for a in ctx.get("assumptions", [])]
    if data.source == "synthetic":
        L.append("- **Synthetic data**: effects are planted (`data/synthetic/planted_effects.json`); "
                 "the numbers validate the pipeline, not the material.")
    for n in interp.get("notes", []):
        L.append(f"- note: {n}")
    L.append("\nFiles: `metrics.csv`, `metrics_folds.csv`, `predictions.csv`, `hyperparameters.csv`, "
             "`best_models.csv`, `importance_folds.csv`, `interpretation/`, `figures/`, "
             "`config_snapshot.yaml`, `environment.json`, `run.log`"
             + (", `test_metrics.csv`, `test_predictions.csv`" if fin else "") + ".\n")
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
