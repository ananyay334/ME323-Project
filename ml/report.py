"""Auto-generated ``report.md`` for a results folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGETS
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
    """One row per (model, feature set): gkf5 and LOAO mean ± std, GPR coverage."""
    s = summary[summary["target"] == target]
    if s.empty:
        return s
    g = s[s["split"] == "gkf5"].set_index(["model", "feature_set"])
    lo = s[s["split"] == "loao"].set_index(["model", "feature_set"])
    b = best[best["target"] == target]
    bkey = (b["model"].iloc[0], b["feature_set"].iloc[0]) if len(b) else None
    reg = registry()
    rows = []
    for key in g.sort_values("RMSE_mean").index:
        r = g.loc[key]
        m = key[0]
        name = f"**{m}**" if key == bkey else m
        if m in reg and not reg[m].selectable:
            name += " (reference)"
        if m == "physics_route":
            name += " (eq.-1 reconstruction)"
        row = {"model": name, "features": key[1],
               "gkf5 R²": pm(r["R2_mean"], r["R2_std"]), "gkf5 RMSE": pm(r["RMSE_mean"], r["RMSE_std"]),
               "gkf5 MAE": pm(r["MAE_mean"], r["MAE_std"])}
        if key in lo.index:
            q = lo.loc[key]
            row.update({"LOAO R²": pm(q["R2_mean"], q["R2_std"]),
                        "LOAO RMSE": pm(q["RMSE_mean"], q["RMSE_std"]),
                        "LOAO MAE": pm(q["MAE_mean"], q["MAE_std"])})
        else:
            row.update({"LOAO R²": "–", "LOAO RMSE": "–", "LOAO MAE": "–"})
        row["95 % cov."] = fmt(r.get("coverage95_mean", np.nan), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, ctx: dict) -> Path:
    """Write ``report.md``; ``ctx`` is assembled by :mod:`ml.train`."""
    out_dir = Path(out_dir)
    cfg, data, summary, best = ctx["cfg"], ctx["data"], ctx["summary"], ctx["best"]
    env = json.loads((out_dir / "environment.json").read_text()) \
        if (out_dir / "environment.json").exists() else {}
    plan = ctx["plan"]
    figs = (ctx.get("interp") or {}).get("figures", {})
    L = ["# ME 323 triboinformatics: model report (40-run DOE)\n",
         f"- results folder: `{out_dir.name}` · data: **{data.source}**"
         + (" · **--quick** (smoke test)" if cfg.get("quick_mode") else ""),
         f"- command: `{' '.join(env.get('argv', []))}` · git `{env.get('git_commit')}` · seed "
         f"{cfg['seed']} · run time {ctx.get('seconds', 0) / 60:.1f} min",
         f"- validation: gkf5 = StratifiedGroupKFold(5) by specimen × {len(plan.seeds)} repeat(s) "
         f"(seeds {plan.seeds}, seed offsets used {plan.offsets}); LOAO = "
         f"{len(plan.loao_labels)} folds ({', '.join(plan.loao_labels)}); nested inner "
         f"GroupKFold({cfg['cv']['inner_folds']})"
         + ("; first grid value only" if cfg.get("grid_first_only") else "; full grids") + "\n"]

    L.append("## 1. Data\n")
    L.append(f"- {len(data.runs)} usable runs in {data.runs['group'].nunique()} groups on "
             f"{data.runs['Specimen_ID'].nunique()} specimens; censoring mode "
             f"`{cfg['censoring']['mode']}`.")
    if data.dropped_inputs:
        L.append("- dropped input columns: " + ", ".join(f"`{c}` ({w})" for c, w in
                                                         data.dropped_inputs.items()))
    rows = []
    for t, info in ctx["target_info"].items():
        rows.append({"regime": info["regime"], "target": t, "rows": info.get("n_rows", 0),
                     "runs": info.get("n_runs", 0),
                     "censored (< LOD)": info.get("n_censored", "–"),
                     "censored excluded": info.get("n_censored_excluded", "–"),
                     "without value": info.get("n_missing", 0)})
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
    L.append("Mean ± std over outer folds (gkf5: folds × repeats; LOAO: one fold per held-out "
             "grade). Bold = selected (lowest gkf5 RMSE, Ridge excluded). `95 % cov.` = GPR / "
             "TabPFN interval coverage on gkf5.\n")
    noise = ctx.get("noise", {})
    for regime, targets in cfg["regimes"].items():
        ts = [t for t in targets if t in set(summary["target"])]
        if not ts:
            continue
        L.append(f"### {regime}\n")
        for t in ts:
            L.append(f"**{t}** — {TARGETS[t].label}")
            if t in noise:
                L.append(f"\nNoise floor (pooled SD of the {noise[t]['n_pairs']} centre replicate "
                         f"pairs): **{fmt(noise[t]['noise_sd'])}** — compare with the RMSE columns.")
            L.append("\n" + md_table(target_table(summary, t, best)))
    if figs.get("parity"):
        L.append(img(figs["parity"], out_dir, "parity"))

    L.append("## 4. Leave-one-alloy-out, per held-out grade\n")
    lg = (ctx.get("interp") or {}).get("loao_grades", pd.DataFrame())
    if lg is not None and len(lg):
        piv = lg.pivot_table(index=["target", "model", "feature_set"], columns="grade",
                             values="RMSE").reset_index()
        piv.columns = [str(c) for c in piv.columns]
        L.append("RMSE on the held-out grade (best feature set per model):\n\n" + md_table(piv))
    L.append(img(figs.get("loao_per_grade"), out_dir, "LOAO per grade"))

    L.append("## 5. Wear trajectory: physics route vs direct models\n")
    s = summary[(summary["target"] == "log10_V_t") & (summary["split"] == "gkf5")]
    if len(s) and (s["model"] == "physics_route").any():
        pr = s[s["model"] == "physics_route"].sort_values("RMSE_mean").iloc[0]
        sel = best[best["target"] == "log10_V_t"]
        dr = s[(s["model"] == sel["model"].iloc[0]) & (s["feature_set"] == sel["feature_set"].iloc[0])] \
            .iloc[0] if len(sel) else s[s["model"] != "physics_route"].sort_values("RMSE_mean").iloc[0]
        L.append(f"- physics route (eq.-1 parameters → `two_stage_wear`): RMSE "
                 f"{pm(pr['RMSE_mean'], pr['RMSE_std'])}, R² {fmt(pr['R2_mean'])} ({pr['feature_set']})")
        L.append(f"- selected direct `wear_checkpoints` model: {dr['model']} ({dr['feature_set']}), RMSE "
                 f"{pm(dr['RMSE_mean'], dr['RMSE_std'])}, R² {fmt(dr['R2_mean'])}")
        ch = ctx.get("route_choices")
        if ch is not None and len(ch):
            L.append("- models chosen by inner CV for the eq.-1 parameters (count over folds): "
                     + "; ".join(f"{p}: " + ", ".join(f"{m} ×{n}" for m, n in v.items())
                                 for p, v in ch.items()))
    else:
        L.append("_Physics route not run (no checkpoint scans or too few eq.-1 fits)._")
    L.append("\n" + img(figs.get("trajectory"), out_dir, "trajectory reconstructions"))

    L.append("## 6. Selected models\n")
    L.append(md_table(best.rename(columns={"R2_mean": "gkf5 R²", "RMSE_mean": "gkf5 RMSE"})))

    L.append("## 7. GPR\n")
    for k in sorted(figs):
        if k.startswith("gpr_surface_"):
            L.append(img(figs[k], out_dir, k))
    L.append(img(figs.get("gpr_lengthscales"), out_dir, "GPR length scales"))
    L.append(img(figs.get("gpr_calibration"), out_dir, "GPR calibration"))
    for n in (ctx.get("interp") or {}).get("notes", []):
        L.append(f"- note: {n}")

    L.append("\n## 8. Assumptions\n\nSee `ml/README.md` → *Assumptions* (copied into every run).\n")
    L += [f"- {a}" for a in ctx.get("assumptions", [])]
    if data.source == "synthetic":
        L.append("- **Synthetic data**: effects are planted (`data/synthetic/planted_effects.json`); "
                 "the numbers validate the pipeline, not the material.")
    L.append("\nFiles: `metrics.csv`, `metrics_folds.csv`, `predictions.csv`, `hyperparameters.csv`, "
             "`best_models.csv`, `physics_route_params.csv`, `interpretation/`, `figures/`, "
             "`config_snapshot.yaml`, `environment.json`, `run.log`.\n")
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
