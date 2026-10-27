"""C2 - Which wear measure?

Most-worn location vs mean of locations, robust vs plain levelling, the earlier 2-D box high-pass, the TA depth
rule, max depth and track width: runs measurable, replicate SD, best R² (P1, P2) of the physics law / power law,
and the wear-volume exponents against Archard (V ~ L^1 S^1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.registry import target_frame
from .common import adhoc_target, cv_oof, ols_with_ci, replicate_sd, score

MEASURES = [
    # key, label, source column in wear_variants (or runs), status column (None = any positive value), physics
    ("V_maxloc_robust", "V most-worn location, robust levelling (final)", "robust__V_mm3", "archard"),
    ("V_meanloc_robust", "V mean of locations, robust levelling", "robust__V_mean_loc_mm3", "archard"),
    ("V_maxloc_plain", "V most-worn location, plain levelling", "plain__V_mm3", "archard"),
    ("V_meanloc_plain", "V mean of locations, plain levelling (old)", "plain__V_mean_loc_mm3", "archard"),
    ("V_maxloc_box", "V most-worn, 2-D box high-pass (anchor implementation)", "box__V_mm3", "archard"),
    ("V_ta_rule", "V most-worn, TA depth rule", "ta_rule__V_mm3", "archard"),
    ("depth_max", "max wear depth", "robust__depth_max_um", None),
    ("width", "track width", "robust__width_um", "width"),
]


def run_c2(cfg: dict, runs: pd.DataFrame, variants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    d = runs.merge(variants, on="run_id")
    rows, expo = [], []
    for key, label, col, physics in MEASURES:
        dd = d.assign(**{key: d[col].where(d[col] > 0)})
        t = adhoc_target(key, True, physics or "width", label=label)
        df = target_frame(dd, t)
        res = {"measure": key, "label": label, "runs_measurable": len(df), "replicate_sd_dex": replicate_sd(df),
               "spread_sd_dex": df.y.std(ddof=1)}
        res["signal_to_noise"] = res["spread_sd_dex"] / res["replicate_sd_dex"] if res["replicate_sd_dex"] > 0 else np.nan
        cands = (["physics"] if physics else []) + ["powerlaw"]
        best = {}
        for m in cands:
            s1, s2 = score(cv_oof(cfg, dd, t, m, "P1")), score(cv_oof(cfg, dd, t, m, "P2"))
            res[f"{m}_R2_P1"], res[f"{m}_R2_P2"] = s1["R2"], s2["R2"]
            best[m] = s1["R2"]
        bm = max(best, key=best.get)
        res["best_model"] = bm
        res["best_R2_P1"] = res[f"{bm}_R2_P1"]
        res["best_R2_P2"] = res[f"{bm}_R2_P2"]
        if key.startswith("V_"):
            X = np.column_stack([np.log10(df.L), df.log10_LS - np.log10(df.L)])  # log10 L, log10 S
            e = ols_with_ci(X, df.y.to_numpy(), df.group.to_numpy(), ["log10_L", "log10_S"],
                            cfg["evaluate"]["n_bootstrap"], cfg["seed"])
            e.insert(0, "measure", key)
            expo.append(e)
            for term in ("log10_L", "log10_S"):
                r = e[e.term == term].iloc[0]
                res[f"exp_{term[6:]}"] = r.estimate
                res[f"exp_{term[6:]}_ci"] = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
        rows.append(res)
    tab = pd.DataFrame(rows)
    ex = pd.concat(expo, ignore_index=True)
    # share of the (robust, most-worn) groove volume each V measure keeps, low vs high load
    ref = d.set_index("run_id")["robust__V_mm3"]
    for key, _, col, _ in MEASURES:
        if key.startswith("V_"):
            kept = (d.set_index("run_id")[col] / ref).where(ref > 0)
            L = d.set_index("run_id").Load_N
            tab.loc[tab.measure == key, "kept_frac_L_le_25N"] = kept[L <= 25].median()
            tab.loc[tab.measure == key, "kept_frac_L_ge_60N"] = kept[L >= 60].median()
    f = tab.set_index("measure")
    a, b = f.loc["V_maxloc_robust"], f.loc["V_meanloc_plain"]
    ta = f.loc["V_ta_rule"]
    verdict = (f"Use the most-worn valid location with robust levelling: {int(a.runs_measurable)} runs measurable with an "
               f"LOD, replicate SD {a.replicate_sd_dex:.2f} dex, Archard R² P1 {a.physics_R2_P1:+.2f} / P2 "
               f"{a.physics_R2_P2:+.2f}; the old mean-of-locations, plain-levelling measure gives "
               f"{b.replicate_sd_dex:.2f} dex and R² {b.physics_R2_P1:+.2f} / {b.physics_R2_P2:+.2f}. "
               f"Exponents: V ~ L^{a.exp_L:.2f} {a.exp_L_ci} S^{a.exp_S:.2f} {a.exp_S_ci} (Archard: 1, 1). "
               f"The TA depth rule scores a higher R² ({ta.best_R2_P1:+.2f}) only because it drops the shallow part of every "
               f"groove: runs at <= 25 N keep a median {100 * ta.kept_frac_L_le_25N:.0f} % of their groove, runs at >= 60 N "
               f"{100 * ta.kept_frac_L_ge_60N:.0f} %, which inflates the load exponent to {ta.exp_L:.1f}; it is not a volume.")
    return tab, ex, verdict
