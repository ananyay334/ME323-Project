"""C1 - Which COF definition?

For every definition: replicate SD (R04/R14/R20), spread over runs / replicate SD, physics-law R² (P1, P2) of
COF = mu0 + c L^(-1/3), and the frequency bias of the instrument mean (its ratio to our offset-free COF vs f).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..models.registry import target_frame
from .common import adhoc_target, cv_oof, replicate_sd, score

DEFINITIONS = {
    "COF_first10s": "first 10 s (primary)",
    "COF_0.5_2m": "0.5-2 m of sliding",
    "COF_end": "end: second-half median",
    "COF_last10s": "last 10 s",
    "COF_whole": "whole run (mean)",
    "COF_instrument_mean": "instrument |Fx|/Fz mean",
}


def run_c1(cfg: dict, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    for col, label in DEFINITIONS.items():
        t = adhoc_target(col, False, "cof", label=label)
        df = target_frame(runs, t)
        p1, p2 = score(cv_oof(cfg, runs, t, "physics", "P1")), score(cv_oof(cfg, runs, t, "physics", "P2"))
        rsd = replicate_sd(df)
        c = np.polyfit(np.exp(df.lnL) ** (-1 / 3), df.y, 1)
        rows.append({"definition": col, "label": label, "n": len(df), "mean": df.y.mean(), "replicate_sd": rsd,
                     "spread_sd": df.y.std(ddof=1), "signal_to_noise": df.y.std(ddof=1) / rsd,
                     "physics_R2_P1": p1["R2"], "physics_R2_P2": p2["R2"], "physics_RMSE_P1": p1["RMSE"],
                     "c_adhesion": c[0], "mu0": c[1]})
    tab = pd.DataFrame(rows)

    ratio = runs[["run_id", "Freq_Hz", "Load_N", "COF_whole", "COF_instrument_mean", "COF_first10s"]].copy()
    ratio["ratio_instrument_to_ours"] = ratio.COF_instrument_mean / ratio.COF_whole
    r, p = stats.pearsonr(ratio.Freq_Hz, ratio.ratio_instrument_to_ours)
    slope = np.polyfit(ratio.Freq_Hz, ratio.ratio_instrument_to_ours, 1)[0]
    tab["instrument_ratio_mean"] = ratio.ratio_instrument_to_ours.mean()
    tab["instrument_ratio_vs_f_r"] = r
    tab["instrument_ratio_vs_f_p"] = p
    tab["instrument_ratio_slope_per_Hz"] = slope

    best = tab.loc[tab.physics_R2_P1.idxmax()]
    f10 = tab.set_index("definition").loc["COF_first10s"]
    end = tab.set_index("definition").loc["COF_end"]
    verdict = (f"Use COF_first10s: it follows the adhesion law best among the definitions (P1 R² {f10.physics_R2_P1:+.2f}, "
               f"P2 {f10.physics_R2_P2:+.2f}; end-of-run COF {end.physics_R2_P1:+.2f}) with signal-to-noise "
               f"{f10.signal_to_noise:.1f}; the instrument mean reads {100 * (1 - ratio.ratio_instrument_to_ours.mean()):.0f} % "
               f"low and its ratio to ours falls with frequency (r = {r:+.2f}).")
    if best.definition != "COF_first10s":
        verdict = (f"Highest physics-law R² (P1) is {best.definition} ({best.physics_R2_P1:+.2f}) vs COF_first10s "
                   f"({f10.physics_R2_P1:+.2f}); " + verdict)
    return tab, ratio, verdict
