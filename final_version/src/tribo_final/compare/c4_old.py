"""C4 - Old pipeline vs final.

Old  = plain ANN ensemble on ln L, ln f, ln t for COF_end and log10 k (mean of the scan locations, plain
       levelling, every run with a positive area - no detection limit).
Final = the selected final model for COF_first10s and for log10 V (most-worn valid location, robust levelling,
        detected runs), plus k as one Archard constant.
Both scored under P1 (and P3 for the hold-out figure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate.interpret import noise_floor
from ..models.registry import TARGETS, target_frame
from .common import adhoc_target, cv_oof, score


def run_c4(cfg: dict, runs: pd.DataFrame, variants: pd.DataFrame, oof: pd.DataFrame, selection: pd.DataFrame,
           n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    d = runs.merge(variants[["run_id", "plain__k_mean_loc"]], on="run_id")
    d["k_old"] = d["plain__k_mean_loc"].where(d["plain__k_mean_loc"] > 0)
    olds = {
        "old_COF": (adhoc_target("COF_end", False, "cof", label="COF_end (old target)"), "COF_end"),
        "old_wear": (adhoc_target("k_old", True, "constant_k", label="log10 k, mean of locations, plain levelling"), "k_old"),
    }
    rows, parity = [], []
    for name, (t, _) in olds.items():
        df = target_frame(d, t)
        nf = noise_floor(df)
        sc = {}
        for prot in ("P1", "P2", "P3"):
            o = cv_oof(cfg, d, t, "ann", prot, n_jobs=n_jobs, df=df)
            o["pipeline"], o["target"] = "old", name
            parity.append(o)
            sc[prot] = score(o, nf["replicate_sd"])
        rows.append({"pipeline": "old", "quantity": "COF" if "COF" in name else "wear", "target": t.label,
                     "model": "ANN (plain)", **sc["P1"], "R2_P2": sc["P2"]["R2"], "R2_P3": sc["P3"]["R2"],
                     "replicate_sd": nf["replicate_sd"], "signal_to_noise": nf["signal_to_noise"]})
    sel = selection.set_index("target")
    for tk, q in (("COF_first10s", "COF"), ("log10_V", "wear")):
        mk = sel.at[tk, "final_model"]
        df = target_frame(runs, TARGETS[tk])
        nf = noise_floor(df)
        sc = {}
        for prot in ("P1", "P2", "P3"):
            o = oof[(oof.target == tk) & (oof.model == mk) & (oof.protocol == prot)].copy()
            o["pipeline"], o["target"] = "final", f"final_{q}"
            parity.append(o)
            sc[prot] = score(o, nf["replicate_sd"])
        rows.append({"pipeline": "final", "quantity": q, "target": TARGETS[tk].label, "model": mk, **sc["P1"],
                     "R2_P2": sc["P2"]["R2"], "R2_P3": sc["P3"]["R2"], "replicate_sd": nf["replicate_sd"],
                     "signal_to_noise": nf["signal_to_noise"]})
    tab = pd.DataFrame(rows)
    par = pd.concat(parity, ignore_index=True)
    oc, fc = tab[(tab.pipeline == "old") & (tab.quantity == "COF")].iloc[0], tab[(tab.pipeline == "final") & (tab.quantity == "COF")].iloc[0]
    ow, fw = tab[(tab.pipeline == "old") & (tab.quantity == "wear")].iloc[0], tab[(tab.pipeline == "final") & (tab.quantity == "wear")].iloc[0]
    verdict = (f"Replace the old pipeline. COF: the old ANN on end-of-run COF scores R² {oc.R2:+.2f} in P1 but "
               f"{oc.R2_P2:+.2f} on the training runs (P2); the final model on COF_first10s scores {fc.R2:+.2f} / "
               f"{fc.R2_P2:+.2f} with two parameters. Wear: R² {ow.R2:+.2f} / {ow.R2_P2:+.2f} (ANN on mean-of-locations k, "
               f"plain levelling, replicate SD {ow.replicate_sd:.2f} dex) -> {fw.R2:+.2f} / {fw.R2_P2:+.2f} (final model on the "
               f"most-worn-location volume, replicate SD {fw.replicate_sd:.2f} dex).")
    return tab, par, verdict
