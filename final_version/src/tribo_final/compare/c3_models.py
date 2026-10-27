"""C3 - Which model? All models x targets x P1/P2/P3, parameter count, uncertainty, physical consistency and a
verdict per model (from the selection rule)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.registry import MODEL_ORDER, MODELS, TARGETS


def run_c3(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    met = tables["metrics_all"]
    ver = tables["verdicts"].set_index(["target", "model"])["verdict"]
    cons = tables["physical_consistency"].set_index(["target", "model"])
    perm = tables["permutation_tests"].set_index(["target", "model"])["p_value"]
    avail = tables["model_availability"].set_index("model")["status"]
    rows = []
    for tk in TARGETS:
        for mk in MODEL_ORDER:
            row = {"target": tk, "model": mk, "label": MODELS[mk].label, "family": MODELS[mk].family,
                   "required": MODELS[mk].required}
            if avail.get(mk, "run") != "run":
                row["verdict"] = avail[mk]
                rows.append(row)
                continue
            for p in ("P1", "P2", "P3"):
                m = met[(met.target == tk) & (met.protocol == p) & (met.model == mk)]
                for c in ("R2", "RMSE"):
                    row[f"{p}_{c}"] = float(m[c].iloc[0]) if len(m) else np.nan
                if p == "P1" and len(m):
                    row["P1_coverage95_pct"] = float(m["coverage95_pct"].iloc[0])
                    row["P1_tercile_kappa"] = float(m["tercile_kappa"].iloc[0])
            row["n_params"] = cons.at[(tk, mk), "n_params"] if (tk, mk) in cons.index else ""
            row["uncertainty"] = "yes" if MODELS[mk].uncertainty else "no"
            pc = cons.at[(tk, mk), "physically_consistent"] if (tk, mk) in cons.index else np.nan
            row["physically_consistent"] = "yes" if pc is True or pc == 1 else ("no" if pc is False or pc == 0 else "–")
            row["perm_p"] = perm.get((tk, mk), np.nan)
            row["verdict"] = ver.get((tk, mk), "")
            rows.append(row)
    tab = pd.DataFrame(rows)
    sel = tables["selection"].set_index("target")
    parts = []
    for tk in TARGETS:
        s = sel.loc[tk]
        chosen = MODELS[s.selected_ml].label if isinstance(s.selected_ml, str) and s.selected_ml else "none"
        parts.append(f"{TARGETS[tk].short}: final = {MODELS[s.final_model].label if s.final_model != 'physics' else 'physics law'}"
                     f" (selected ML: {chosen})")
    verdict = "; ".join(parts) + "."
    return tab, verdict
