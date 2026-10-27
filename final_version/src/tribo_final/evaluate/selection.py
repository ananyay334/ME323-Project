"""Final model selection rule (implemented here so the README reports the winner from the results).

1. Among the required ML models (ANN, GPR, TabPFN; raw or hybrid) keep those that beat the constant model
   (lower RMSE) in BOTH P1 and P2.
2. Among those, take the ones tied with the best P1 RMSE (1-SE rule over all models), prefer hybrids
   (physics-consistent), then the best P2 score (highest P2 R²).
3. Report it next to the physics baseline.
4. It "beats the physics baseline robustly" only if it has the lower RMSE in P1 AND P2 and is not tied with
   physics (1-SE rule) in either. Otherwise say so: the physics law stays the recommended equation and the ML
   models confirm the trend, quantify uncertainty and show there is no further structure.

Tie rule: tied if mean(d) <= SE(d) with d the per-run squared-error difference, or if the two RMSEs differ by
less than TIE_RMSE_TOL (practically identical predictions, e.g. a hybrid on a constant physics term vs the raw
model, where SE(d) itself is tiny).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.registry import MODELS
from .metrics import one_se_tie

TIE_RMSE_TOL = 0.02


def _sq_err(oof: pd.DataFrame, target: str, protocol: str, model: str) -> pd.Series:
    o = oof[(oof.target == target) & (oof.protocol == protocol) & (oof.model == model)]
    return o.set_index("run_id").eval("(pred - y) ** 2")


def pair_tie(oof: pd.DataFrame, target: str, protocol: str, model: str, ref: str) -> tuple[bool, float, float]:
    a, b = _sq_err(oof, target, protocol, model).align(_sq_err(oof, target, protocol, ref), join="inner")
    tied, dmean, dse = one_se_tie(a.to_numpy(), b.to_numpy())
    close = np.sqrt(a.mean()) <= (1 + TIE_RMSE_TOL) * np.sqrt(b.mean())
    return bool(tied or close), dmean, dse


def tie_table(oof: pd.DataFrame, metrics: pd.DataFrame, target: str, protocol: str = "P1") -> pd.DataFrame:
    m = metrics[(metrics.target == target) & (metrics.protocol == protocol)].set_index("model")
    if m.empty:
        return pd.DataFrame()
    best = m["RMSE"].idxmin()
    rows = []
    for k in m.index:
        tied, dmean, dse = pair_tie(oof, target, protocol, k, best)
        rows.append({"target": target, "protocol": protocol, "model": k, "best_model": best, "tied_with_best": tied,
                     "mean_sq_err_diff": dmean, "se_diff": dse})
    return pd.DataFrame(rows)


def select_final(metrics: pd.DataFrame, ties: pd.DataFrame, target: str, oof: pd.DataFrame) -> dict:
    def get(model, prot, col="RMSE"):
        s = metrics[(metrics.target == target) & (metrics.protocol == prot) & (metrics.model == model)][col]
        return float(s.iloc[0]) if len(s) else np.nan

    tie = ties[(ties.target == target) & (ties.protocol == "P1")].set_index("model")["tied_with_best"]
    available = set(metrics[(metrics.target == target) & (metrics.protocol == "P1")].model)
    c1, c2 = get("constant", "P1"), get("constant", "P2")
    ph1, ph2 = get("physics", "P1"), get("physics", "P2")
    required = [k for k in MODELS if MODELS[k].required and k in available]
    beats_const = [k for k in required if get(k, "P1") < c1 and get(k, "P2") < c2]
    tied = [k for k in beats_const if bool(tie.get(k, False))]
    pool = [k for k in tied if MODELS[k].family == "hybrid"] or tied
    chosen = max(pool, key=lambda k: get(k, "P2", "R2")) if pool else None
    beats_phys = False
    if chosen:
        beats_phys = all(get(chosen, p) < get("physics", p) and not pair_tie(oof, target, p, "physics", chosen)[0]
                         for p in ("P1", "P2"))
    # for log10 k the physics baseline IS a constant (Archard), so it counts as the physics answer
    phys_beats_const = (ph1 < c1 and ph2 < c2) or bool(np.isclose(ph1, c1) and np.isclose(ph2, c2))
    if chosen and beats_phys:
        final, reason = chosen, (f"{MODELS[chosen].label} beats the constant and, beyond the 1-SE tie, the physics "
                                 f"baseline in both P1 and P2, and is tied with the best P1 RMSE")
    elif chosen:
        final = "physics" if phys_beats_const else chosen
        reason = (f"{MODELS[chosen].label} is the selected ML model (beats the constant in P1 and P2, tied with the "
                  f"best P1 RMSE{', hybrid preferred' if MODELS[chosen].family == 'hybrid' else ''}) but does not beat "
                  f"the physics baseline robustly (lower RMSE and not 1-SE-tied in both P1 and P2)")
    else:
        final = "physics" if phys_beats_const else "constant"
        reason = ("no required ML model beats the constant model in both P1 and P2" if not beats_const else
                  "required ML models beat the constant in P1 and P2, but none is tied with the best P1 RMSE")
    best_p1 = ties[(ties.target == target) & (ties.protocol == "P1")]["best_model"].iloc[0] if len(tie) else ""
    return {
        "target": target,
        "best_P1_model": best_p1,
        "ml_beating_constant_P1_P2": ";".join(beats_const),
        "ml_tied_with_best": ";".join(tied),
        "selected_ml": chosen or "",
        "selected_ml_beats_physics_P1_P2": beats_phys,
        "physics_beats_constant_P1_P2": phys_beats_const,
        "final_model": final,
        "reason": reason,
    }


def most_influential_run(oof: pd.DataFrame, target: str, model: str, reference: str = "constant") -> str:
    """The run whose P1 prediction gains most over the reference: argmax(e_ref^2 - e_model^2)."""
    o = oof[(oof.target == target) & (oof.protocol == "P1")]
    a = o[o.model == model].set_index("run_id").eval("(pred - y) ** 2")
    b = o[o.model == reference].set_index("run_id").eval("(pred - y) ** 2")
    a, b = a.align(b, join="inner")
    return str((b - a).idxmax())


def apply_influence_check(sel: dict, check: dict, physics_beats_const: bool) -> dict:
    """Added safeguard (not in the original rule): the selected ML model must still beat the constant - and, to be
    promoted over physics, the physics baseline - in P1 after removing the single run that helps it most."""
    sel = dict(sel)
    sel.update({f"influence_{k}": v for k, v in check.items()})
    robust_const = check["RMSE_model"] < check["RMSE_constant"]
    robust_phys = check["RMSE_model"] < check["RMSE_physics"]
    sel["selected_ml_robust_one_run"] = bool(robust_const)
    if sel["final_model"] == sel["selected_ml"] and not (robust_const and robust_phys):
        sel["final_model"] = "physics" if physics_beats_const else "constant"
        sel["reason"] += (f"; but without its most influential run ({check['run']}) it no longer beats the "
                          f"{'constant' if not robust_const else 'physics baseline'} in P1, so the gain is not robust")
    elif not robust_const:
        sel["reason"] += f"; without its most influential run ({check['run']}) it no longer beats the constant in P1"
    return sel


def verdicts(metrics: pd.DataFrame, ties: pd.DataFrame, selection: pd.DataFrame, unavailable: dict[str, str]) -> pd.DataFrame:
    """One verdict per model x target for C3."""
    rows = []
    for target, sel in selection.set_index("target").iterrows():
        m1 = metrics[(metrics.target == target) & (metrics.protocol == "P1")].set_index("model")
        m2 = metrics[(metrics.target == target) & (metrics.protocol == "P2")].set_index("model")
        tie = ties[(ties.target == target) & (ties.protocol == "P1")].set_index("model")["tied_with_best"]
        for k, spec in MODELS.items():
            if k in unavailable:
                rows.append({"target": target, "model": k, "verdict": unavailable[k]})
                continue
            if k not in m1.index:
                continue
            r1, r2_ = m1.at[k, "RMSE"], m2.at[k, "RMSE"] if k in m2.index else np.nan
            c1, c2 = m1.at["constant", "RMSE"], m2.at["constant", "RMSE"]
            p1 = m1.at["physics", "RMSE"]
            if k == sel["selected_ml"]:
                v = "chosen (selected ML model)" + ("" if sel["selected_ml_beats_physics_P1_P2"] else "; not better than physics")
            elif k == "constant":
                v = "reference" + (" (final)" if sel["final_model"] == "constant" else "")
            elif k == "physics":
                v = "physics baseline" + (" (final model)" if sel["final_model"] == "physics" else "")
            elif r1 >= c1:
                v = "no better than constant"
            elif not (r2_ < c2):
                v = "unstable (beats constant in P1 only)"
            elif bool(tie.get(k, False)):
                v = "tied with best" + ("" if spec.required else " (not a required model)")
            elif r1 > p1:
                v = "worse than physics"
            else:
                v = "beats constant, not tied with best"
            rows.append({"target": target, "model": k, "verdict": v})
    return pd.DataFrame(rows)
