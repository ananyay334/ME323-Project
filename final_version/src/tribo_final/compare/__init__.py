"""Comparison studies C1-C6: each writes CSV tables (and figure data) and a one-line verdict."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..config import Paths
from ..models.registry import TARGETS
from ..utils import LOG, save_table
from .common import adhoc_target, cv_oof, score

QUESTIONS = {
    "C1": "Which COF definition?",
    "C2": "Which wear measure?",
    "C3": "Which model?",
    "C4": "Old pipeline vs final?",
    "C5": "Does the literature data help?",
    "C6": "Can the TA script be used?",
}


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def c6_summary(cfg: dict, per_run: pd.DataFrame) -> pd.DataFrame:
    """Replicate spread and model R² (P1) for the original TA script, the fixed copy and our pipeline."""
    rows = []
    rep = per_run[per_run.replicate_group == per_run.loc[per_run.run_id == "R04", "replicate_group"].iloc[0]]
    base = per_run.assign(role=per_run.role, Run_time_s=per_run.Run_time_s)
    for name, kcol, vcol, cofcol in (("original TA script", "orig_k", "orig_V_mm3", "orig_cof_overall_mean"),
                                     ("fixed TA copy", "fixed_k", "fixed_V_mm3", "fixed_cof"),
                                     ("ours", "ours_k", "ours_V_mm3", "COF_first10s")):
        kk = rep.set_index("run_id")[kcol]
        lk = np.log10(kk[kk > 0])
        d = base.assign(V=base[vcol].where(base[vcol] > 0), C=base[cofcol], wear_status="detected")
        d["replicate_group"] = per_run.replicate_group
        tV = adhoc_target("V", True, "archard", label="log10 V")
        tC = adhoc_target("C", False, "cof", label="COF")
        sV, sC = score(cv_oof(cfg, d, tV, "physics", "P1")), score(cv_oof(cfg, d, tC, "physics", "P1"))
        rows.append({"pipeline": name, **{f"k_{r}": kk.get(r, np.nan) for r in ("R04", "R14", "R20")},
                     "replicate_sd_log10k": lk.std(ddof=1) if len(lk) > 1 else np.nan,
                     "replicate_spread_factor": 10 ** (lk.max() - lk.min()) if len(lk) > 1 else np.nan,
                     "runs_with_volume": int((base[vcol] > 0).sum()), "archard_R2_P1": sV["R2"], "cof_law_R2_P1": sC["R2"]})
    return pd.DataFrame(rows)


def k_fragility(cfg: dict, runs: pd.DataFrame, variants: pd.DataFrame, selection: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    """Does an ML gain on log10 k survive small changes to the data? P1 R² of the raw ANN, the selected k model and
    the constant on: all detected runs; without each high-load run (>= 80 N); the 2-D box high-pass wear list."""
    from ..models.registry import TARGETS

    t = TARGETS["log10_k"]
    sel = selection.set_index("target").at["log10_k", "selected_ml"]
    models = ["constant", "ann"] + ([sel] if isinstance(sel, str) and sel and sel != "ann" else [])
    box = runs.drop(columns=["wear_status", "k_mm3_Nm"]).merge(
        variants[["run_id", "box__status", "box__k"]].rename(columns={"box__status": "wear_status", "box__k": "k_mm3_Nm"}),
        on="run_id")
    det = runs[runs.wear_status == "detected"]
    sets = {"all detected runs": runs}
    for rid in det[det.Load_N >= 80].run_id:
        sets[f"without {rid}"] = runs[runs.run_id != rid]
    sets["2-D box high-pass wear list"] = box
    rows = []
    for name, d in sets.items():
        row = {"dataset": name, "n": int((d.wear_status == "detected").sum())}
        for mk in models:
            row[f"R2_{mk}"] = score(cv_oof(cfg, d, t, mk, "P1", n_jobs=n_jobs))["R2"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_comparisons(cfg: dict, paths: Paths) -> None:
    from ..stages import _cached, _hash_df, availability
    from .c1_cof import run_c1
    from .c2_wear import run_c2
    from .c3_models import run_c3
    from .c4_old import run_c4
    from .c5_literature import run_c5
    from .c6_ta import bug_evidence, run_c6

    T = paths.tables
    n_jobs = cfg.get("n_jobs", -1)
    runs = pd.read_csv(paths.data / "runs.csv")
    variants = pd.read_csv(paths.data / "wear_variants.csv")
    per_loc = pd.read_csv(paths.data / "per_location_wear.csv")
    oof = pd.read_csv(paths.data / "oof_predictions.csv")
    tabs = {n: pd.read_csv(T / f"{n}.csv") for n in
            ("metrics_all", "verdicts", "physical_consistency", "permutation_tests", "model_availability", "selection")}
    verdicts = {}

    c1, c1r, verdicts["C1"] = run_c1(cfg, runs)
    save_table(c1, T / "c1_cof_definitions.csv")
    save_table(c1r, T / "c1_instrument_ratio.csv")
    LOG.info("C1: %s", verdicts["C1"])

    c2, c2e, verdicts["C2"] = run_c2(cfg, runs, variants)
    save_table(c2, T / "c2_wear_measures.csv")
    save_table(c2e, T / "c2_wear_exponents.csv")
    LOG.info("C2: %s", verdicts["C2"])

    c3, verdicts["C3"] = run_c3(tabs)
    save_table(c3, T / "c3_models.csv")
    key = _key(_hash_df(runs), _hash_df(variants), _hash_df(tabs["selection"]), cfg["models"], cfg["seed"], "fragv1")
    frag = _cached(paths, "kfrag", key, lambda: k_fragility(cfg, runs, variants, tabs["selection"], n_jobs))
    save_table(frag, T / "c3_k_fragility.csv")
    LOG.info("C3: %s", verdicts["C3"])

    key = _key(_hash_df(runs), _hash_df(variants), _hash_df(tabs["selection"]), cfg["models"], cfg["seed"], "c4v2")
    c4, c4o, verdicts["C4"] = _cached(paths, "c4", key, lambda: run_c4(cfg, runs, variants, oof, tabs["selection"], n_jobs))
    save_table(c4, T / "c4_old_vs_final.csv")
    save_table(c4o, paths.data / "c4_oof.csv")
    LOG.info("C4: %s", verdicts["C4"])

    paper_raw = pd.read_csv(paths.inputs / "ponnuswamy2024_table5.csv")
    key = _key(_hash_df(runs), _hash_df(paper_raw), cfg["models"], cfg["evaluate"], cfg["seed"], "c5v1")
    c5t, c5s, c5o, paper, verdicts["C5"] = _cached(paths, "c5", key, lambda: run_c5(cfg, runs, paper_raw, n_jobs))
    save_table(c5t, T / "c5_trends.csv")
    save_table(c5s, T / "c5_strategies.csv")
    save_table(c5o, paths.data / "c5_oof.csv")
    save_table(paper, paths.data / "paper_table.csv")
    LOG.info("C5: %s", verdicts["C5"])

    ta_file = paths.inputs / "analyze_wear.py"
    key = _key(_hash_df(runs), hashlib.sha1(ta_file.read_bytes()).hexdigest(), "c6v1")
    scans, per_run = _cached(paths, "c6", key, lambda: run_c6(cfg, paths, runs, per_loc, n_jobs))
    bugs = bug_evidence(scans, per_run, per_loc)
    summ = c6_summary(cfg, per_run)
    save_table(scans, T / "c6_ta_scans.csv")
    save_table(per_run, T / "c6_ta_runs.csv")
    save_table(bugs, T / "c6_ta_bugs.csv")
    save_table(summ, T / "c6_ta_summary.csv")
    s = summ.set_index("pipeline")
    o, f, u = s.loc["original TA script"], s.loc["fixed TA copy"], s.loc["ours"]
    verdicts["C6"] = (f"Do not use the TA script as written: {int(scans.flipped.sum())} of {len(scans)} scans are read "
                      f"along the track, and the R04/R14/R20 replicates spread {o.replicate_spread_factor:.0f}x in k "
                      f"(SD {o.replicate_sd_log10k:.2f} dex); fixed copy {f.replicate_sd_log10k:.2f} dex, ours "
                      f"{u.replicate_sd_log10k:.2f} dex (R14/R20 only). Archard R² (P1): original {o.archard_R2_P1:+.2f}, "
                      f"fixed {f.archard_R2_P1:+.2f}, ours {u.archard_R2_P1:+.2f}.")
    LOG.info("C6: %s", verdicts["C6"])

    save_table(pd.DataFrame([{"id": k, "question": QUESTIONS[k], "verdict": v} for k, v in verdicts.items()]),
               T / "comparison_verdicts.csv")
