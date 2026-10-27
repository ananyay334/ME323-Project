"""Pipeline stages: extract -> dataset -> models -> evaluate -> compare -> figures -> report -> readme.

Every stage reads what the earlier stages wrote to outputs/, so any stage can be rerun alone.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Paths
from .evaluate import interpret
from .evaluate.cv import PROTOCOLS, collect_oof, protocol_splits, run_tasks
from .evaluate.metrics import metrics_table, tercile_confusion
from .evaluate.permutation import perm_test_model, perm_test_physics
from .evaluate.selection import apply_influence_check, most_influential_run, select_final, tie_table, verdicts
from .extract.dataset import build_dataset, extract_all
from .models.registry import MODEL_ORDER, MODELS, TARGETS, make_model, model_available, target_frame
from .utils import LOG, save_table

MODEL_COST = {"ann": 30, "ann_phys": 30, "xgb": 8, "gpr": 3, "gpr_phys": 3, "tabpfn": 10, "tabpfn_phys": 10, "svr": 2}


# --------------------------------------------------------------------------------------------- helpers
def load_runs(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.data / "runs.csv")


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]


def _cached(paths: Paths, name: str, key: str, compute):
    f = paths.cache / f"{name}_{key}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
            LOG.info("%s: using cache %s", name, f.name)
            return obj
        except Exception:  # written by another Python / pandas version: recompute
            LOG.info("%s: cache %s unreadable here, recomputing", name, f.name)
    obj = compute()
    for old in paths.cache.glob(f"{name}_*.pkl"):
        old.unlink()
    with open(f, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


def availability() -> dict[str, str]:
    """Models that cannot run -> reason."""
    out = {}
    for k in MODEL_ORDER:
        ok, why = model_available(k)
        if not ok:
            out[k] = why
    return out


def rep_sd_by_target(runs: pd.DataFrame) -> dict[str, float]:
    return {k: interpret.noise_floor(target_frame(runs, t))["replicate_sd"] for k, t in TARGETS.items()}


def most_common_best(oof: pd.DataFrame, target: str, model: str) -> dict | None:
    s = oof[(oof.target == target) & (oof.model == model) & (oof.protocol == "P1")].drop_duplicates("fold")["best"]
    s = [x for x in s if x]
    if not s:
        return None
    return ast.literal_eval(Counter(s).most_common(1)[0][0])  # dict literal written by cv.collect_oof


# --------------------------------------------------------------------------------------------- stages
def stage_extract(cfg: dict, paths: Paths, force: bool = False) -> None:
    extract_all(cfg, paths, force=force)


def stage_dataset(cfg: dict, paths: Paths) -> None:
    ex = extract_all(cfg, paths)
    d = build_dataset(ex, cfg, paths)
    runs = d["runs"]
    # plasticity ratio is only meaningful where a groove was detected
    runs.loc[runs.wear_status != "detected", "plasticity_ratio"] = np.nan
    save_table(runs, paths.data / "runs.csv")
    save_table(d["per_cycle_cof"], paths.data / "per_cycle_cof.csv")
    save_table(d["per_location_wear"], paths.data / "per_location_wear.csv")
    save_table(d["wear_variants"], paths.data / "wear_variants.csv")
    save_table(d["missing_runs"], paths.data / "missing_runs.csv")
    save_table(runs, paths.tables / "runs.csv")
    st = runs.wear_status.value_counts().to_dict()
    LOG.info("dataset: %d runs, wear status %s", len(runs), st)


def stage_models(cfg: dict, paths: Paths) -> pd.DataFrame:
    runs = load_runs(paths)
    unavailable = availability()
    for k, why in unavailable.items():
        LOG.warning("model %s: %s", MODELS[k].label, why)
    key = hashlib.sha1(json.dumps({"runs": _hash_df(runs), "m": cfg["models"], "e": cfg["evaluate"], "s": cfg["seed"],
                                   "skip": sorted(unavailable), "v": 2}, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def compute():
        tasks = []
        for tk, t in TARGETS.items():
            df = target_frame(runs, t)
            for mk in MODEL_ORDER:
                if mk in unavailable:
                    continue
                fac = partial(make_model, mk, cfg, t, cfg["seed"])
                for prot in PROTOCOLS:
                    for fold, (tr, te) in enumerate(protocol_splits(df, prot)):
                        tasks.append({"target": tk, "model": mk, "protocol": prot, "fold": fold, "factory": fac,
                                      "df": df, "tr": tr, "te": te, "cost": MODEL_COST.get(mk, 1)})
        LOG.info("models: %d fit/predict tasks (%d targets x %d models x P1/P2/P3 folds)", len(tasks), len(TARGETS),
                 len(MODEL_ORDER) - len(unavailable))
        t0 = time.time()
        res = run_tasks(tasks, cfg.get("n_jobs", -1))
        LOG.info("models: done in %.0f s", time.time() - t0)
        return collect_oof(tasks, res, ["target", "model", "protocol"])

    oof = _cached(paths, "oof", key, compute)
    save_table(oof, paths.data / "oof_predictions.csv")
    save_table(pd.DataFrame([{"model": k, "label": MODELS[k].label, "family": MODELS[k].family,
                              "required": MODELS[k].required, "status": unavailable.get(k, "run")}
                             for k in MODEL_ORDER]), paths.tables / "model_availability.csv")
    return oof


def stage_evaluate(cfg: dict, paths: Paths) -> None:
    runs = load_runs(paths)
    oof = pd.read_csv(paths.data / "oof_predictions.csv")
    seed, ev = cfg["seed"], cfg["evaluate"]
    A_mm = cfg["experiment"]["stroke_amplitude_mm"]
    unavailable = availability()

    # noise floor
    nf = pd.DataFrame([{"target": k, **interpret.noise_floor(target_frame(runs, t))} for k, t in TARGETS.items()])
    save_table(nf, paths.tables / "noise_floor.csv")
    rep = dict(zip(nf.target, nf.replicate_sd))

    # metrics per model x target x protocol
    met = metrics_table(oof, rep)
    met["model_label"] = met.model.map(lambda k: MODELS[k].label)
    met = met.sort_values(["target", "protocol", "model"], key=lambda s: s.map(
        {k: i for i, k in enumerate(list(TARGETS) + PROTOCOLS + MODEL_ORDER)}))
    save_table(met, paths.tables / "metrics_all.csv")
    for p in PROTOCOLS:
        save_table(met[met.protocol == p], paths.tables / f"metrics_{p}.csv")

    # confusion matrices (long format)
    cm_rows = []
    for (tk, p, mk), g in oof.groupby(["target", "protocol", "model"]):
        if len(g) < 6:
            continue
        cm, acc, kap = tercile_confusion(g.y.to_numpy(), g.pred.to_numpy())
        for i in range(3):
            for j in range(3):
                cm_rows.append({"target": tk, "protocol": p, "model": mk, "measured": ["low", "mid", "high"][i],
                                "predicted": ["low", "mid", "high"][j], "count": int(cm[i, j]),
                                "accuracy": acc, "kappa": kap})
    save_table(pd.DataFrame(cm_rows), paths.tables / "confusion_matrices.csv")

    # 1-SE ties and the selection rule
    ties = pd.concat([tie_table(oof, met, tk, p) for tk in TARGETS for p in ("P1", "P2")], ignore_index=True)
    save_table(ties, paths.tables / "ties_1se.csv")
    sel_rows = []
    for tk, t in TARGETS.items():
        s = select_final(met, ties, tk, oof)
        if s["selected_ml"]:
            s = apply_influence_check(s, influence_check(cfg, runs, oof, t, s["selected_ml"]),
                                      s["physics_beats_constant_P1_P2"])
        sel_rows.append(s)
    sel = pd.DataFrame(sel_rows)
    save_table(sel, paths.tables / "selection.csv")
    LOG.info("selection:\n%s", sel[["target", "selected_ml", "final_model"]].to_string(index=False))

    # final parameters (physics on all runs, run-bootstrap CIs) + k as one material constant
    params = []
    for tk, t in TARGETS.items():
        df = target_frame(runs, t)
        p = interpret.physics_params_ci(t.physics, df, ev["n_bootstrap"], seed)
        p.insert(0, "target", tk)
        p.insert(1, "model", "physics")
        params.append(p)
    params = pd.concat(params, ignore_index=True)
    det = runs[runs.wear_status == "detected"]
    kc = interpret.k_material_constant(det.k_mm3_Nm.to_numpy(), det.replicate_group.to_numpy(), ev["n_bootstrap"], seed)
    save_table(params, paths.tables / "final_parameters.csv")
    save_table(pd.DataFrame([kc]), paths.tables / "k_material_constant.csv")

    # power-law exponents with bootstrap CIs
    pl = []
    for tk, t in TARGETS.items():
        e = interpret.power_law_exponents(target_frame(runs, t), t.log10, ev["n_bootstrap"], seed)
        e.insert(0, "target", tk)
        pl.append(e)
    save_table(pd.concat(pl, ignore_index=True), paths.tables / "power_law_exponents.csv")

    # interpretation: GPR length scales, ANN importance, physical consistency, final-model hyper-parameters
    t_int = time.time()
    jobs = []
    for tk, t in TARGETS.items():
        for mk in MODEL_ORDER:
            if mk in unavailable:
                continue
            fixed = most_common_best(oof, tk, mk) if mk in ("ann", "ann_phys") else None
            jobs.append((mk, tk, fixed))
    res = Parallel(n_jobs=cfg.get("n_jobs", -1))(
        delayed(_interpret_job)(cfg, runs, tk, mk, fixed, rep[tk], A_mm, ev["n_importance_repeats"], seed)
        for mk, tk, fixed in jobs)
    ls_rows = [r["ls"] for r in res if "ls" in r]
    imp_rows = [r["imp"] for r in res if "imp" in r]
    hyp_rows = [r["hyp"] for r in res if "hyp" in r]
    cons_rows = [r["cons"] for r in res]
    save_table(pd.DataFrame(ls_rows), paths.tables / "gpr_length_scales.csv")
    save_table(pd.concat(imp_rows, ignore_index=True), paths.tables / "ann_permutation_importance.csv")
    save_table(pd.DataFrame(cons_rows), paths.tables / "physical_consistency.csv")
    save_table(pd.DataFrame(hyp_rows), paths.tables / "final_hyper_parameters.csv")
    LOG.info("interpretation tables done in %.0f s", time.time() - t_int)

    # permutation tests (physics exact; hybrids refit with fixed hyper-parameters)
    key = hashlib.sha1(json.dumps({"runs": _hash_df(runs), "e": ev, "m": cfg["models"], "v": 1},
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]

    def compute_perm():
        rows = []
        for tk, t in TARGETS.items():
            df = target_frame(runs, t)
            if t.physics != "constant_k":
                rows.append({"target": tk, "model": "physics", "R2_P1_full_model": np.nan,
                             **perm_test_physics(t.physics, df, ev["n_permutations"], seed)})
            else:
                rows.append({"target": tk, "model": "physics", "p_value": np.nan, "n_perm": 0,
                             "note": "constant model: no fitted trend to test"})
            t0 = time.time()
            gfac = partial(_gpr_phys_fixed, cfg, t, seed, ev["perm_gpr_restarts"])
            n_gpr = ev.get("n_permutations_gpr", ev["n_permutations"])
            r = perm_test_model(gfac, df, n_gpr, seed, cfg.get("n_jobs", -1))
            rows.append({"target": tk, "model": "gpr_phys", **r,
                         "note": f"GPR optimiser restarts = {ev['perm_gpr_restarts']} inside the test, {n_gpr} shuffles"})
            LOG.info("permutation %s gpr_phys: %d shuffles in %.0f s", tk, n_gpr, time.time() - t0)
            t0 = time.time()
            fixed = most_common_best(oof, tk, "ann_phys")
            afac = partial(_ann_phys_fixed, cfg, t, seed, fixed, ev["perm_ann_seeds"])
            n_ann = ev.get("n_permutations_ann", ev["n_permutations"])
            r = perm_test_model(afac, df, n_ann, seed, cfg.get("n_jobs", -1))
            rows.append({"target": tk, "model": "ann_phys", **r,
                         "note": f"hyper-parameters fixed at {fixed}, {ev['perm_ann_seeds']}-net ensemble, {n_ann} shuffles"})
            LOG.info("permutation %s ann_phys: %d shuffles in %.0f s", tk, n_ann, time.time() - t0)
        return pd.DataFrame(rows)

    perm = _cached(paths, "perm", key, compute_perm)
    p1 = met[met.protocol == "P1"].set_index(["target", "model"])["R2"]
    perm["R2_P1_full_model"] = [p1.get((a, b), np.nan) for a, b in zip(perm.target, perm.model)]
    save_table(perm, paths.tables / "permutation_tests.csv")

    # C3 verdicts need everything above
    v = verdicts(met, ties, sel, unavailable)
    save_table(v, paths.tables / "verdicts.csv")


def _interpret_job(cfg, runs, tk, mk, fixed, rep_sd, A_mm, n_repeats, seed) -> dict:
    """Fit one model on all runs of a target; physical consistency, plus GPR length scales / ANN importance."""
    from .utils import quiet_warnings

    quiet_warnings()
    t = TARGETS[tk]
    df = target_frame(runs, t)
    fac = partial(make_model, mk, cfg, t, seed, fixed) if fixed else partial(make_model, mk, cfg, t, seed)
    m = interpret.fit_on_all(fac, df)
    out = {"cons": {"target": tk, "model": mk, "n_params": m.n_params(),
                    **interpret.physical_consistency(m, t.physics, df, rep_sd, A_mm)}}
    inner = m.residual if hasattr(m, "residual") else m
    if mk in ("gpr", "gpr_phys"):
        out["ls"] = {"target": tk, "model": mk, **interpret.gpr_length_scales(m)}
        if not inner.flat:
            out["hyp"] = {"target": tk, "model": mk, "hyper_parameters": str(inner.m.kernel_)}
    if mk in ("ann", "ann_phys"):
        imp = interpret.ann_importance(m, df, n_repeats, seed)
        imp.insert(0, "target", tk)
        imp.insert(1, "model", mk)
        out["imp"] = imp
        out["hyp"] = {"target": tk, "model": mk, "hyper_parameters": str(inner.best)}
    return out


def influence_check(cfg: dict, runs: pd.DataFrame, oof: pd.DataFrame, t, model: str) -> dict:
    """Re-run P1 for the selected model, the constant and physics without the run that helps the model most."""
    run = most_influential_run(oof, t.key, model)
    df = target_frame(runs, t)
    df = df[df.run_id != run].reset_index(drop=True)
    splits = protocol_splits(df, "P1")
    out = {"run": run}
    for mk in (model, "constant", "physics"):
        tasks = [{"model": mk, "fold": i, "factory": partial(make_model, mk, cfg, t, cfg["seed"]), "df": df, "tr": tr,
                  "te": te} for i, (tr, te) in enumerate(splits)]
        o = collect_oof(tasks, run_tasks(tasks, cfg.get("n_jobs", -1)), ["model"])
        e = o.pred - o.y
        name = "model" if mk == model else mk
        out[f"RMSE_{name}"] = float(np.sqrt(np.mean(e**2)))
        out[f"R2_{name}"] = float(1 - np.sum(e**2) / np.sum((o.y - o.y.mean()) ** 2))
    LOG.info("influence check %s / %s without %s: R2 %.2f (constant %.2f, physics %.2f)", t.key, model, run,
             out["R2_model"], out["R2_constant"], out["R2_physics"])
    return out


def _gpr_phys_fixed(cfg, t, seed, restarts):
    from .models.gpr import GPR
    from .models.hybrid import Hybrid
    from .models.physics import LinearPhysics

    return Hybrid(LinearPhysics(t.physics), GPR(cfg["models"]["gpr"], seed, n_restarts=restarts))


def _ann_phys_fixed(cfg, t, seed, fixed, n_seeds):
    from .models.ann import ANNEnsemble
    from .models.hybrid import Hybrid
    from .models.physics import LinearPhysics

    fixed = fixed or {"hidden": cfg["models"]["ann"]["hidden"][0], "alpha": cfg["models"]["ann"]["alpha"][-1]}
    return Hybrid(LinearPhysics(t.physics), ANNEnsemble(cfg["models"]["ann"], seed, fixed=fixed, n_seeds=n_seeds))
