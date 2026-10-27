"""Targets and the model catalogue (one place that knows every model's name, family and factory)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ann import ANNEnsemble
from .baselines import XGBCV, Constant, PowerLaw, RidgeCV, SVRCV, xgb_available
from .gpr import GPR
from .hybrid import Hybrid
from .physics import PHYSICS_LABEL, LinearPhysics
from .tabpfn_model import TabPFN, tabpfn_status


@dataclass(frozen=True)
class TargetSpec:
    key: str
    column: str
    log10: bool
    subset: str  # all | detected
    physics: str
    label: str
    short: str
    primary: bool = False


TARGETS: dict[str, TargetSpec] = {
    "COF_first10s": TargetSpec("COF_first10s", "COF_first10s", False, "all", "cof", "COF, first 10 s (primary)", "COF first 10 s", True),
    "COF_end": TargetSpec("COF_end", "COF_end", False, "all", "cof", "COF, end of run (second-half median)", "COF end"),
    "log10_V": TargetSpec("log10_V", "V_mm3", True, "detected", "archard", "log10 wear volume V [mm³]", "log10 V"),
    "log10_width": TargetSpec("log10_width", "track_width_um", True, "all", "width", "log10 track width [µm]", "log10 width"),
    "log10_k": TargetSpec("log10_k", "k_mm3_Nm", True, "detected", "constant_k", "log10 k [mm³/(N·m)]", "log10 k"),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str  # reference | physics | comparison | required | hybrid
    required: bool  # one of the teaching team's required ML models (ANN, GPR, TabPFN; raw or hybrid)
    uncertainty: bool


MODELS: dict[str, ModelSpec] = {m.key: m for m in [
    ModelSpec("constant", "Constant", "reference", False, True),
    ModelSpec("physics", "Physics", "physics", False, True),
    ModelSpec("powerlaw", "Power law (OLS)", "comparison", False, True),
    ModelSpec("ridge", "Ridge", "comparison", False, False),
    ModelSpec("svr", "SVR", "comparison", False, False),
    ModelSpec("xgb", "XGBoost", "comparison", False, False),
    ModelSpec("ann", "ANN", "required", True, True),
    ModelSpec("gpr", "GPR", "required", True, True),
    ModelSpec("tabpfn", "TabPFN", "required", True, True),
    ModelSpec("ann_phys", "ANN + physics", "hybrid", True, True),
    ModelSpec("gpr_phys", "GPR + physics", "hybrid", True, True),
    ModelSpec("tabpfn_phys", "TabPFN + physics", "hybrid", True, True),
]}
MODEL_ORDER = list(MODELS)


def model_available(key: str) -> tuple[bool, str]:
    if key == "xgb":
        return (True, "ok") if xgb_available() else (False, "not run (install xgboost)")
    if key.startswith("tabpfn"):
        return tabpfn_status()
    return True, "ok"


def make_model(key: str, cfg: dict, target: TargetSpec, seed: int, fixed: dict | None = None):
    m = cfg["models"]
    inner = cfg["evaluate"]["inner_folds"]
    if key == "constant":
        return Constant()
    if key == "physics":
        return LinearPhysics(target.physics)
    if key == "powerlaw":
        return PowerLaw()
    if key == "ridge":
        return RidgeCV(m["ridge"], inner, seed)
    if key == "svr":
        return SVRCV(m["svr"], inner, seed)
    if key == "xgb":
        return XGBCV(m["xgb"], inner, seed)
    if key == "ann":
        return ANNEnsemble(m["ann"], seed, fixed=fixed)
    if key == "gpr":
        return GPR(m["gpr"], seed)
    if key == "tabpfn":
        return TabPFN(m["tabpfn"], seed)
    if key == "ann_phys":
        return Hybrid(LinearPhysics(target.physics), ANNEnsemble(m["ann"], seed, fixed=fixed))
    if key == "gpr_phys":
        return Hybrid(LinearPhysics(target.physics), GPR(m["gpr"], seed))
    if key == "tabpfn_phys":
        return Hybrid(LinearPhysics(target.physics), TabPFN(m["tabpfn"], seed))
    raise KeyError(key)


def model_label(key: str, target: TargetSpec | None = None) -> str:
    if key == "physics" and target is not None:
        return PHYSICS_LABEL[target.physics]
    return MODELS[key].label


def target_frame(runs: pd.DataFrame, t: TargetSpec) -> pd.DataFrame:
    """Rows usable for target t, with model inputs (ln L, ln f, ln t), physics columns and y."""
    d = runs.copy()
    if t.subset == "detected":
        d = d[d["wear_status"] == "detected"]
    y = d[t.column].astype(float)
    if t.log10:
        y = np.log10(y.where(y > 0))
    d = d.assign(y=y).dropna(subset=["y"])
    out = pd.DataFrame({
        "run_id": d["run_id"].to_numpy(),
        "role": d["role"].to_numpy(),
        "group": d["replicate_group"].to_numpy(),
        "lnL": np.log(d["Load_N"].to_numpy(float)),
        "lnf": np.log(d["Freq_Hz"].to_numpy(float)),
        "lnt": np.log(d["Run_time_s"].to_numpy(float)),
        "L": d["Load_N"].to_numpy(float),
        "log10_LS": np.log10(d["Load_N"].to_numpy(float) * d["S_m"].to_numpy(float)),
        "y": d["y"].to_numpy(float),
    })
    return out.reset_index(drop=True)
