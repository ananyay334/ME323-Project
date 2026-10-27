"""Collects every number quoted by the README, the slides and the PDF from the result tables (never typed in)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Paths
from ..models.registry import MODELS, TARGETS


def f(x, nd=2, sign=False) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def e(x, nd=1) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{nd}e}" if np.isfinite(x) else "–"


def mlabel(key: str, target: str | None = None) -> str:
    if key == "physics" and target:
        return {"cof": "μ₀ + c·L^(−1/3)", "archard": "Archard (slope 1)", "width": "a + b·ln L",
                "constant_k": "Archard constant k"}[TARGETS[target].physics]
    if key == "constant" and target == "log10_k":
        return "Archard constant k"
    return MODELS[key].label if key in MODELS else str(key)


class Numbers:
    def __init__(self, cfg: dict, paths: Paths):
        self.cfg, self.paths = cfg, paths
        T = paths.tables
        rd = lambda n: pd.read_csv(T / f"{n}.csv")  # noqa: E731
        self.runs = pd.read_csv(paths.data / "runs.csv")
        self.met = rd("metrics_all")
        self.sel = rd("selection").set_index("target")
        self.par = rd("final_parameters")
        self.kc = rd("k_material_constant").iloc[0]
        self.nf = rd("noise_floor").set_index("target")
        self.perm = rd("permutation_tests").set_index(["target", "model"])
        self.ver = rd("comparison_verdicts").set_index("id")["verdict"]
        self.c1 = rd("c1_cof_definitions").set_index("definition")
        self.c2 = rd("c2_wear_measures").set_index("measure")
        self.c3 = rd("c3_models")
        self.c4 = rd("c4_old_vs_final")
        self.c5s = rd("c5_strategies")
        self.c5t = rd("c5_trends").set_index(["study", "quantity", "variable"])
        self.c6s = rd("c6_ta_summary").set_index("pipeline")
        self.c6b = rd("c6_ta_bugs")
        self.c6r = rd("c6_ta_runs")
        self.c6sc = rd("c6_ta_scans")
        self.pl = rd("power_law_exponents")
        self.ls = rd("gpr_length_scales")
        self.imp = rd("ann_permutation_importance")
        self.avail = rd("model_availability").set_index("model")["status"]
        self.cons = rd("physical_consistency")
        self.missing = pd.read_csv(paths.data / "missing_runs.csv")
        envf = paths.outputs / "environment.json"
        self.env = json.loads(envf.read_text()) if envf.exists() else {}

    # ---------------------------------------------------------------- accessors
    def m(self, target: str, model: str, protocol: str = "P1", col: str = "R2") -> float:
        d = self.met[(self.met.target == target) & (self.met.model == model) & (self.met.protocol == protocol)]
        return float(d[col].iloc[0]) if len(d) else np.nan

    def p(self, target: str, param: str, col: str = "estimate") -> float:
        d = self.par[(self.par.target == target) & (self.par.param == param)]
        return float(d[col].iloc[0]) if len(d) else np.nan

    def pci(self, target: str, param: str, nd: int = 3) -> str:
        return f"{f(self.p(target, param), nd)} [{f(self.p(target, param, 'ci_lo'), nd)}, {f(self.p(target, param, 'ci_hi'), nd)}]"

    def perm_p(self, target: str, model: str) -> float:
        try:
            return float(self.perm.at[(target, model), "p_value"])
        except KeyError:
            return np.nan

    def final(self, target: str) -> str:
        return self.sel.at[target, "final_model"]

    def selected(self, target: str) -> str:
        s = self.sel.at[target, "selected_ml"]
        return s if isinstance(s, str) and s else ""

    def status_runs(self, status: str) -> list[str]:
        return list(self.runs[self.runs.wear_status == status].run_id)

    def run(self, rid: str) -> pd.Series:
        return self.runs.set_index("run_id").loc[rid]

    def unstable(self) -> list[tuple[str, str]]:
        v = self.c3[self.c3.verdict.fillna("").str.startswith("unstable")]
        return list(zip(v.target, v.model))

    def expo(self, target: str, inp: str) -> str:
        r = self.pl[(self.pl.target == target) & (self.pl.input == inp)].iloc[0]
        return f"{r.exponent:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"

    def trend(self, study: str, q: str, v: str) -> str:
        r = self.c5t.loc[(study, q, v)]
        return f"{r.exponent:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"

    @property
    def tabpfn_status(self) -> str:
        return self.avail.get("tabpfn", "run")

    def coverage(self, model: str) -> str:
        d = self.met[(self.met.model == model) & (self.met.protocol == "P1")].coverage95_pct
        return f"{d.min():.0f}-{d.max():.0f} %" if d.notna().any() else "–"
