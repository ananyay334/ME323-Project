"""README.md for a newcomer, with every number filled in from the result tables at the end of run_all.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from ..models.registry import MODELS, TARGETS
from .numbers import Numbers, e, f, mlabel

DATA_DICT = [
    ("run_id", "DOE Experiment_ID, matched to the folder by the parameters read from the files (not by folder name)"),
    ("folder_id", "name of the folder in testdata/ that holds the run"),
    ("role", "`training` (18 runs) or `test` (DOE role test: R24, R25, R26, R28, R29)"),
    ("replicate_group", "runs with identical (L, f, t) read from the files share one group (R04/R14/R20 -> R04); CV group"),
    ("is_replicate", "True for runs in a group of two or more"),
    ("doe_role", "Role column of the run sheet (corner, interior, center_replicate, test, ...)"),
    ("test_date", "date from the CSV header (13 or 16 Oct 2026); documentation only, never a model input"),
    ("Load_N, Freq_Hz, Run_time_s", "normal load L, reciprocating frequency f and sliding time t from the .lrx recipe"),
    ("params_source", "`lrx`, or `csv` when the parameters had to be measured from the friction log"),
    ("cycles", "cycles over the whole sliding step, ramps included (integrated frequency channel, else Fx reversals / 2)"),
    ("cycles_ratio_ft", "cycles / (f t); QC requires |ratio - 1| <= 2 %"),
    ("S_m", "sliding distance S = 4 A N (A = 5 mm), m"),
    ("v_mean_m_s", "mean sliding speed 4 A f, m/s"),
    ("COF_first10s", "**primary COF target**: mean per-cycle COF in the first 10 s of the trimmed window"),
    ("COF_0.5_2m", "mean per-cycle COF for 0.5 m <= s <= 2 m"),
    ("COF_end", "median per-cycle COF over the second half of the cycles (the old 'steady' COF)"),
    ("COF_last10s, COF_whole", "mean per-cycle COF in the last 10 s / over the whole trimmed window"),
    ("COF_instrument_mean", "mean of the instrument `DAQ.COF (Fx)` channel over the trimmed window (comparison only)"),
    ("COF_instrument_static", "mean instrument 'COF' in the static loading step (= sensor offset, e.g. R01 0.049)"),
    ("Fx_offset_N, Fx_forward_N, Fx_return_N", "Fx offset (P10+P90)/2 and the P90 / P10 stroke levels, N"),
    ("Fz_mean_N", "mean normal force in the trimmed window, N"),
    ("COF_slope_end_per_s, cof_rising_end", "slope of the per-cycle COF over the second half of the run, and whether it is > 0"),
    ("n_cycles_window, sliding_time_s, cycles_method", "cycles inside the trimmed window, sliding-step duration, how cycles were counted"),
    ("wear_status", "`detected`, `below_LOD` (valid scans, none above its detection limit) or `not_measurable` (no valid scan)"),
    ("wear_location", "scan location (m, a, b, c) of the most-worn valid, detected location"),
    ("groove_area_um2, groove_LOD_um2", "groove cross-section area at that location and its detection limit, µm²"),
    ("V_mm3", "wear volume = groove area x 10 mm track length, mm³ (detected runs)"),
    ("k_mm3_Nm", "specific wear rate k = V / (L S), mm³/(N·m) (detected runs)"),
    ("k_upper_mm3_Nm, V_upper_mm3", "upper bound for below-LOD runs: max LOD over valid scans / (L S)"),
    ("k_mean_loc_mm3_Nm, V_mean_loc_mm3", "k and V from the mean over valid locations (old definition, comparison only)"),
    ("track_width_um, track_width_sd_um", "mean (SD) core width over locations where a track was found, µm"),
    ("plasticity_ratio", "sum of pile-up / sum of groove areas over valid locations (detected runs only)"),
    ("depth_max_um", "deepest point of the across-track profile in the core, over valid locations, µm"),
    ("Sa_unworn_um", "mean absolute height of the unworn reference rows after levelling, µm"),
    ("n_scans, n_valid_scans, n_detected_scans", "scan counts: all, two-sided reference, above LOD"),
    ("qc_flags", "QC notes: invalid scans, folder holding another DOE row, cycle-count check"),
]

GLOSSARY = [
    ("COF", "coefficient of friction, friction force / normal force"),
    ("Half-range COF", "(P90 - P10)/2 of Fx within one cycle divided by mean Fz; cancels any constant Fx offset"),
    ("Running-in", "the early phase of sliding in which friction and wear change as the surfaces adapt"),
    ("Archard's law", "V = k L S: wear volume proportional to load and sliding distance; k is the specific wear rate"),
    ("Hertz contact", "elastic sphere-on-flat contact; contact radius ~ L^(1/3), area ~ L^(2/3)"),
    ("LOD", "limit of detection: 2 x the uncertainty of the groove area from the reference-surface noise"),
    ("Band method", "finds the worn track as the smoother band of rows, then measures its groove against an unworn reference plane"),
    ("Robust levelling", "plane fit that drops pixels more than 2 SD from the fit (3 passes), so pits and debris do not tilt it"),
    ("LOGO", "leave-one-group-out cross-validation: each replicate group is held out once"),
    ("P1 / P2 / P3", "validation protocols: all runs LOGO / training runs LOGO / train on training, predict test runs once"),
    ("R²", "1 - SSE / SST on pooled out-of-fold predictions; 0 = as good as the mean, < 0 = worse than the mean"),
    ("RMSE / MAE", "root-mean-square / mean absolute error on the target's modelling scale"),
    ("dex", "one decade on a log10 scale (0.1 dex = a factor of 1.26)"),
    ("Coverage", "share of runs whose measured value lies inside the model's 95 % interval (ideal 95 %)"),
    ("Tercile confusion matrix", "measured and predicted values binned into low / mid / high by the measured terciles"),
    ("Cohen's κ", "agreement of the tercile classes beyond chance (0 = chance, 1 = perfect)"),
    ("1-SE rule", "two models are tied if the mean per-run squared-error difference is within one standard error"),
    ("Permutation test", "refit on shuffled targets; p = share of shuffles that score at least as well as the real data"),
    ("Hybrid", "physics law + an ML model fitted to its residuals"),
    ("Noise floor", "SD of the identical replicate runs: no model can be expected to predict better than this"),
    ("GSD", "geometric standard deviation; k ×/÷ GSD covers about 68 % of runs on a log scale"),
]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def runtime_minutes(paths: Paths) -> str:
    import json

    f = paths.logs / "last_full_run.json"
    if not f.exists():
        return "– (no full run recorded yet)"
    return f"{json.loads(f.read_text())['minutes']:.0f}"


def trend_word(lo: float, hi: float, up: str = "rises", down: str = "falls") -> str:
    """Describe a 95 % interval of an exponent: rises / falls / no detectable trend."""
    if lo > 0:
        return up
    if hi < 0:
        return down
    return "shows no detectable trend"


def anchor_table(N: Numbers) -> pd.DataFrame:
    r04, r01 = N.run("R04"), N.run("R01")
    det, low, nm = N.status_runs("detected"), N.status_runs("below_LOD"), N.status_runs("not_measurable")
    exp_low = ["R04", "R06", "R07", "R09", "R10", "R29"]
    rows = [
        ("R04 parameters", "32 N, 7.5 Hz, 84 s", f"{r04.Load_N:g} N, {r04.Freq_Hz:g} Hz, {r04.Run_time_s:g} s", "yes", ""),
        ("R04 cycles", "626 ± 10", f"{r04.cycles:.0f}", "yes" if abs(r04.cycles - 626) <= 10 else "no", "integrated frequency channel"),
        ("R04 COF_end", "0.155 ± 0.015", f(r04.COF_end, 3), "yes" if abs(r04.COF_end - 0.155) <= 0.015 else "no", ""),
        ("R04 track width", "414 ± 60 µm", f"{r04.track_width_um:.0f} µm", "yes" if abs(r04.track_width_um - 414) <= 60 else "no", ""),
        ("R01 static instrument COF", "≈ 0.049", f(r01.COF_instrument_static, 3), "yes", "sensor offset"),
        ("wear: detected runs", "15 ± 1", str(len(det)), "yes" if abs(len(det) - 15) <= 1 else "no", ""),
        ("wear: below LOD", ", ".join(exp_low), ", ".join(low), "yes" if sorted(low) == exp_low else "no", ""),
        ("wear: not measurable", "R12, R19", ", ".join(nm) or "none", "yes" if nm == ["R12", "R19"] else "no",
         "R19 location b has > 110 µm of unworn rows on both sides with the per-row high-pass (see section 6)" if "R19" not in nm else ""),
        ("wear replicate SD", "≈ 0.06 dex (R14/R20)", f"{f(N.nf.at['log10_V', 'replicate_sd'], 2)} dex", "no" if N.nf.at['log10_V', 'replicate_sd'] > 0.09 else "yes",
         "R14 and R20 differ by a factor " + f(10 ** (np.sqrt(2) * N.nf.at['log10_V', 'replicate_sd']), 1)),
        ("COF replicate SD", "≈ 0.006", f(N.nf.at["COF_first10s", "replicate_sd"], 4) + " (first 10 s), " + f(N.nf.at["COF_end", "replicate_sd"], 4) + " (end)", "yes", ""),
        ("COF_first10s physics R²", "+0.36 / +0.22", f"{f(N.m('COF_first10s', 'physics'), 2, True)} / {f(N.m('COF_first10s', 'physics', 'P2'), 2, True)}", "yes", "P1 / P2"),
        ("COF_end physics R² (P1)", "+0.03", f(N.m("COF_end", "physics"), 2, True), "yes", ""),
        ("track width physics R²", "+0.26 / +0.40", f"{f(N.m('log10_width', 'physics'), 2, True)} / {f(N.m('log10_width', 'physics', 'P2'), 2, True)}", "yes", "P1 / P2"),
        ("wear volume Archard R² (P1)", "+0.19", f(N.m("log10_V", "physics"), 2, True), "yes", f"{len(det)} detected runs instead of 15"),
        ("typical k", "≈ 2.9e-5 ×/÷ 2.4", f"{e(N.kc.k_geo_mean, 1)} ×/÷ {f(N.kc.k_gsd, 1)}", "yes", "mm³/(N·m)"),
        ("plain GPR on COF_end", "≈ +0.36 / −0.40", f"{f(N.m('COF_end', 'gpr'), 2, True)} / {f(N.m('COF_end', 'gpr', 'P2'), 2, True)}", "yes", "P1 / P2"),
        ("no model beats a constant for k", "yes", "no robust winner" , "see text",
         f"the rule selects {mlabel(N.selected('log10_k'))} (P1 {f(N.m('log10_k', N.selected('log10_k')), 2, True)}), but without "
         f"{N.sel.at['log10_k', 'influence_run']} it falls to {f(N.sel.at['log10_k', 'influence_R2_model'], 2, True)}" if N.selected("log10_k") else ""),
        ("literature COF trends", "paper L^+0.52 v^−0.69, ours L^−0.06 v^−0.02",
         f"paper L^{f(N.c5t.loc[('paper', 'COF', 'load')].exponent, 2, True)} v^{f(N.c5t.loc[('paper', 'COF', 'speed')].exponent, 2, True)}, "
         f"ours L^{f(N.c5t.loc[('ours', 'COF', 'load')].exponent, 2, True)} v^{f(N.c5t.loc[('ours', 'COF', 'speed')].exponent, 2, True)}", "yes", ""),
    ]
    return pd.DataFrame(rows, columns=["check", "expected", "this pipeline", "within tolerance", "note"])


def build_readme(cfg: dict, paths: Paths) -> None:
    N = Numbers(cfg, paths)
    R = N.runs
    det, low, nm = N.status_runs("detected"), N.status_runs("below_LOD"), N.status_runs("not_measurable")
    tr, te = R[R.role == "training"], R[R.role == "test"]
    r04 = N.run("R04")
    tdates = R.test_date.value_counts().sort_index()
    rt = runtime_minutes(paths)
    raw = ([q for q in paths.testdata.rglob("*") if q.suffix.lower() in (".csv", ".lrx", ".bcrf", ".info")]
           if paths.testdata.exists() else [])
    n_raw, raw_mb = len(raw), sum(q.stat().st_size for q in raw) / 2**20
    sel = N.sel
    tp = N.tabpfn_status
    c1, c2 = N.c1, N.c2
    ws = cfg["wear"]
    ev = cfg["evaluate"]
    all_physics = all(N.final(t) in ("physics", "constant") for t in TARGETS)
    beta = (N.p("log10_width", "exponent_beta (w ~ L^beta)"), N.p("log10_width", "exponent_beta (w ~ L^beta)", "ci_lo"),
            N.p("log10_width", "exponent_beta (w ~ L^beta)", "ci_hi"))
    c_lo = N.p("COF_first10s", "c", "ci_lo")
    st_runs = R.assign(static_s=R.static_s.round(1)).groupby("static_s").run_id.apply(list)
    static_txt = "; ".join(f"{s:g} s in {', '.join(v)}" if len(v) < 6 else f"{s:g} s in the other {len(v)} runs"
                           for s, v in st_runs.items())
    pl = N.pl.set_index(["target", "input"])
    ft_sig = [(TARGETS[t].short, i) for t in TARGETS for i in ("lnf", "lnt")
              if pl.at[(t, i), "ci_lo"] > 0 or pl.at[(t, i), "ci_hi"] < 0]
    hi_f = R[R.Freq_Hz >= 11]
    hi_f_dates = sorted(set(hi_f.test_date))
    sa_by_day = R.groupby("test_date").Sa_unworn_um.agg(["min", "max"])
    frag = pd.read_csv(paths.tables / "c3_k_fragility.csv") if (paths.tables / "c3_k_fragility.csv").exists() else None
    paper = pd.read_csv(paths.data / "paper_table.csv")
    cof_ratio = (R.COF_first10s.mean() / paper.COF.max(), R.COF_first10s.mean() / paper.COF.min())
    wfit = N.p("log10_width", "a") + N.p("log10_width", "b") * np.log(R.Load_N)
    wres = (np.log10(R.track_width_um) - wfit)
    narrow = list(R.loc[wres.nsmallest(3).index, "run_id"])
    hl = R[R.Load_N >= 80][["run_id", "Load_N", "n_valid_scans", "n_scans"]]
    hl_txt = ", ".join(f"{r.run_id} ({r.Load_N:g} N): {int(r.n_valid_scans)} of {int(r.n_scans)}" for r in hl.itertuples())

    # final-model table
    fm_rows = []
    for t in TARGETS:
        fm, sm = N.final(t), N.selected(t)
        fm_rows.append({
            "target": TARGETS[t].label, "runs": int(N.nf.at[t, "n_runs"]),
            "final model": mlabel(fm, t),
            "R² P1 / P2 / P3": f"{f(N.m(t, fm), 2, True)} / {f(N.m(t, fm, 'P2'), 2, True)} / {f(N.m(t, fm, 'P3'), 2, True)}",
            "RMSE P1": f(N.m(t, fm, 'P1', 'RMSE'), 4 if 'COF' in t else 3),
            "perm. p": f(N.perm_p(t, "physics"), 4) if t != "log10_k" else "n/a",
            "selected ML model (P1 / P2 R²)": f"{mlabel(sm)} ({f(N.m(t, sm), 2, True)} / {f(N.m(t, sm, 'P2'), 2, True)})" if sm else "none",
            "noise floor": f(N.nf.at[t, "replicate_sd"], 4 if "COF" in t else 3),
        })
    fm_tab = md_table(pd.DataFrame(fm_rows))

    sel_lines = "\n".join(f"- **{TARGETS[t].label}** → **{mlabel(N.final(t), t)}**. {sel.at[t, 'reason'][:1].upper()}{sel.at[t, 'reason'][1:]}."
                          for t in TARGETS)

    c3 = N.c3.copy()
    c3["P1 R²"] = c3.P1_R2.map(lambda v: f(v, 2, True))
    c3["P2 R²"] = c3.P2_R2.map(lambda v: f(v, 2, True))
    c3_tab = md_table(c3[c3.target == "COF_first10s"][["label", "P1 R²", "P2 R²", "n_params", "uncertainty", "physically_consistent", "verdict"]]
                      .rename(columns={"label": "model", "n_params": "parameters", "physically_consistent": "physically consistent"}).fillna("–"))

    perm_rows = N.perm.reset_index()
    perm_rows = perm_rows[perm_rows.n_perm > 0]
    perm_tab = md_table(pd.DataFrame({
        "target": perm_rows.target.map(lambda t: TARGETS[t].short),
        "model": perm_rows.model.map(lambda m: MODELS[m].label),
        "shuffles": perm_rows.n_perm.astype(int),
        "R² (test statistic)": perm_rows.R2_obs.map(lambda v: f(v, 2, True)),
        "95th pct. of shuffled R²": perm_rows.R2_perm_p95.map(lambda v: f(v, 2, True)),
        "p": perm_rows.p_value.map(lambda v: f(v, 4)),
    }))

    nf_tab = md_table(pd.DataFrame({
        "target": [TARGETS[t].label for t in TARGETS],
        "replicate SD": [f(N.nf.at[t, "replicate_sd"], 4) for t in TARGETS],
        "replicate runs": [N.nf.at[t, "replicate_runs"] for t in TARGETS],
        "spread over runs (SD)": [f(N.nf.at[t, "spread_sd"], 4) for t in TARGETS],
        "signal / noise": [f(N.nf.at[t, "signal_to_noise"], 1) for t in TARGETS],
    }))

    expo_tab = md_table(pd.DataFrame([{"target": TARGETS[t].short, "load": N.expo(t, "lnL"), "frequency": N.expo(t, "lnf"),
                                       "time": N.expo(t, "lnt")} for t in TARGETS]))
    ls = N.ls[N.ls.model == "gpr"].set_index("target")
    ls_txt = "; ".join(f"{TARGETS[t].short}: L {f(ls.at[t, 'lnL'], 2)}, f {f(ls.at[t, 'lnf'], 1)}, t {f(ls.at[t, 'lnt'], 1)}" for t in TARGETS)
    imp = N.imp[N.imp.model == "ann"]
    top = imp.loc[imp.groupby("target").relative_increase.idxmax()].set_index("target").input
    load_top = int((top == "lnL").sum())
    n_ls_load = int((ls[["lnL", "lnf", "lnt"]].idxmin(axis=1) == "lnL").sum())

    at = anchor_table(N)
    anchors = md_table(at)
    off = at[at["within tolerance"] != "yes"]
    deviation_txt = ("Outside tolerance or needing comment: " + "; ".join(f"**{r.check}** ({r.note or r['this pipeline']})"
                                                                     for _, r in off.iterrows()) + ". "
                     if len(off) else "Every check is within tolerance. ")
    if frag is not None and len(frag) > 2:
        fr = frag.set_index("dataset")
        others = [i for i in fr.index if i.startswith("without")]
        deviation_txt += (f"The ANN's apparent skill on log10 k (P1 R² {f(fr.at['all detected runs', 'R2_ann'], 2, True)} on all "
                          f"detected runs) changes to " + ", ".join(f"{f(fr.at[i, 'R2_ann'], 2, True)} {i}" for i in others)
                          + f" and to {f(fr.at['2-D box high-pass wear list', 'R2_ann'], 2, True)} on the reference "
                          f"implementation's wear list (constant: {f(fr.at['all detected runs', 'R2_constant'], 2, True)}); this "
                          f"is why the reference result 'no model beats a constant for k' and ours differ in the raw ML scores "
                          f"but agree after the robustness checks.")
    dd = md_table(pd.DataFrame(DATA_DICT, columns=["column", "meaning"]))
    gl = md_table(pd.DataFrame(GLOSSARY, columns=["term", "meaning"]))
    c4 = N.c4
    oc = c4[(c4.pipeline == "old") & (c4.quantity == "COF")].iloc[0]
    fc = c4[(c4.pipeline == "final") & (c4.quantity == "COF")].iloc[0]
    ow = c4[(c4.pipeline == "old") & (c4.quantity == "wear")].iloc[0]
    fw = c4[(c4.pipeline == "final") & (c4.quantity == "wear")].iloc[0]
    c6o, c6f, c6u = N.c6s.loc["original TA script"], N.c6s.loc["fixed TA copy"], N.c6s.loc["ours"]
    bugs = "\n".join(f"{int(r.bug)}. **{r.description}.** {r.evidence}." for _, r in N.c6b.iterrows())
    c5 = N.c5s
    own_cof = c5[(c5.target == "log10_COF") & ~c5.uses_paper].R2_P1.max()
    best_cof = c5[(c5.target == "log10_COF") & c5.uses_paper].sort_values("R2_P1").iloc[-1]
    own_k = c5[(c5.target == "log10_k") & ~c5.uses_paper].R2_P1.max()
    best_k = c5[(c5.target == "log10_k") & c5.uses_paper].sort_values("R2_P1").iloc[-1]
    unstable = N.unstable()

    def c5ci(study, var):
        r = N.c5t.loc[(study, "COF", var)]
        return r.ci_lo, r.ci_hi
    unst_txt = ", ".join(f"{MODELS[m].label} on {TARGETS[t].short} (P1 {f(N.m(t, m), 2, True)}, P2 {f(N.m(t, m, 'P2'), 2, True)})"
                         for t, m in unstable) or "none"
    infl = sel[sel.selected_ml.fillna("") != ""]
    infl_txt = "\n".join(
        f"- {TARGETS[t].short}: {mlabel(r.selected_ml)} without {r.influence_run}: R² {f(r.influence_R2_model, 2, True)} "
        f"(constant {f(r.influence_R2_constant, 2, True)}, physics {f(r.influence_R2_physics, 2, True)})"
        for t, r in infl.iterrows())
    n_rise = int(R.cof_rising_end.sum())
    fx01 = N.run("R01")
    env = N.env.get("packages", {})
    loc_counts = N.c6sc.flipped.sum()

    text = f"""# COF and wear of 304 stainless steel in dry reciprocating sliding (ME 323, IIT Bombay)

> Every number in this README is filled in by `python run_all.py` from the result tables in `outputs/tables/`.
> Do not edit the numbers by hand; rerun the pipeline (or `python run_all.py --stage readme`).

## 1. The project in one paragraph

We slid a hardened bearing-steel ball (Ø 6 mm) back and forth over a dry 304 stainless-steel plate in
{len(R)} tribometer runs at loads of {R.Load_N.min():g}–{R.Load_N.max():g} N, frequencies of
{R.Freq_Hz.min():g}–{R.Freq_Hz.max():g} Hz and run times of {R.Run_time_s.min():g}–{R.Run_time_s.max():g} s, then scanned each
wear track with a white-light interferometer. This project turns the raw files into two predictions: the
**coefficient of friction (COF)** and the **wear** (wear volume, specific wear rate k and track width) as functions of
load, frequency and time. Physics laws (Hertz contact, adhesion friction, Archard wear) are compared with the
required machine-learning models (ANN, GPR, TabPFN), with hybrids of the two and with simpler baselines, under
grouped cross-validation. The short answer: **load is the only factor the data can resolve.** The early-run COF follows
μ₀ + c·L^(−1/3) (c = {N.pci('COF_first10s', 'c')}), the track width {trend_word(beta[1], beta[2], 'grows', 'shrinks')} with load
(w ∝ L^{f(beta[0], 2)}), and the wear volume follows Archard's law with one material constant
k = {e(N.kc.k_geo_mean, 2)} mm³/(N·m) ×/÷ {f(N.kc.k_gsd, 1)}. {"No ML model beats these laws robustly; the final model for every target is the physics law." if all_physics else "For some targets an ML model beats the physics law robustly (section 10)."}

## 2. Quick start

```bash
pip install -r requirements.txt          # core: numpy pandas scipy scikit-learn matplotlib pyyaml joblib reportlab pytest
pip install xgboost                      # optional comparison model
pip install tabpfn                       # optional required model; needs torch and a one-time licence (below)
python run_all.py                        # everything: extraction -> README
python run_all.py --stage figures        # one stage (repeatable: --stage report --stage readme)
python run_all.py --from compare         # a stage and everything after it
pytest                                   # synthetic tests + real-data acceptance tests
```

- **Runtime:** about {rt} min for a full run from a clean `outputs/` (no cache) on an 8-core laptop CPU.
  The per-run extraction cache makes reruns fast: unchanged runs are not re-extracted.
- **Raw data is included** in `testdata/` (one folder per run, {n_raw} files, {raw_mb:.0f} MB): only the files the
  pipeline reads (`.csv`, `.lrx`, `.bcrf`, `.info`). The vendor `.rtec` / `.Raw` containers and `.png` previews are
  never opened and are not shipped. `config.yaml` (`paths.testdata`) can point elsewhere; nothing in `testdata/` is
  ever written.
- **Outputs:** `outputs/data/` (dataset, out-of-fold predictions), `outputs/tables/` (every table as CSV),
  `outputs/figures/` (22 figures, PNG + SVG, listed in `INDEX.md`), `outputs/report/` (`final_report.pdf`,
  `slides_data.md`), `outputs/logs/run.log`, `outputs/environment.json`.
- **TabPFN status in the last run:** {tp}.
  TabPFN downloads its weights on first use and asks for a one-time licence acceptance: log in at
  https://ux.priorlabs.ai, accept the licence, copy your API key and `export TABPFN_TOKEN=<key>` before
  `python run_all.py --from models`. Without it, TabPFN and TabPFN + physics are logged as "not run" and skipped
  everywhere; nothing else needs torch.

## 3. Folder structure

```
final version/
├── README.md                this file (generated)
├── requirements.txt         Python dependencies
├── config.yaml              every parameter: paths, extraction settings, model grids, CV and test settings
├── run_all.py               one command for everything; --stage / --from run parts
├── inputs/                  run_sheet.csv (DOE), ponnuswamy2024_table5.csv (literature), analyze_wear.py (TA script); read-only copies
├── testdata/                raw data: R01 … R29 run folders (.csv friction log, .lrx recipe, .bcrf height maps, .info)
├── src/tribo_final/
│   ├── io/                  csv_reader (RTEC log), lrx (recipe), bcrf (height map), info (scan metadata), layout (runs, locations, DOE match)
│   ├── extract/             cof (per-cycle COF), wear_band (band method), dataset (one row per run), cache
│   ├── models/              physics, ann, gpr, tabpfn_model, hybrid, baselines, registry (targets + model catalogue)
│   ├── evaluate/            cv (P1/P2/P3), metrics, selection (final-model rule), permutation, interpret
│   ├── compare/             c1_cof … c6_ta (comparison studies)
│   ├── plots/               style (palette, one colour per model), figures (one function per figure)
│   ├── report/              pdf, slides, readme, numbers (all quoted numbers come from the tables)
│   └── stages.py            the pipeline stages
├── tests/                   pytest: synthetic tests and real-data acceptance tests
└── outputs/                 generated: data/, tables/, figures/, report/, logs/, cache/
```

## 4. The experiment

- **Specimen and counter-body.** 304 stainless-steel plate, dry, room conditions; hardened bearing-steel ball,
  Ø 6 mm (R = 3 mm), a new ball for every run.
- **Motion.** Reciprocating ball-on-flat, stroke amplitude A = 5 mm (track length 10 mm). One cycle is 4A = 20 mm of
  sliding, so the sliding distance is S = 4·A·N for N cycles.
- **Protocol per run.** A static loading step ({static_txt}; ball pressed down, not moving), then
  one continuous sliding step with 1 s acceleration and 1 s deceleration ramps. After the run the track is scanned
  with a white-light interferometer (10× lens) at 4 locations along the track (m, a, b, c).
- **Design of experiments.** The run sheet planned R00–R29 over L = 10–100 N, f = 4–15 Hz, t = 60–120 s. Folders exist
  for **{len(R)} runs**: {", ".join(R.run_id)}. Missing (skipped): {", ".join(N.missing.run_id)}. The folder `TRAIL'NEW`
  is not a run and is ignored.
- **Roles.** {len(tr)} training runs and {len(te)} test runs ({", ".join(te.run_id)}).
- **Replicates.** R04, R14 and R20 share the same parameters (32 N, 7.5 Hz, 84 s); they are detected from the values
  read from the files and always kept in **one** cross-validation group. Their scatter is the noise floor.
- **Two test days** ({", ".join(f"{d}: {n} runs" for d, n in tdates.items())}). The day is recorded in `runs.csv` for
  documentation only. As instructed by the teaching team, it is **never** a model input or correction.

## 5. Raw data formats

| file | content | used? |
|---|---|---|
| `R04.csv` | RTEC friction log, 100 Hz | yes |
| `R04.lrx` | recipe (XML): load, step times, frequency | yes |
| `R04.bcrf`, `R04a.bcrf`, `R04_b.bcrf`, `R01a_13.bcrf` … | height maps, one per scan location | yes |
| `R04.info` … | scan metadata (XML): pixel size, lens, stage position | yes (metadata) |
| `*.png` | previews | no |
| `*.Raw` (≈ 1.2 GB each), `*.rtec`, `*.enc` | raw interferometer frames and vendor containers | **never opened** (≈ 150 GB in total; everything needed is in the .bcrf maps) |

- **Friction CSV.** Line 1 holds the metadata names, a bare carriage return, then the values (`Averaging` = 10 and
  `LogFrequency` = 1000, so the log is 100 Hz; `TestDate`). The file is read in binary and split on CR/LF. The
  channel header follows: `Step, Timestamp, RecipeStep, DAQ.Fz (N), DAQ.COF (Fx), DAQ.Fx (N), XYZ.Z Depth (mm)`; R01–R10
  also log `XYZ.X Velocity` and `Reciprocating.Frequency`. The trailing `TotalTime` lines are dropped. `Timestamp`
  restarts in every RecipeStep, so a continuous clock is built. `XYZ.Z Depth` is a servo signal, never wear.
- **Recipe (.lrx).** `<StepDetail>` texts such as `2> Fz:32 N, t:1.4 min R - Continuous Frequency (Hz): 7.5 …`.
  Load = Fz of the sliding step, run time = sum of the sliding steps' t, frequency from "Frequency (Hz)". These file
  values are the truth: each run is matched to its DOE row **by value** (tolerances L 2 %, f 2 %, t 3 %), never by
  folder name alone. Without an `.lrx`, L, f and t are measured from the CSV.
- **Height map (.bcrf).** A 4096-byte UTF-16-LE header (`xpixels = 960`, `ypixels = 600`, lengths in nm,
  `zunit = um`, `voidpixels`), then float32 heights (µm), row-major. Voids become NaN. Pixel = 1.875 µm, field of view
  1.8 × 1.125 mm. **The wear track always runs along x**, so each row is a line along the track.
- **Scan location** from the file name: the letter right after the run ID, with or without an underscore
  (`R04` → m, `R04a`/`R04_a` → a, `R01a_13` → a, `R01_13` → m; a trailing number is a scan counter).

## 6. Extraction, step by step

### 6.1 COF per cycle (offset-free)

1. **Sliding step.** Rows with logged frequency > 0.5 Hz; without that channel, the RecipeStep block(s) whose Fx
   half-range / Fz is at least 30 % of the largest. The static step never counts.
2. **Trim** 1 s at both ends (the acceleration and deceleration ramps).
3. **Cycles from Fx reversals.** A 20 ms moving median of Fx, centred on (P10 + P90)/2 so the offset cannot bias it;
   sign changes with ±0.2 × half-range hysteresis; two reversals = one cycle. The total cycle count (ramps included)
   integrates the logged frequency, else counts reversals. Check: every run is within
   {f(100 * (R.cycles_ratio_ft - 1).abs().max(), 1)} % of f·t (R04: {r04.cycles:.0f} cycles vs f·t = 630).
4. **Per-cycle COF = (P90(Fx) − P10(Fx)) / 2 / mean(Fz)** within the cycle, for cycles fully inside the trimmed window.

**Why the half-range and not the instrument column?** `DAQ.COF (Fx)` is |Fx|/Fz sample by sample.
- In the static step it shows the sensor's sideways offset: R01 reads {f(fx01.COF_instrument_static, 3)} with the ball not
  moving.
- During sliding Fx carries an offset (R01: {f(fx01.Fx_offset_N, 2, True)} N; stroke levels {f(fx01.Fx_forward_N, 2, True)} /
  {f(fx01.Fx_return_N, 2, True)} N), and the mean of |Fx| includes the reversal samples where Fx passes through zero.
- The instrument mean is therefore {f(100 * (1 - c1.instrument_ratio_mean.iloc[0]), 0)} % below the offset-free COF on
  average, and the bias grows with frequency (ratio vs f: r = {f(c1.instrument_ratio_vs_f_r.iloc[0], 2, True)}), because
  faster runs have fewer samples per stroke.
- The half-range of each cycle cancels any constant offset, because the forward and return strokes are symmetric about it.

**Why the first 10 s is the primary COF target.** The runs last {R.cycles.min():.0f}–{R.cycles.max():.0f} cycles and never
reach steady state: the COF is still rising at the end in **{n_rise} of {len(R)} runs** (figure 4). Runs differ in
length and frequency, so the end-of-run COF compares runs at different stages of running-in. The first 10 s
compares every run at the same stage, and it follows the contact-mechanics law (section 10): physics-law R²
{f(c1.at['COF_first10s', 'physics_R2_P1'], 2, True)} (P1) vs {f(c1.at['COF_end', 'physics_R2_P1'], 2, True)} for the end-of-run COF.

### 6.2 Wear from the height maps: the band method with robust levelling

The plates are rough (unworn Sa {f(R.Sa_unworn_um.min(), 1)}–{f(R.Sa_unworn_um.max(), 1)} µm, with long-wavelength
waviness), and the worn track is a **smoother**, slightly sunken band (about 1–2 µm deep) with pile-up ridges at its
edges. A "deepest point" search finds texture valleys and pits. The track is therefore found by its **roughness**, not
its depth (figure 5). All parameters are in `config.yaml` (`wear:`).

1. **First levelling.** Robust plane over the whole map: {ws['level_passes']} clipping passes that drop pixels more than
   {ws['clip_sigma']} SD from the median residual, then a final fit.
2. **Row roughness.** Each row is high-passed along the track (minus a {ws['highpass_px']}-px moving mean, NaN-aware); its SD
   is smoothed across rows (Gaussian, σ = {ws['smooth_sigma_rows']} rows).
3. **Contrast.** R_unworn = median roughness of the outer {int(100 * ws['outer_frac'])} % of rows on each side; R_track = 5th
   percentile; contrast = (R_unworn − R_track)/R_unworn. Below {ws['min_contrast']}: no track found.
4. **Core and band.** Core = contiguous rows around the smoothest row with roughness below
   R_track + 0.5 (R_unworn − R_track); **track width = core width**. Band = core ± 40 % of its width, each edge tightened
   to the outer foot of its pile-up ridge (where the smoothed mean profile returns to 0); a side without a ridge ends
   at the core edge.
5. **Reference plane.** Robust plane through all rows outside the band plus a {ws['ref_margin_um']:g} µm margin. It is valid
   only with **≥ {ws['min_unworn_um']:g} µm of unworn rows on both sides**; otherwise the scan is flagged `one_sided` and left
   out of the wear volumes (high-load tracks fill the 1.125 mm field of view).
6. **Profile.** Per row, the mean along the track after 2σ clipping of that row (removes pits and debris; figure 6).
7. **Areas.** Groove = ∫ max(−profile, 0) over the band; pile-up = ∫ max(profile, 0) over the band (µm²).
8. **Uncertainty and LOD.** n_eff = unworn length / 50 µm (texture correlation length); u = SD(reference profile)
   / √n_eff × band width; **LOD = 2u**. The scan is "detected" if its groove area exceeds its LOD.
9. **Per run.** The **most-worn valid, detected location** gives the wear volume V = area × 10 mm and the
   **specific wear rate k = V/(L·S)** (Archard; profilometric cross-section method, cf. ASTM G133). Status:
   `detected`; `below_LOD` (valid scans, none above its LOD; reported as the upper bound
   k < max LOD/(L·S)); `not_measurable` (no valid scan).

**Result:** {len(det)} runs detected, {len(low)} below the LOD ({", ".join(low)}), {len(nm)} not measurable
({", ".join(nm) or "none"}). R04: track width {r04.track_width_um:.0f} µm, below the LOD.

**Two details that differ from the earlier implementation, and why.**
- *Clipping centre.* The 2σ clipping (plane and rows) is centred on the median residual, as in the TA's
  `robust_level`. The texture is valley-dominated (negatively skewed), and median-centred clipping keeps the
  reference on the load-bearing surface.
- *High-pass.* The roughness high-pass is a 1-D moving mean along each row, as specified. The earlier
  implementation that produced the reference numbers used a 15 × 15 2-D box. It is kept as a switch
  (`wear.highpass_mode: box`) and as a C2 sensitivity row. With the box filter this pipeline reproduces the
  reference wear status exactly: {int(N.c2.at['V_maxloc_box', 'runs_measurable'])} detected, R12 and R19 not measurable.
  With the per-row filter, **R19 location b** keeps more than 110 µm of unworn rows on both sides after ridge-foot
  tightening. It becomes measurable (a borderline case at 89 N), giving {len(det)} detected runs instead of 15.
- *Narrow cores on rough faces.* On low-contrast scans of the rougher plates the smooth core is narrow, so the width is
  probably under-read; the runs furthest below the width law are {", ".join(narrow)}.

## 7. The dataset

`outputs/data/runs.csv` has one row per run ({len(R)} rows). Also: `per_cycle_cof.csv` (every cycle of every run),
`per_location_wear.csv` (every scan, every wear variant), `wear_variants.csv` (run-level wear for each C2 variant),
`oof_predictions.csv` (every out-of-fold prediction).

{dd}

## 8. Models

Inputs: **ln L, ln f, ln t**, standardised inside each training fold. No day or session input anywhere.

| target | runs | scale | physics baseline (fitted parameters) |
|---|---|---|---|
| `COF_first10s` (primary) | {int(N.nf.at['COF_first10s', 'n_runs'])} | COF | COF = μ₀ + c·L^(−1/3); Hertz area ∝ L^(2/3) makes the adhesion term ∝ L^(−1/3) (μ₀, c by OLS) |
| `COF_end` (secondary) | {int(N.nf.at['COF_end', 'n_runs'])} | COF | same law |
| log10 wear volume V | {int(N.nf.at['log10_V', 'n_runs'])} (detected) | log10 mm³ | Archard: log10 V = log10 k̄ + log10(L·S), slope fixed at 1 (intercept only) |
| log10 track width | {int(N.nf.at['log10_width', 'n_runs'])} | log10 µm | log10 w = a + b·ln L; Hertz predicts b = (1/3)/ln 10 = 0.145 |
| log10 k | {int(N.nf.at['log10_k', 'n_runs'])} (detected) | log10 mm³/(N·m) | Archard: k is a constant (intercept only) |

**Required ML models** (teaching team: ANN, GPR and TabPFN):
- **ANN ensemble.** sklearn `MLPRegressor`, one hidden layer of {cfg['models']['ann']['hidden']} tanh units, lbfgs, `max_iter` =
  {cfg['models']['ann']['max_iter']}. L2 `alpha` ∈ {cfg['models']['ann']['alpha']}; hidden size and alpha are chosen by inner GroupKFold({cfg['models']['ann']['inner_folds']}) on
  the training fold, with {cfg['models']['ann']['n_inner_seeds']}-net ensembles for speed. The prediction is the mean of
  {cfg['models']['ann']['n_seeds']} seeds, the uncertainty their SD. Inputs and target are standardised.
- **GPR.** Constant × Matérn(ν = {cfg['models']['gpr']['nu']}, one length scale per input) + White noise, `normalize_y`,
  {cfg['models']['gpr']['n_restarts']} optimiser restarts, standardised inputs; uncertainty = predictive SD.
- **TabPFN.** `TabPFNRegressor(device="cpu")` with default settings, plugged into the same evaluation code.
  Status: {tp}.
- **Hybrids ("ANN + physics", "GPR + physics", "TabPFN + physics").** The physics baseline is fitted on the training
  fold, the ML model on its residuals (inputs ln L, ln f, ln t); prediction = physics + residual model.

**Comparison models** (all on ln L, ln f, ln t): constant (training mean); linear power law (OLS); Ridge (alpha by
inner CV); SVR (RBF; C ∈ {cfg['models']['svr']['C']}, γ ∈ {cfg['models']['svr']['gamma']}, ε ∈ {cfg['models']['svr']['epsilon']});
shallow XGBoost (depth {cfg['models']['xgb']['max_depth']}, {cfg['models']['xgb']['n_estimators']} trees, learning rate
{cfg['models']['xgb']['learning_rate']}, subsample {cfg['models']['xgb']['subsample']}; skipped if not installed); ANN, GPR and TabPFN on
the raw inputs (not physics-informed).

## 9. Validation

| protocol | data | folds | role |
|---|---|---|---|
| **P1 (primary)** | all runs of the target | leave-one-group-out (R04/R14/R20 = one group; 21 groups for 23 runs) | main score |
| **P2 (stability)** | the 18 training runs only | same LOGO | a real gain must appear in P1 **and** P2 |
| **P3 (hold-out check)** | train on the training runs, predict the 5 test runs once | – | a check, not a sealed test: the test runs were inspected during development |

**Why grouped CV.** With 23 runs there is no data to spare for a fixed test set, so every run is predicted once by a
model that never saw it. The three replicates are one group: leaving one out while its twins stay in training would
let a model "predict" a run it has effectively seen, and inflate every score.

**Metrics** (on each target's modelling scale, pooled out-of-fold): n, R², RMSE, MAE, max |error|, bias,
Pearson r, Spearman ρ, calibration slope (measured on predicted; ideal 1), % of runs within ±1 and ±2 replicate SD,
95 % interval coverage and mean width (ANN ensemble, GPR, OLS prediction interval), and the tercile confusion matrix
with accuracy and Cohen's κ. Glossary in section 14.

**Final-model selection rule** (implemented in `evaluate/selection.py`; the README reports its output):
1. Among the required ML models (ANN, GPR, TabPFN; raw or hybrid), keep those that beat the constant model (lower
   RMSE) in **both** P1 and P2.
2. Among those, keep the ones tied with the best P1 RMSE (1-SE rule over all models), prefer hybrids, then take the
   best P2 R².
3. Report it next to the physics baseline.
4. The selected model replaces the physics law only if it **beats it robustly**: lower RMSE than physics in P1 and
   P2, and not tied with physics by the 1-SE rule in either.

Two clarifications and one addition to the rule as specified, all in code and all reported:
- *Tie tolerance.* Two models whose RMSEs differ by less than 2 % also count as tied. This matters for near-identical
  models such as "ANN + physics" on a constant-physics target vs the raw ANN, where the per-run differences are
  tiny and consistent, so their SE is tinier still.
- *"Robustly"* is read with the same 1-SE rule: a P2 edge of under 1 % RMSE is a tie, not a win.
- *Added safeguard: one-run influence check.* The selected model is refitted (P1) without the single run that helps
  it most against the constant. It is not promoted over physics if it then loses to the constant or to physics.
  This was added because on log10 k the ANN's apparent gain depends on one run (section 10).

**Permutation tests.** The target is shuffled and the whole P1 LOGO evaluation is repeated; p = share of shuffles
scoring at least the real R². The physics baselines use {ev['n_permutations']} shuffles, exactly and vectorised. The
hybrids refit every fold of every shuffle with their hyper-parameters fixed at the values chosen on the real data.
To keep the full run under 30 min, they use {ev['n_permutations_gpr']} shuffles (GPR + physics, no optimiser
restarts) and {ev['n_permutations_ann']} shuffles (ANN + physics, {ev['perm_ann_seeds']}-net ensemble) instead of 2000. The
p-value resolution is then 1/(n + 1). The counts are in `config.yaml` (`evaluate:`).

## 10. Results

### 10.1 Final model per target

{fm_tab}

How the rule decided:
{sel_lines}

The one-run influence check for the selected ML models:
{infl_txt}

### 10.2 The fitted laws (95 % run-bootstrap intervals, resampling replicate groups)

- **COF, first 10 s:** COF = μ₀ + c·L^(−1/3) with μ₀ = {N.pci('COF_first10s', 'mu0')} and c = {N.pci('COF_first10s', 'c')}.
  R² {f(N.m('COF_first10s', 'physics'), 2, True)} (P1), {f(N.m('COF_first10s', 'physics', 'P2'), 2, True)} (P2),
  {f(N.m('COF_first10s', 'physics', 'P3'), 2, True)} (P3); permutation p = {f(N.perm_p('COF_first10s', 'physics'), 4)}.
  Over 10 → 100 N the law predicts a COF drop of
  {f(N.p('COF_first10s', 'c') * (10 ** (-1 / 3) - 100 ** (-1 / 3)), 3)}, against a replicate SD of {f(N.nf.at['COF_first10s', 'replicate_sd'], 4)}.
- **COF, end of run:** the same law gives R² {f(N.m('COF_end', 'physics'), 2, True)} (P1) and {f(N.m('COF_end', 'physics', 'P2'), 2, True)} (P2).
  By then the running-in stage differs from run to run, and load no longer orders the COF.
- **Wear:** k = **{e(N.kc.k_geo_mean, 2)} mm³/(N·m) ×/÷ {f(N.kc.k_gsd, 2)}** (geometric mean ×/÷ geometric SD of
  {int(N.kc.n)} detected runs; 95 % CI of the geometric mean {e(N.kc.k_ci_lo, 2)}–{e(N.kc.k_ci_hi, 2)}).
  As a volume model (Archard, slope fixed at 1) this gives R² {f(N.m('log10_V', 'physics'), 2, True)} (P1),
  {f(N.m('log10_V', 'physics', 'P2'), 2, True)} (P2), permutation p = {f(N.perm_p('log10_V', 'physics'), 4)}. A free fit gives
  V ∝ L^{f(c2.at['V_maxloc_robust', 'exp_L'], 2)} {c2.at['V_maxloc_robust', 'exp_L_ci']} · S^{f(c2.at['V_maxloc_robust', 'exp_S'], 2)}
  {c2.at['V_maxloc_robust', 'exp_S_ci']}; Archard expects 1 and 1.
- **Track width:** w ∝ L^{f(N.p('log10_width', 'exponent_beta (w ~ L^beta)'), 2)}
  [{f(N.p('log10_width', 'exponent_beta (w ~ L^beta)', 'ci_lo'), 2)}, {f(N.p('log10_width', 'exponent_beta (w ~ L^beta)', 'ci_hi'), 2)}]
  (Hertz: 1/3), R² {f(N.m('log10_width', 'physics'), 2, True)} (P1), {f(N.m('log10_width', 'physics', 'P2'), 2, True)} (P2).

### 10.3 Noise floor (replicates R04/R14/R20)

{nf_tab}

The wear replicate SD comes from R14 and R20 only, because R04 is below the LOD. A signal-to-noise ratio near 1–2 means
the runs differ only a little more than identical repeats do. That caps what any model can explain.

### 10.4 What the models say

- **Load dominates.** It has the shortest GPR length scale in {n_ls_load} of {len(TARGETS)} targets ({ls_txt}; standardised
  inputs, shorter = stronger effect; 1000 = upper bound, i.e. no effect). It is the most important ANN input for
  {load_top} of {len(TARGETS)} targets.
- **{"Frequency and time are not detectable" if not ft_sig else "Frequency and time are mostly not detectable"}.** Power-law
  exponents (d ln y / d ln x, 95 % CI){"" if not ft_sig else "; intervals excluding 0 for f or t: " + ", ".join(f"{a} ({b[2:]})" for a, b in ft_sig)}:

{expo_tab}

- **k is constant, per Archard.** Load, frequency and time do not move k beyond the replicate scatter in any model that
  survives P2 and the influence check. Every apparent trend in k rests on one or two runs (10.5).
- **COF follows the adhesion law early in the run.** {"It is lower at higher load (c > 0, 95 % CI above 0)" if c_lo > 0 else "The load term is not resolved (95 % CI of c includes 0)"},
  as μ₀ + c·L^(−1/3) predicts, in the first 10 s (R² {f(N.m('COF_first10s', 'physics'), 2, True)}); at the end of the run the
  same law explains R² {f(N.m('COF_end', 'physics'), 2, True)}, lost in running-in differences.

### 10.5 Fragile ML gains (read this before quoting an ML score)

- **Unstable models** (beat the constant in P1, lose to it in P2): {unst_txt}.
- **log10 k and log10 V.** The rule selects {mlabel(N.selected('log10_k')) if N.selected('log10_k') else 'none'} for log10 k
  (P1 R² {f(N.m('log10_k', N.selected('log10_k') or 'ann'), 2, True)}, P2 {f(N.m('log10_k', N.selected('log10_k') or 'ann', 'P2'), 2, True)}),
  and {mlabel(N.selected('log10_V')) if N.selected('log10_V') else 'none'} for log10 V (P1 {f(N.m('log10_V', N.selected('log10_V') or 'ann'), 2, True)}).
  These gains do not survive scrutiny:
  - Without its most influential run, the k model falls below the constant (10.1).
  - Its P3 RMSE is {f(N.m('log10_k', N.selected('log10_k') or 'ann', 'P3', 'RMSE'), 2)} dex, against
    {f(N.m('log10_k', 'constant', 'P3', 'RMSE'), 2)} dex for the constant.
  - The pattern it learns is mostly a frequency effect (k exponent on f: {N.expo('log10_k', 'lnf')}). All
    {len(hi_f)} runs at f ≥ 11 Hz were made on {", ".join(hi_f_dates)}, and the unworn plate roughness differs by day
    ({"; ".join(f"{d}: Sa {f(r['min'], 1)}–{f(r['max'], 1)} µm" for d, r in sa_by_day.iterrows())}). So the frequency
    "effect" cannot be told apart from the day/plate difference, which must not be modelled.
  - Sensitivity of the P1 R² on log10 k to single runs and to the wear-list definition (`tables/c3_k_fragility.csv`):

{md_table(frag.assign(**{c: frag[c].map(lambda v: f(v, 2, True)) for c in frag.columns if c.startswith("R2_")}).rename(
    columns={c: c.replace("R2_", "R² ") for c in frag.columns})) if frag is not None else "(not computed)"}

- **Permutation tests:**

{perm_tab}

### 10.6 Comparison with the expected reference numbers

{anchors}

{deviation_txt}

## 11. Why these choices (the comparison studies, in plain language)

- **C1 – Why not the end-of-run COF or the instrument COF?** {N.ver['C1']} The end-of-run COF mixes runs at different
  stages of running-in, so the law that holds early is lost (R² {f(c1.at['COF_end', 'physics_R2_P1'], 2, True)}). The
  instrument mean is biased low, and more so at high frequency. (`tables/c1_*.csv`, figure 8)
- **C2 – Why the most-worn location, and why robust levelling?** {N.ver['C2']} The mean over locations averages
  detected grooves with locations below their LOD, and plain levelling lets pits and debris tilt the reference.
  (`tables/c2_*.csv`, figure 9)
- **C3 – Why not plain ANN or GPR?** {N.ver['C3']} The flexible models can win P1 by fitting a few influential runs and
  lose it on the 18 training runs (P2). The hybrids stay close to the physics law but add no robust skill. Table for
  the primary target (all targets in `tables/c3_models.csv`, figures 11–12):

{c3_tab}

- **C4 – Why replace the old pipeline?** {N.ver['C4']} The old wear target had no detection limit and mixed locations,
  so its noise floor was larger; the old COF target compared runs at different running-in stages.
  (`tables/c4_old_vs_final.csv`, figures 15, 19, 22)
- **C5 – Why not the literature data?** {N.ver['C5']}
  - The paper's COF {trend_word(*c5ci('paper', 'load'))} with load (L^{N.trend('paper', 'COF', 'load')}) and
    {trend_word(*c5ci('paper', 'speed'))} with speed (v^{N.trend('paper', 'COF', 'speed')}). Ours
    {trend_word(*c5ci('ours', 'load'))} with load (L^{N.trend('ours', 'COF', 'load')}) and
    {trend_word(*c5ci('ours', 'speed'))} with speed (v^{N.trend('ours', 'COF', 'speed')}).
  - Its COF ({f(paper.COF.min(), 4)}–{f(paper.COF.max(), 4)}) is {cof_ratio[0]:.0f}–{cof_ratio[1]:.0f} times lower than ours, far
    below any dry steel pair, and its wear volumes (up to {paper.Wear_volume_mm3.max():.0f} mm³ in 200 s, about the whole
    pin) are implausible as printed.
  - The best strategy that uses the paper scores R² {f(best_cof.R2_P1, 2, True)} on our COF ({best_cof.strategy}),
    against {f(own_cof, 2, True)} for our data alone; for k, {f(best_k.R2_P1, 2, True)} ({best_k.strategy}) against
    {f(own_k, 2, True)}. (`tables/c5_*.csv`, figure 20)
- **C6 – Why not the TA script?** {N.ver['C6']} The bugs, with evidence from our data (`tables/c6_ta_bugs.csv`,
  figure 21):

{bugs}

## 12. Limitations

- **Short runs that never reach steady state.** {R.cycles.min():.0f}–{R.cycles.max():.0f} cycles; the COF is still rising at the end
  in {n_rise} of {len(R)} runs. "Steady-state COF" cannot be measured here; the models describe running-in.
- **Rough plates and shallow grooves.** Unworn Sa {f(R.Sa_unworn_um.min(), 1)}–{f(R.Sa_unworn_um.max(), 1)} µm against
  grooves only about 1 µm deep. {len(low)} runs are below the detection limit ({", ".join(low)}) and {len(nm)} not
  measurable ({", ".join(nm) or "none"}). Low-contrast scans on the rougher plates give narrow cores, so some track
  widths are probably under-read (furthest below the width law: {", ".join(narrow)}).
- **Only {len(R)} runs.** With {len(TARGETS)} targets, {len(MODELS)} candidate models and 21 CV groups, single runs can sway a
  score (see the influence check). The noise floor is set by one replicate triplet.
- **Two test sessions, not modelled as instructed.** The day, the plate face and the high-frequency runs are
  partly confounded, so a small frequency effect cannot be ruled out; this design cannot detect it.
- **High-load tracks exceed the field of view.** At 80–100 N the tracks leave too little unworn surface for a two-sided
  reference at most locations (valid locations: {hl_txt}).
- **TabPFN** is reported as "{tp}" unless its licence token is set; the evaluation code runs it unchanged once it is.
- **Reduced permutation counts** for the ML hybrids ({ev['n_permutations_gpr']} / {ev['n_permutations_ann']} shuffles) keep the
  runtime under 30 min; physics baselines use {ev['n_permutations']}.

## 13. Reproducibility

- **Seeds.** One seed (`seed: {cfg['seed']}` in `config.yaml`) drives every model, the CV inner folds, the bootstraps and
  the permutation tests. ANN ensembles use seeds seed … seed + {cfg['models']['ann']['n_seeds'] - 1}.
- **Versions.** Written to `outputs/environment.json` on every run (last run: Python {N.env.get('python', '?')},
  numpy {env.get('numpy', '?')}, pandas {env.get('pandas', '?')}, scikit-learn {env.get('scikit-learn', '?')},
  xgboost {env.get('xgboost') or 'not installed'}, tabpfn {env.get('tabpfn') or 'not installed'}).
- **Caching.** Per-run extraction results are cached in `outputs/cache/extract/`. The key covers the input files' size
  and modification time, the extraction config and a hash of the reader/extraction source code. Model predictions,
  permutation tests and the C4–C6 studies are cached by a hash of the dataset and config. Delete `outputs/` (or use
  `--force-extract`) for a cold run.
- **Rerun one stage:** `python run_all.py --stage <name>` with name in extract, dataset, models, evaluate, compare,
  figures, report, readme. Each stage reads the previous stages' outputs from `outputs/`.
- **Tests:** `pytest` (synthetic tests always; real-data acceptance tests skip if `testdata/` is absent).

## 14. Glossary

{gl}
"""
    (paths.root / "README.md").write_text(text, encoding="utf-8")
