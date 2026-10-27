"""outputs/report/slides_data.md: one section per planned slide with the key numbers and the figure to use."""

from __future__ import annotations

from ..config import Paths
from ..models.registry import TARGETS
from .numbers import Numbers, e, f, mlabel


def build_slides(cfg: dict, paths: Paths) -> None:
    N = Numbers(cfg, paths)
    r04 = N.run("R04")
    det, low, nm = N.status_runs("detected"), N.status_runs("below_LOD"), N.status_runs("not_measurable")
    tr = N.runs[N.runs.role == "training"]
    s = []

    def slide(n, title, bullets, fig):
        s.append(f"## Slide {n} – {title}\n")
        s.extend(f"- {b}" for b in bullets)
        s.append(f"- **Figure:** {fig}\n")

    slide(1, "Question and pipeline", [
        "Predict COF and wear of 304 SS (dry, ball-on-flat, A = 5 mm) from load L, frequency f and time t",
        f"{len(N.runs)} runs with data (of 30 planned); missing: {', '.join(N.missing.run_id)}",
        "Models must include ANN, GPR and TabPFN; the test day is never an input",
    ], "fig01_pipeline")
    slide(2, "Test matrix", [
        f"L {N.runs.Load_N.min():g}–{N.runs.Load_N.max():g} N, f {N.runs.Freq_Hz.min():g}–{N.runs.Freq_Hz.max():g} Hz, "
        f"t {N.runs.Run_time_s.min():g}–{N.runs.Run_time_s.max():g} s",
        f"{len(tr)} training, {len(N.runs) - len(tr)} test runs; replicates R04/R14/R20 (32 N, 7.5 Hz, 84 s) = one CV group",
        f"cycles {N.runs.cycles.min():.0f}–{N.runs.cycles.max():.0f}; sliding distance {N.runs.S_m.min():.1f}–{N.runs.S_m.max():.1f} m",
    ], "fig02_test_matrix")
    slide(3, "COF extraction", [
        f"R04: {r04.cycles:.0f} cycles counted (f·t = 630), COF_end {f(r04.COF_end, 3)}",
        f"R01 static step: instrument 'COF' {f(N.run('R01').COF_instrument_static, 3)} = sensor offset only",
        f"instrument mean / offset-free COF = {f(N.c1.instrument_ratio_mean.iloc[0], 2)}; falls with f "
        f"(r = {f(N.c1.instrument_ratio_vs_f_r.iloc[0], 2, True)})",
        f"COF still rising at the end in {int(N.runs.cof_rising_end.sum())} of {len(N.runs)} runs",
    ], "fig03_raw_signal, fig04_cof_time")
    slide(4, "Which COF? (C1)", [
        f"COF_first10s: replicate SD {f(N.c1.at['COF_first10s', 'replicate_sd'], 4)}, physics R² P1 "
        f"{f(N.c1.at['COF_first10s', 'physics_R2_P1'], 2, True)} / P2 {f(N.c1.at['COF_first10s', 'physics_R2_P2'], 2, True)}",
        f"COF_end: physics R² P1 {f(N.c1.at['COF_end', 'physics_R2_P1'], 2, True)} / P2 {f(N.c1.at['COF_end', 'physics_R2_P2'], 2, True)}",
        N.ver["C1"],
    ], "fig08_c1_cof_definitions")
    slide(5, "Wear extraction", [
        f"band method: {len(det)} detected, {len(low)} below LOD ({', '.join(low)}), {len(nm)} not measurable ({', '.join(nm)})",
        f"R04 track width {f(r04.track_width_um, 0)} µm; unworn Sa {f(N.runs.Sa_unworn_um.min(), 1)}–{f(N.runs.Sa_unworn_um.max(), 1)} µm",
        f"wear replicate SD (R14/R20) {f(N.nf.at['log10_V', 'replicate_sd'], 2)} dex",
    ], "fig05_height_map, fig06_robust_levelling")
    slide(6, "Which wear measure? (C2)", [
        f"most-worn + robust: {int(N.c2.at['V_maxloc_robust', 'runs_measurable'])} runs, replicate SD "
        f"{f(N.c2.at['V_maxloc_robust', 'replicate_sd_dex'], 2)} dex, Archard R² P1 {f(N.c2.at['V_maxloc_robust', 'physics_R2_P1'], 2, True)}",
        f"old mean-of-locations, plain: replicate SD {f(N.c2.at['V_meanloc_plain', 'replicate_sd_dex'], 2)} dex, R² P1 "
        f"{f(N.c2.at['V_meanloc_plain', 'physics_R2_P1'], 2, True)}",
        f"V ~ L^{f(N.c2.at['V_maxloc_robust', 'exp_L'], 2)} {N.c2.at['V_maxloc_robust', 'exp_L_ci']} · "
        f"S^{f(N.c2.at['V_maxloc_robust', 'exp_S'], 2)} {N.c2.at['V_maxloc_robust', 'exp_S_ci']} (Archard 1, 1)",
    ], "fig09_c2_wear_measures")
    slide(7, "COF model", [
        f"COF_first10s = μ₀ + c·L^(−1/3): μ₀ = {N.pci('COF_first10s', 'mu0')}, c = {N.pci('COF_first10s', 'c')}",
        f"R² P1 {f(N.m('COF_first10s', 'physics'), 2, True)}, P2 {f(N.m('COF_first10s', 'physics', 'P2'), 2, True)}, "
        f"P3 {f(N.m('COF_first10s', 'physics', 'P3'), 2, True)}; permutation p = {f(N.perm_p('COF_first10s', 'physics'), 4)}",
        f"noise floor {f(N.nf.at['COF_first10s', 'replicate_sd'], 4)}, S/N {f(N.nf.at['COF_first10s', 'signal_to_noise'], 1)}",
        f"COF_end with the same law: R² P1 {f(N.m('COF_end', 'physics'), 2, True)}",
    ], "fig10_physics_fits")
    slide(8, "Wear model", [
        f"k = {e(N.kc.k_geo_mean, 2)} mm³/(N·m) ×/÷ {f(N.kc.k_gsd, 2)} (95 % CI {e(N.kc.k_ci_lo, 2)}–{e(N.kc.k_ci_hi, 2)}, n = {int(N.kc.n)})",
        f"Archard log10 V: R² P1 {f(N.m('log10_V', 'physics'), 2, True)}, P2 {f(N.m('log10_V', 'physics', 'P2'), 2, True)}, "
        f"permutation p = {f(N.perm_p('log10_V', 'physics'), 4)}",
        f"track width ~ L^{f(N.p('log10_width', 'exponent_beta (w ~ L^beta)'), 2)} "
        f"[{f(N.p('log10_width', 'exponent_beta (w ~ L^beta)', 'ci_lo'), 2)}, {f(N.p('log10_width', 'exponent_beta (w ~ L^beta)', 'ci_hi'), 2)}] "
        f"(Hertz 1/3); R² P1 {f(N.m('log10_width', 'physics'), 2, True)}, P2 {f(N.m('log10_width', 'physics', 'P2'), 2, True)}",
    ], "fig07_wear_per_run, fig10_physics_fits")
    slide(9, "Model comparison (C3)", [
        *(f"{TARGETS[t].short}: final {mlabel(N.final(t), t)}; selected ML {mlabel(N.selected(t)) if N.selected(t) else 'none'} "
          f"(P1 {f(N.m(t, N.selected(t)), 2, True) if N.selected(t) else '–'}, P2 {f(N.m(t, N.selected(t), 'P2'), 2, True) if N.selected(t) else '–'})"
          for t in TARGETS),
        f"TabPFN: {N.tabpfn_status}",
    ], "fig11_c3_models")
    slide(10, "Stability and fragility", [
        f"unstable (win P1, lose P2): {', '.join(f'{m} on {t}' for t, m in N.unstable()) or 'none'}",
        f"plain GPR on COF_end: P1 {f(N.m('COF_end', 'gpr'), 2, True)}, P2 {f(N.m('COF_end', 'gpr', 'P2'), 2, True)}",
        (f"selected k model {mlabel(N.selected('log10_k'))}: P1 {f(N.m('log10_k', N.selected('log10_k')), 2, True)}, without "
         f"{N.sel.at['log10_k', 'influence_run']} {f(N.sel.at['log10_k', 'influence_R2_model'], 2, True)} "
         f"(constant {f(N.sel.at['log10_k', 'influence_R2_constant'], 2, True)})") if N.selected("log10_k")
        else f"raw ANN on log10 k: P1 {f(N.m('log10_k', 'ann'), 2, True)}, P2 {f(N.m('log10_k', 'ann', 'P2'), 2, True)}",
    ], "fig12_stability")
    slide(11, "Final models: parity, residuals, uncertainty", [
        *(f"{TARGETS[t].short}: R² P1 {f(N.m(t, N.final(t)), 2, True)}, coverage {f(N.m(t, N.final(t), 'P1', 'coverage95_pct'), 0)} %, "
          f"tercile κ {f(N.m(t, N.final(t), 'P1', 'tercile_kappa'), 2, True)}" for t in TARGETS),
        f"ANN-ensemble coverage {N.coverage('ann')} (spread of seeds only); GPR {N.coverage('gpr')}",
    ], "fig13_parity, fig14_residuals, fig15_confusion, fig16_uncertainty")
    slide(12, "What the models say", [
        f"COF_first10s exponents: L {N.expo('COF_first10s', 'lnL')}, f {N.expo('COF_first10s', 'lnf')}, t {N.expo('COF_first10s', 'lnt')}",
        f"width exponents: L {N.expo('log10_width', 'lnL')}, f {N.expo('log10_width', 'lnf')}, t {N.expo('log10_width', 'lnt')}",
        "load dominates (shortest GPR length scale, largest ANN importance); f and t not detectable",
    ], "fig17_interpretation, fig18_surfaces")
    slide(13, "Old pipeline vs final (C4) and hold-out (P3)", [N.ver["C4"],
        f"P3 COF_first10s physics R² {f(N.m('COF_first10s', 'physics', 'P3'), 2, True)}; log10 V Archard {f(N.m('log10_V', 'physics', 'P3'), 2, True)}"],
        "fig19_c4_old_vs_final, fig22_p3_holdout")
    slide(14, "Literature data (C5)", [
        f"paper COF ~ L^{N.trend('paper', 'COF', 'load')} v^{N.trend('paper', 'COF', 'speed')}",
        f"ours COF ~ L^{N.trend('ours', 'COF', 'load')} v^{N.trend('ours', 'COF', 'speed')}",
        N.ver["C5"],
    ], "fig20_c5_literature")
    slide(15, "TA script (C6)", [*(f"bug {r.bug}: {r.evidence}" for _, r in N.c6b.head(3).iterrows()), N.ver["C6"]],
          "fig21_c6_ta_script")
    out = ["# Slide data", "", "Numbers copied from outputs/tables by `run_all.py` (stage report). Figures are in outputs/figures.", ""]
    (paths.report / "slides_data.md").write_text("\n".join(out + s) + "\n", encoding="utf-8")
