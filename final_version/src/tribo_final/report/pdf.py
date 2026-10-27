"""outputs/report/final_report.pdf (reportlab): title page, data summary, one model card per target, comparison
tables C1-C6 with verdicts, and every figure. Tables and figures, minimal prose."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..config import Paths
from ..models.registry import TARGETS
from .numbers import Numbers, e, f, mlabel

PAGE = landscape(A4)


def _styles():
    ss = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=20, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=14, spaceAfter=6, spaceBefore=6),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=12.5),
        "small": ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10),
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontSize=7.5, leading=9),
    }


def _t(s: str) -> str:
    """Characters the built-in PDF fonts lack, rewritten as reportlab markup."""
    from xml.sax.saxutils import escape

    return escape(str(s)).replace("₀", "<sub>0</sub>")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def df_table(df: pd.DataFrame, st, col_widths=None, max_rows=60) -> Table:
    df = df.head(max_rows)
    data = [[Paragraph(f"<b>{_t(c)}</b>", st["cell"]) for c in df.columns]]
    data += [[Paragraph(_t(_fmt(v)), st["cell"]) for v in row] for row in df.itertuples(index=False)]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceae4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def model_card(N: Numbers, tk: str, st) -> list:
    t = TARGETS[tk]
    fm, sm = N.final(tk), N.selected(tk)
    par = N.par[N.par.target == tk][["param", "estimate", "ci_lo", "ci_hi"]]
    eq = {"cof": f"{t.short} = μ₀ + c·L^(−1/3)", "archard": "log10 V = log10 k + log10(L·S)",
          "width": "log10 w = a + b·ln L", "constant_k": "log10 k = constant (Archard)"}[t.physics]
    rows = []
    keys = ["physics", fm, sm or "physics"] + ([] if t.physics == "constant_k" else ["constant"])
    for mk in dict.fromkeys("physics" if k == "constant" and t.physics == "constant_k" else k for k in keys):
        rows.append([mlabel(mk, tk)] + [_fmt(N.m(tk, mk, p, c)) for p in ("P1", "P2", "P3") for c in ("R2", "RMSE")]
                    + [_fmt(N.m(tk, mk, "P1", "coverage95_pct")), _fmt(N.perm_p(tk, mk))])
    met = pd.DataFrame(rows, columns=["model", "R² P1", "RMSE P1", "R² P2", "RMSE P2", "R² P3", "RMSE P3", "cov95 P1 %", "perm p"])
    s = N.sel.loc[tk]
    out = [Paragraph(f"Model card: {t.label}", st["h2"]),
           Paragraph(f"<b>Final model:</b> {_t(mlabel(fm, tk))} &nbsp; <b>Selected ML model:</b> {mlabel(sm) if sm else 'none'} "
                     f"&nbsp; <b>Equation:</b> {_t(eq)}", st["body"]),
           Paragraph(f"<b>Selection rule:</b> {_t(s.reason)}.", st["body"]),
           Paragraph(f"<b>Noise floor:</b> replicate SD {f(N.nf.at[tk, 'replicate_sd'], 4)} ({N.nf.at[tk, 'replicate_runs']}); "
                     f"spread over runs / replicate SD = {f(N.nf.at[tk, 'signal_to_noise'], 1)}; n = {int(N.nf.at[tk, 'n_runs'])} runs.",
                     st["body"]), Spacer(1, 4),
           df_table(par, st, [6 * cm, 3 * cm, 3 * cm, 3 * cm]), Spacer(1, 4), df_table(met, st), Spacer(1, 8)]
    if tk == "log10_k":
        out.insert(3, Paragraph(f"<b>k as one material constant:</b> {e(N.kc.k_geo_mean, 2)} mm³/(N·m) ×/÷ {f(N.kc.k_gsd, 2)} "
                                f"(95 % bootstrap CI of the geometric mean {e(N.kc.k_ci_lo, 2)} – {e(N.kc.k_ci_hi, 2)}).", st["body"]))
    return out


def _small_copy(png, folder, width: int = 2000):
    """Downscaled copy of a 300-dpi figure for the PDF (keeps the report a few MB)."""
    from PIL import Image as PILImage

    folder.mkdir(parents=True, exist_ok=True)
    out = folder / png.name
    with PILImage.open(png) as im:
        im.convert("RGB").resize((width, round(width * im.height / im.width)), PILImage.LANCZOS).save(out, optimize=True)
    return out


def build_pdf(cfg: dict, paths: Paths) -> None:
    N = Numbers(cfg, paths)
    st = _styles()
    doc = SimpleDocTemplate(str(paths.report / "final_report.pdf"), pagesize=PAGE, leftMargin=1.5 * cm,
                            rightMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                            title="ME 323 - COF and wear of 304 stainless steel", author="ME 323 project")
    W = PAGE[0] - 3 * cm
    el = [Spacer(1, 3 * cm), Paragraph("COF and wear of 304 stainless steel in dry reciprocating sliding", st["h1"]),
          Paragraph("ME 323, IIT Bombay - final report (generated by run_all.py)", st["h2"]),
          Paragraph(f"{len(N.runs)} runs, ball-on-flat (Ø 6 mm bearing-steel ball), A = 5 mm, L 10–100 N, f 4–15 Hz, t 60–120 s. "
                    f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", st["body"]), Spacer(1, 1 * cm)]
    summ = [[TARGETS[t].label, mlabel(N.final(t), t), mlabel(N.selected(t)) if N.selected(t) else "none",
             f(N.m(t, N.final(t)), 2, True), f(N.m(t, N.final(t), "P2"), 2, True)] for t in TARGETS]
    el += [df_table(pd.DataFrame(summ, columns=["target", "final model", "selected ML model", "R² P1", "R² P2"]), st),
           PageBreak()]
    # data summary
    el.append(Paragraph("Data summary", st["h1"]))
    cols = ["run_id", "role", "replicate_group", "test_date", "Load_N", "Freq_Hz", "Run_time_s", "cycles", "S_m",
            "COF_first10s", "COF_end", "wear_status", "wear_location", "k_mm3_Nm", "k_upper_mm3_Nm", "track_width_um"]
    el += [df_table(N.runs[cols], st), Spacer(1, 6),
           Paragraph(f"Missing DOE runs (no folder): {', '.join(N.missing.run_id)}. Wear: {len(N.status_runs('detected'))} detected, "
                     f"{len(N.status_runs('below_LOD'))} below LOD, {len(N.status_runs('not_measurable'))} not measurable.", st["body"]),
           PageBreak()]
    # model cards
    el.append(Paragraph("Final model cards", st["h1"]))
    for i, tk in enumerate(TARGETS):
        el += model_card(N, tk, st)
        if i % 2 == 1:
            el.append(PageBreak())
    el.append(PageBreak())
    # comparisons
    el.append(Paragraph("Comparison studies C1–C6", st["h1"]))
    T = paths.tables
    comp = [("C1", "c1_cof_definitions", ["label", "n", "mean", "replicate_sd", "signal_to_noise", "physics_R2_P1", "physics_R2_P2", "c_adhesion"]),
            ("C2", "c2_wear_measures", ["label", "runs_measurable", "replicate_sd_dex", "signal_to_noise", "best_model", "best_R2_P1", "best_R2_P2", "exp_L", "exp_L_ci", "exp_S", "exp_S_ci"]),
            ("C3", "c3_models", ["target", "label", "P1_R2", "P1_RMSE", "P2_R2", "P3_R2", "n_params", "uncertainty", "physically_consistent", "perm_p", "verdict"]),
            ("C4", "c4_old_vs_final", ["pipeline", "quantity", "target", "model", "n", "R2", "RMSE", "tercile_kappa", "replicate_sd", "signal_to_noise"]),
            ("C5", "c5_strategies", ["target", "strategy", "uses_paper", "n", "R2_P1", "RMSE_P1", "bias"]),
            ("C6", "c6_ta_summary", ["pipeline", "k_R04", "k_R14", "k_R20", "replicate_sd_log10k", "replicate_spread_factor", "runs_with_volume", "archard_R2_P1", "cof_law_R2_P1"])]
    from .. compare import QUESTIONS
    for cid, name, cols in comp:
        d = pd.read_csv(T / f"{name}.csv")
        el += [Paragraph(f"{cid} – {QUESTIONS[cid]}", st["h2"]), Paragraph(f"<b>Verdict:</b> {_t(N.ver[cid])}", st["body"]),
               Spacer(1, 3), df_table(d[[c for c in cols if c in d.columns]], st, max_rows=70), Spacer(1, 6)]
        if cid == "C5":
            el += [df_table(pd.read_csv(T / "c5_trends.csv"), st), Spacer(1, 6)]
        if cid == "C6":
            el += [df_table(N.c6b, st, [1.2 * cm, 7 * cm, W - 8.2 * cm]), Spacer(1, 6)]
        el.append(PageBreak())
    # figures
    el.append(Paragraph("Figures", st["h1"]))
    idx = paths.figures / "INDEX.md"
    rows = [l.split("|") for l in idx.read_text().splitlines() if l.startswith("| ") and ".png" in l] if idx.exists() else []
    for r in rows:
        fn = r[2].strip().split("](")[0].strip("[")
        p = paths.figures / fn
        if not p.exists():
            continue
        el += [Paragraph(f"Figure {r[1].strip()}: {_t(r[3].strip())}", st["small"]),
               Image(str(_small_copy(p, paths.cache / "pdf_figures")), width=W * 0.92, height=W * 0.92 * 7.5 / 13.333),
               Paragraph(f"Data: {r[4].strip()}", st["small"]),
               PageBreak()]
    doc.build(el)
