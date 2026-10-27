"""Shared figure style: one colour per model / method everywhere, colour-blind-safe categorical palette
(validated: adjacent CVD dE >= 9 in light mode), hairline grid, no dual axes, fonts >= 12 pt at slide size."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots (reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a87", "#b8302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd"] + BLUE_RAMP)

# model -> (colour, marker, filled). Raw and hybrid versions of a model share the colour; hybrids are filled.
MODEL_STYLE = {
    "constant": (MUTED, "o", True),
    "physics": (INK, "s", True),
    "powerlaw": (YELLOW, "^", True),
    "ridge": (MAGENTA, "v", True),
    "svr": (GREEN, "P", True),
    "xgb": (VIOLET, "X", True),
    "ann": (BLUE, "o", False),
    "ann_phys": (BLUE, "o", True),
    "gpr": (ORANGE, "D", False),
    "gpr_phys": (ORANGE, "D", True),
    "tabpfn": (AQUA, "h", False),
    "tabpfn_phys": (AQUA, "h", True),
}
TARGET_MARKER = {"COF_first10s": "o", "COF_end": "s", "log10_V": "^", "log10_width": "D", "log10_k": "v"}
ROLE_STYLE = {"training": (BLUE, "o"), "test": (ORANGE, "s")}
PIPELINE_COLOR = {"original TA script": RED, "fixed TA copy": YELLOW, "ours": INK}
STUDY_COLOR = {"ours": INK, "paper": RED}
INPUT_COLOR = {"lnL": "#256abf", "lnf": MUTED, "lnt": AXIS}
INPUT_LABEL = {"lnL": "load", "lnf": "frequency", "lnt": "time"}

SLIDE = (13.333, 7.5)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 13,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.color": INK,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "svg.fonttype": "none",
    })


def model_scatter_kw(key: str, size: float = 70) -> dict:
    c, m, filled = MODEL_STYLE.get(key, (MUTED, "o", True))
    return {"marker": m, "s": size, "facecolors": c if filled else SURFACE, "edgecolors": c if not filled else SURFACE,
            "linewidths": 1.8 if not filled else 1.0}


def title(fig, text: str, sub: str | None = None) -> None:
    fig.suptitle(text, x=0.01, ha="left", y=0.985, fontsize=18, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 0.925, sub, ha="left", va="center", fontsize=13, color=INK2)


def save(fig, folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300)
    fig.savefig(folder / f"{name}.svg")
    plt.close(fig)
    return f"{name}.png"
