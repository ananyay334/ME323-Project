"""Figure style and reusable plot helpers (matplotlib, Agg backend, PNG output).

Palette (validated): categorical slots in a fixed order, colour follows the entity
(model -> slot never changes between figures), one-hue sequential ramps for magnitude
(blue for means, orange for uncertainty), an ordinal blue ramp for the four carbon
levels, hairline solid grids, 2 px lines, markers >= 8 px with a surface ring, and a
legend whenever two or more series share an axis. Every plotted number is also in a
CSV next to the figures.
"""
from __future__ import annotations

from pathlib import Path

import functools

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde3d6", "#f9c2a8", "#f39d78", "#eb6834", "#c9501f", "#a03d16", "#7a2d0f"]
CMAP_MEAN = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
CMAP_SD = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)

# colour follows the entity: a model keeps its slot in every figure
MODEL_SLOT = {"gpr": 0, "mlp": 1, "kan": 2, "mt_mlp": 3, "mt_kan": 4, "tabpfn": 5,
              "xgb": 6, "rf": 7}
FS_SLOT = {"base": 0, "physics": 1}
STRATEGY_SLOT = {"variance": 0, "ucb": 1, "random": 2}


def model_color(name: str) -> str:
    return SLOTS[MODEL_SLOT.get(name, 7) % len(SLOTS)]


STYLE = {}


def styled(fn):
    """Run a figure function inside the package style (global rcParams are left untouched)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapper


def _init_style() -> None:
    STYLE.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        # a family list enables per-glyph fallback (→, Σ, φ, ±) to DejaVu Sans
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


_init_style()


def new_fig(nrows=1, ncols=1, w=None, h=None, **kw):
    """A pyplot-free Figure (no backend switching, safe in notebooks and worker processes)."""
    fig = Figure(figsize=(w or 4.2 * ncols, h or 3.4 * nrows))
    ax = fig.subplots(nrows, ncols, squeeze=False, **kw)
    return fig, ax


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def dots(ax, x, y, color, size=30, label=None, marker="o", zorder=3, alpha=1.0):
    """Markers >= 8 px with a 2 px surface ring (legible where they overlap)."""
    return ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors=SURFACE, linewidths=1.2,
                      label=label, zorder=zorder, alpha=alpha)


@styled
def bar_rmse(summary, target: str, label: str, path: Path) -> Path | None:
    """Mean ± std outer-fold RMSE per model, one bar per feature set (colour = feature set)."""
    s = summary[summary["target"] == target]
    if s.empty:
        return None
    order = s.groupby("model")["RMSE_mean"].min().sort_values().index.tolist()
    fss = [f for f in FS_SLOT if f in set(s["feature_set"])]
    fig, ax = new_fig(w=5.6, h=0.42 * len(order) + 1.2)
    ax = ax[0, 0]
    hgt = 0.8 / max(len(fss), 1)
    for i, fs in enumerate(fss):
        sub = s[s["feature_set"] == fs].set_index("model").reindex(order)
        ypos = np.arange(len(order)) + (i - (len(fss) - 1) / 2) * hgt
        ax.barh(ypos, sub["RMSE_mean"], height=hgt * 0.85, xerr=sub["RMSE_std"],
                color=SLOTS[FS_SLOT[fs]], label=fs, error_kw={"ecolor": INK2, "elinewidth": 0.8,
                                                              "capsize": 0})
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"outer-fold RMSE ({label}); bars = mean, whiskers = ± 1 sd")
    ax.set_title(f"{target}: group-CV error by model")
    if len(fss) > 1:
        ax.legend(title="feature set", loc="lower right")
    return save(fig, path)


def parity(ax, y, yhat, color, label=None, lo=None, hi=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    a = np.nanmin([y.min(), yhat.min()]) if lo is None else lo
    b = np.nanmax([y.max(), yhat.max()]) if hi is None else hi
    pad = 0.04 * (b - a)
    ax.plot([a - pad, b + pad], [a - pad, b + pad], color=AXIS, lw=1, zorder=1)
    dots(ax, y, yhat, color, size=22, label=label)
    ax.set_xlim(a - pad, b + pad)
    ax.set_ylim(a - pad, b + pad)
    ax.set_aspect("equal", adjustable="box")


def nice_levels(lo: float, hi: float, n: int = 10) -> np.ndarray:
    """Round contour levels spanning [lo, hi]."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-9
    return MaxNLocator(n).tick_values(lo, hi)
