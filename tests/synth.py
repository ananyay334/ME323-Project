"""Synthetic wear-track generator used to validate the screenshot -> wear pipeline.

Takes a real WLI screenshot, adds a parabolic groove of known size to the height
it encodes, and re-colours the map pixels (keeping the software's shading and
all overlays) - i.e. produces the screenshot the software *would* show for a
worn surface. Known answer: worn cross-section area = 2/3 * width * depth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image

from tribo_extract.wli import _colorbar_lut, _find_layout, _hue_cut, _unwrap


def add_groove(png_in, png_out, zmin, zmax, fov_um, width_um=220.0, depth_um=1.0,
               angle_deg=10.0, offset_um=0.0):
    rgb = np.asarray(Image.open(png_in).convert("RGB")).astype(float) / 255
    hsv = rgb_to_hsv(rgb)
    m, b = _find_layout(hsv)
    hue_bar, _, _ = _colorbar_lut(hsv, b)
    cut = _hue_cut(hue_bar)
    hb = _unwrap(hue_bar, cut)
    pos = np.linspace(0, 1, len(hb))
    hb_inc, pos_inc = (hb[::-1], pos[::-1]) if hb[0] > hb[-1] else (hb, pos)
    hb_inc = np.maximum.accumulate(hb_inc)

    sub = hsv[m["y0"]:m["y1"], m["x0"]:m["x1"]].copy()
    ny, nx = sub.shape[:2]
    dx, dy = fov_um[0] / nx, fov_um[1] / ny
    yy, xx = np.mgrid[0:ny, 0:nx]
    X, Y = (xx - nx / 2) * dx, -(yy - ny / 2) * dy          # y up, as displayed
    a = np.radians(angle_deg)
    d = -X * np.sin(a) + Y * np.cos(a) - offset_um            # distance across the track
    groove = np.where(np.abs(d) < width_um / 2, -depth_um * (1 - (2 * d / width_um) ** 2), 0.0)

    hp = _unwrap(sub[..., 0] * 360, cut)
    on_bar = (sub[..., 1] > 0.3) & (hp >= hb.min() - 2) & (hp <= hb.max() + 2)
    f0 = np.interp(hp, hb_inc, pos_inc)
    f1 = np.clip(f0 + groove / (zmax - zmin), 0, 1)
    new_h = np.mod(np.interp(f1, pos, hb), 360) / 360
    sub[..., 0] = np.where(on_bar, new_h, sub[..., 0])
    hsv[m["y0"]:m["y1"], m["x0"]:m["x1"]] = sub
    Image.fromarray((hsv_to_rgb(hsv) * 255).round().astype(np.uint8)).save(png_out)
    return 2 / 3 * width_um * depth_um


# ---------------------------------------------------------------------------
# 304 SS DOE: synthetic 1000 Hz tribometer CSV and height maps with known grooves
# ---------------------------------------------------------------------------
def tribo_csv(path, mu=0.15, fz=20.0, freq=10.0, step_s=60.0, amp_mm=5.0, offset_N=0.29,
              noise_N=0.03, fs=1000, ramp_s=1.0, spike_N=0.6, spike_s=0.003, position=False, seed=0):
    """RTEC-like export: approach (2 s), slide step, scan pause (3 s), slide step.

    Fx = mu*Fz*sign(v) + offset + noise + a short spike at every reversal; frequency ramps
    linearly over ``ramp_s`` at every start and stop. Returns the true cycles per step.
    """
    rng = np.random.default_rng(seed)
    blocks, true_cycles = [], []
    for rs, kind, dur in [(1, "idle", 2.0), (2, "slide", step_s), (3, "idle", 3.0), (4, "slide", step_s)]:
        t = np.arange(int(round(dur * fs))) / fs
        if kind == "slide":
            f_t = freq * np.clip(np.minimum(t, dur - t) / ramp_s, 0, 1)
            phi = 2 * np.pi * np.cumsum(f_t) / fs
            true_cycles.append(phi[-1] / (2 * np.pi))
            x = amp_mm * np.sin(phi)
            v = np.cos(phi) * (f_t > 0)
            fx = mu * fz * np.sign(v)
            rev = np.flatnonzero(np.diff(np.sign(v)) != 0) + 1
            for r in rev:                                     # static-friction overshoot
                fx[r:r + int(spike_s * fs)] += spike_N * np.sign(v[min(r + 1, len(v) - 1)])
        else:
            x, fx = np.zeros_like(t), np.zeros_like(t)
        fxm = fx + offset_N + rng.normal(0, noise_N, len(t))
        fzm = fz + rng.normal(0, 0.05, len(t))
        cols = {"Step": rs, "Timestamp": np.round(t, 4), "RecipeStep": rs, "DAQ.Fz (N)": fzm,
                "DAQ.Fx (N)": fxm, "XYZ.Z Depth (mm)": 1e-4 * rng.normal(size=len(t)),
                "DAQ.COF (Fx)": fxm / fzm}
        if position:
            cols["Recip.Position (mm)"] = x
        blocks.append(pd.DataFrame(cols))
    df = pd.concat(blocks, ignore_index=True)
    meta = "Type,Recipe,TestDate\rReciprocating,synthetic,2026-10-14\r\n"
    with open(path, "w", newline="") as fh:
        fh.write(meta)
        fh.write(",".join(df.columns) + ",\r\n")
        fh.write(df.to_csv(index=False, header=False, float_format="%.5g", lineterminator="\n"))
        fh.write("TotalTime\n00:02:08.00\n")
    return true_cycles


def height_map(shape=(188, 300), px_um=3.0, width_um=0.0, depth_um=0.0, angle_deg=0.0,
               offset_um=0.0, seed=0, shift_px=(0, 0)):
    """Rough flat (smooth texture + tilt) with an optional parabolic groove of known size.

    Same ``seed`` = same surface texture (so baseline subtraction cancels it). Returns
    ``(z_um, true_area_um2)`` with area = 2/3 * width * depth.
    """
    from scipy import ndimage as ndi
    ny, nx = shape
    rng = np.random.default_rng(seed)
    pad = 20
    tex = ndi.gaussian_filter(rng.normal(size=(ny + 2 * pad, nx + 2 * pad)), 3)
    tex = 0.08 * tex / tex.std()
    tex = np.roll(np.roll(tex, shift_px[0], 0), shift_px[1], 1)[pad:pad + ny, pad:pad + nx]
    yy, xx = np.mgrid[0:ny, 0:nx]
    X, Y = (xx - nx / 2) * px_um, -(yy - ny / 2) * px_um
    a = np.radians(angle_deg)
    d = -X * np.sin(a) + Y * np.cos(a) - offset_um
    groove = np.where(np.abs(d) < width_um / 2, -depth_um * (1 - (2 * d / max(width_um, 1e-9)) ** 2), 0.0)
    z = 20.0 + 0.002 * X + 0.001 * Y + tex + groove
    return z, 2 / 3 * width_um * depth_um
