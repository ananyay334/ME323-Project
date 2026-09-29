"""Synthetic wear-track generator used to validate the screenshot -> wear pipeline.

Takes a real WLI screenshot, adds a parabolic groove of known size to the height
it encodes, and re-colours the map pixels (keeping the software's shading and
all overlays) - i.e. produces the screenshot the software *would* show for a
worn surface. Known answer: worn cross-section area = 2/3 * width * depth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
