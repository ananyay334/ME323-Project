"""BCR-F unicode height maps (.bcrf).

Header: UTF-16-LE, 2048 characters (4096 bytes) of `key = value` lines.
Data:   float32 little-endian, row-major (ny, nx), heights in um; values > 1e30 are void -> NaN.
The wear track always runs along x: each row is a line along the track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_BYTES = 4096


@dataclass
class HeightMap:
    z: np.ndarray  # (ny, nx) um, NaN = void
    pixel_um: float  # along x
    pixel_y_um: float
    meta: dict
    path: Path | None = None

    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def nx(self) -> int:
        return self.z.shape[1]


def read_bcrf(path: Path | str) -> HeightMap:
    path = Path(path)
    raw = path.read_bytes()
    head = raw[:HEADER_BYTES].decode("utf-16-le", errors="ignore").replace("\x00", "")
    meta: dict[str, str] = {}
    for line in head.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    nx, ny = int(meta["xpixels"]), int(meta["ypixels"])
    z = np.frombuffer(raw[HEADER_BYTES : HEADER_BYTES + nx * ny * 4], dtype="<f4").reshape(ny, nx).astype(float)
    z[~np.isfinite(z) | (z > 1e30)] = np.nan
    px = float(meta["xlength"]) / nx / 1000.0
    py = float(meta.get("ylength", meta["xlength"])) / ny / 1000.0
    return HeightMap(z=z, pixel_um=px, pixel_y_um=py, meta=meta, path=path)


def write_bcrf(path: Path | str, z: np.ndarray, pixel_um: float = 1.875) -> None:
    """Write a BCR-F unicode file (used by the tests)."""
    ny, nx = z.shape
    lines = [
        "fileformat = bcrf_unicode",
        "headersize = 2048",
        f"xpixels = {nx}",
        f"ypixels = {ny}",
        f"xlength = {nx * pixel_um * 1000:.4f}",
        f"ylength = {ny * pixel_um * 1000:.4f}",
        "voidpixels = 3.402823E+38",
        "bit2nm = 1",
        " xunit = nm",
        " yunit = nm",
        " zunit = um",
    ]
    text = "\n".join(lines) + "\n"
    text = text + " " * (2048 - len(text))
    data = np.where(np.isfinite(z), z, 3.402823e38).astype("<f4")
    Path(path).write_bytes(text.encode("utf-16-le") + data.tobytes())
