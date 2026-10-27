import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tribo_final.config import get_paths, load_config  # noqa: E402
from tribo_final.utils import quiet_warnings  # noqa: E402

quiet_warnings()


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def paths(cfg):
    return get_paths(cfg)


def write_rtec_csv(path: Path, *, f_hz=5.0, slide_s=30.0, static_s=2.0, fz=20.0, mu=0.15, offset=0.3,
                   with_freq=False, fs=100.0, spikes=True, seed=0) -> float:
    """Synthetic RTEC friction log (100 Hz): static step + one sliding step with 1 s ramps, Fx offset and
    reversal spikes. Returns the true number of cycles."""
    import numpy as np

    rng = np.random.default_rng(seed)
    dt = 1 / fs
    rows = []
    ns = int(static_s * fs)
    for i in range(ns):
        Fz = fz + rng.normal(0, 0.02)
        Fx = offset + rng.normal(0, 0.01)
        rows.append((1, (i + 1) * dt, 1, Fz, abs(Fx) / Fz, Fx, 0.5, 0.0))
    n = int(slide_s * fs)
    t = (np.arange(n) + 1) * dt
    freq = np.minimum(1.0, np.minimum(t, slide_s - t + dt)) * f_hz  # 1 s linear ramps
    freq = np.clip(freq, 0, None)
    phase = np.cumsum(freq) * dt
    v = np.cos(2 * np.pi * phase)
    sgn = np.tanh(8 * v)
    Fz = fz + 0.3 * np.sin(4 * np.pi * phase) + rng.normal(0, 0.03, n)
    Fx = offset + mu * Fz * sgn + rng.normal(0, 0.01, n)
    if spikes:
        rev = np.flatnonzero(np.diff(np.sign(v)) != 0) + 2
        rev = rev[rev < n]
        Fx[rev] += 0.25 * mu * fz * np.sign(sgn[rev])
    for i in range(n):
        rows.append((2, t[i], 2, Fz[i], abs(Fx[i]) / Fz[i], Fx[i], 0.5, freq[i]))
    meta_n = "Recipe, UpperSample, LowerSample, OtherInfo, Averaging, LogFrequency, Type, Radius,TestDate"
    meta_v = "TEST.Rx,,,,10,1000,Reciprocating,1,10/13/2026 3:36:41 AM"
    head = "Step, Timestamp, RecipeStep, DAQ.Fz (N),DAQ.COF (Fx),DAQ.Fx (N),XYZ.Z Depth (mm),"
    if with_freq:
        head += "XYZ.X Velocity (mm/s),Reciprocating.Frequency (Hz),"
    lines = []
    for r in rows:
        vals = list(r[:7]) + ([0.0, r[7]] if with_freq else [])
        lines.append(",".join(f"{x:.6g}" for x in vals) + ",")
    text = meta_n + "\r" + meta_v + "\r\n" + head + "\r\n" + "\n".join(lines) + "\n\nTotalTime\n00:00:32.0\n"
    path.write_bytes(text.encode())
    return float(phase[-1])


def synthetic_surface(seed: int, *, band=(220, 420), depth=1.0, sq=3.0, ridge=0.6, ny=600, nx=960, pit=False):
    """Rough plate (Sq ~ 3 um) with a smoother sunken band and pile-up ridges; returns (z, true_profile)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    r = gaussian_filter(rng.normal(size=(ny, nx)), (5, 5))
    r = r / r.std() * sq
    yy = np.arange(ny)
    prof = np.zeros(ny)
    z = r.copy()
    if band is not None:
        a, b = band
        smooth = gaussian_filter(rng.normal(size=(ny, nx)), (1.5, 1.5))
        smooth = smooth / smooth.std() * 0.4
        z[a:b] = -depth + smooth[a:b]
        prof[a:b] = -depth
        rid = ridge * (np.exp(-0.5 * ((yy - (a - 18)) / 9) ** 2) + np.exp(-0.5 * ((yy - (b + 18)) / 9) ** 2))
        rid[a:b] = 0
        prof += rid
        z += rid[:, None]
    yy2, xx2 = np.mgrid[0:ny, 0:nx]
    z = z + 0.004 * xx2 - 0.006 * yy2 + 5.0  # tilt and offset
    if pit:
        z[(yy2 - 120) ** 2 + (xx2 - 300) ** 2 < 25 ** 2] -= 25.0
    return z, prof
