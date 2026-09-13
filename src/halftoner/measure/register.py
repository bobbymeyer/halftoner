"""Misregistration: how far each plate sits from the key plate, and how it walks.

Screen phase can't tell you registration (the screen origin is arbitrary), but
the image content on each plate can. Each ink's amount map is descreened, then
phase-correlated against the key ink in overlapping tiles. A press walks
smoothly, so the tile readings are fitted with a plane: its value at the sheet
center is the offset, its departure at the corners is the walk, and the fit's
covariance gives the walk a standard error so an unresolvable walk says so.

This needs shared edges between plates (a duotone with shapes, a keyline,
overlapping forms). Smooth tones alone mislead correlation: plates screened at
different angles quantize a soft gradient slightly differently, which reads as
a phantom offset. Tiles with a weak correlation peak are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..canvas import MM_PER_INCH
from .common import downsample_linear

MIN_SHARPNESS = 10.0
OVERLAP_SE_FACTOR = 2.0  # half-overlapping tiles: ~1/4 as many independent readings, so sqrt(4)
MIN_FIT_TILES = 5


@dataclass
class RegistrationMeasurement:
    dx_mm: float  # offset at the sheet center
    dy_mm: float
    drift_mm: float  # largest departure of the fitted walk from the center offset, at the sheet corners
    noise_mm: float = 0.0  # RMS of tile readings about the fitted walk
    tiles: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x_mm, y_mm, dx_mm, dy_mm)
    drift_se_mm: float = float("inf")  # standard error of the walk at the corners

    @property
    def magnitude_mm(self) -> float:
        return float(np.hypot(self.dx_mm, self.dy_mm))

    @property
    def walk_resolved(self) -> bool:
        """The walk stands clear of what tile scatter alone would produce."""
        return self.drift_mm > 2 * self.drift_se_mm


def phase_correlate(a: np.ndarray, b: np.ndarray, lowpass_sigma_px: float = 0.0) -> tuple[float, float, float]:
    """(dy, dx, sharpness) such that b(p) ~ a(p - d). Band-limited so descreened content drives it."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    Fa = np.fft.fft2((a - a.mean()) * win)
    Fb = np.fft.fft2((b - b.mean()) * win)
    cross = Fb * np.conj(Fa)
    cross /= np.abs(cross) + 1e-12
    if lowpass_sigma_px > 0:
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        cross *= np.exp(-2 * np.pi**2 * lowpass_sigma_px**2 * (fx**2 + fy**2))
    r = np.real(np.fft.ifft2(cross))
    iy, ix = np.unravel_index(np.argmax(r), r.shape)

    def sub(m1, c0, p1):
        d = m1 - 2 * c0 + p1
        return 0.5 * (m1 - p1) / d if d != 0 else 0.0

    dy = iy + sub(r[(iy - 1) % h, ix], r[iy, ix], r[(iy + 1) % h, ix])
    dx = ix + sub(r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dy), float(dx), float(r.max() / (r.std() + 1e-12))


def _fit_walk(readings: np.ndarray, width_mm: float, height_mm: float):
    """(dx, dy, walk, walk_se, noise, kept) from (x, y, dx, dy, sharpness) rows.

    Sharpness-weighted plane fit with outlier rejection. Offsets are the plane at
    the sheet center, so tiles dropped unevenly across the sheet don't skew them.
    """
    center = np.array([width_mm / 2, height_mm / 2])
    rel = np.array([[0, 0], [width_mm, 0], [0, height_mm], [width_mm, height_mm]]) - center
    keep = np.ones(len(readings), dtype=bool)
    for _ in range(3):  # fit, drop outliers, refit
        r = readings[keep]
        if len(r) >= MIN_FIT_TILES:
            wts = r[:, 4]
            A = np.column_stack([np.ones(len(r)), r[:, 0] - center[0], r[:, 1] - center[1]])
            Aw = A * wts[:, None]
            gx, *_ = np.linalg.lstsq(Aw, r[:, 2] * wts, rcond=None)
            gy, *_ = np.linalg.lstsq(Aw, r[:, 3] * wts, rcond=None)
            dof = max(1, len(r) - 3)
            s2 = (np.sum(((r[:, 2] - A @ gx) * wts) ** 2) + np.sum(((r[:, 3] - A @ gy) * wts) ** 2)) / dof
            v = np.column_stack([np.zeros(4), rel])
            inv = np.linalg.pinv(Aw.T @ Aw)
            se = OVERLAP_SE_FACTOR * float(np.sqrt(np.max(s2 * np.einsum("ij,jk,ik->i", v, inv, v))))
        else:
            gx = np.array([np.median(r[:, 2]), 0.0, 0.0])
            gy = np.array([np.median(r[:, 3]), 0.0, 0.0])
            se = np.inf
        design = np.column_stack([np.ones(len(readings)), readings[:, 0] - center[0], readings[:, 1] - center[1]])
        resid = np.hypot(readings[:, 2] - design @ gx, readings[:, 3] - design @ gy)
        mad = float(np.median(resid[keep])) or 1e-6
        new_keep = resid <= max(3 * mad, 0.02)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    walk = float(np.hypot(rel @ gx[1:], rel @ gy[1:]).max())
    noise = float(np.sqrt(np.mean(resid[keep] ** 2)))
    return float(gx[0]), float(gy[0]), walk, se, noise, keep


def measure_registration(u8: np.ndarray, model, dpi: float, ruling_lpi: float, key: int = 0, tiles: int = 3,
                         min_sharpness: float = MIN_SHARPNESS) -> dict[str, RegistrationMeasurement | None]:
    pitch_px = dpi / ruling_lpi
    factor = max(1, int(pitch_px / 3))  # keep ~3 samples per cell
    small = model.amounts(downsample_linear(u8, factor))
    sigma = pitch_px / factor  # one cell: removes the screen, keeps the picture
    mm_per_px = factor * MM_PER_INCH / dpi
    h, w = small.shape[:2]
    th, tw = h // tiles, w // tiles
    out = {model.names[key]: RegistrationMeasurement(0.0, 0.0, 0.0)}
    if th < 32 or tw < 32:
        return {**out, **{n: None for i, n in enumerate(model.names) if i != key}}

    # Half-overlapping tiles: more readings for the fit.
    origins = [(y, x) for y in range(0, h - th + 1, th // 2) for x in range(0, w - tw + 1, tw // 2)]
    for i, name in enumerate(model.names):
        if i == key:
            continue
        rows = []
        for y0, x0 in origins:
            a = small[y0 : y0 + th, x0 : x0 + tw, key]
            b = small[y0 : y0 + th, x0 : x0 + tw, i]
            if a.std() < 0.01 or b.std() < 0.01:
                continue
            dy, dx, sharp = phase_correlate(a, b, sigma)
            if sharp >= min_sharpness:
                rows.append(((x0 + tw / 2) * mm_per_px, (y0 + th / 2) * mm_per_px, dx * mm_per_px, dy * mm_per_px, sharp))
        if not rows:
            out[name] = None
            continue
        arr = np.array(rows)
        cdx, cdy, walk, se, noise, keep = _fit_walk(arr, w * mm_per_px, h * mm_per_px)
        out[name] = RegistrationMeasurement(cdx, cdy, walk, noise, [tuple(map(float, r[:4])) for r in arr[keep]], se)
    return out
