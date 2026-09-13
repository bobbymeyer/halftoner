"""Ruling and angle from the Fourier spectrum of each ink's amount map.

A halftone screen is a lattice, so its spectrum is a lattice of peaks. The
fundamental is the nearest strong peak to the origin: its distance is the
ruling and its direction the angle (mod 90 for a square cell). Harmonics sit
further out; the (1,1) diagonal can dominate near 50% checkerboards, so the
nearest peak within 30% of the strongest wins, and several windows are pooled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import crop_linear

_RING_BINS = 8
_ISOLATION = 3.0


@dataclass
class ScreenMeasurement:
    ruling_lpi: float
    angle: float  # degrees counterclockwise on the page, mod 90
    confidence: float  # peak over median spectrum in the search band
    px_per_cell: float
    windows: int
    spread_lpi: float
    spread_angle: float


def _parabolic(m1: float, c0: float, p1: float) -> float:
    denom = m1 - 2 * c0 + p1
    return 0.5 * (m1 - p1) / denom if denom != 0 else 0.0


def spectrum_peak(amount: np.ndarray, dpi: float, min_lpi: float = 15, max_lpi: float = 400):
    """(lpi, angle, confidence) for one square window, or None."""
    N = amount.shape[0]
    a = (amount - amount.mean()) * np.outer(np.hanning(N), np.hanning(N))
    M = 2 * N
    F = np.abs(np.fft.fftshift(np.fft.fft2(a, s=(M, M))))
    f = (np.arange(M) - M // 2) / M
    FX, FY = np.meshgrid(f, f)
    R = np.hypot(FX, FY) * dpi
    band = (R >= min_lpi) & (R <= max_lpi) & ((FY < 0) | ((FY == 0) & (FX > 0)))
    Fb = np.where(band, F, 0.0)
    if Fb.max() <= 0:
        return None
    neighbours = [np.roll(np.roll(Fb, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    local = np.argwhere((Fb >= np.maximum.reduce(neighbours)) & (Fb > 0))
    local = local[np.argsort(-Fb[local[:, 0], local[:, 1]])[:400]]

    # A screen is a lattice of isolated spectral points; picture edges and patch
    # boundaries are lines through the spectrum. Keep peaks that stand clear of a
    # ring just outside the Hann main lobe (4 bins, doubled by padding).
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring_y = np.round(_RING_BINS * np.sin(theta)).astype(int)
    ring_x = np.round(_RING_BINS * np.cos(theta)).astype(int)
    ys = np.clip(local[:, 0:1] + ring_y[None], 0, M - 1)
    xs = np.clip(local[:, 1:2] + ring_x[None], 0, M - 1)
    ring = F[ys, xs].max(axis=1)
    vals = Fb[local[:, 0], local[:, 1]]
    isolated = local[vals >= _ISOLATION * ring]
    if not len(isolated):
        return None
    ivals = Fb[isolated[:, 0], isolated[:, 1]]
    peak = ivals.max()
    r_peak = R[tuple(isolated[np.argmax(ivals)])]
    radii = R[isolated[:, 0], isolated[:, 1]]
    # The fundamental can be weaker than the (1,1) diagonal near 50%, but never nearer than 1/sqrt(2) of it.
    keep = (ivals >= 0.3 * peak) & (radii >= 0.6 * r_peak)
    cand, radii = isolated[keep], radii[keep]
    near = cand[radii <= radii.min() * 1.03]
    iy, ix = max(near, key=lambda c: Fb[c[0], c[1]])
    sx = _parabolic(F[iy, ix - 1], F[iy, ix], F[iy, ix + 1])
    sy = _parabolic(F[iy - 1, ix], F[iy, ix], F[iy + 1, ix])
    fx, fy = (ix + sx - M // 2) / M, (iy + sy - M // 2) / M
    angle = float(np.degrees(np.arctan2(-fy, fx)) % 90)
    return float(np.hypot(fx, fy) * dpi), angle, float(Fb[iy, ix] / np.median(F[band]))


def _circular_mean_90(angles) -> tuple[float, float]:
    z = np.exp(1j * np.deg2rad(np.asarray(angles)) * 4)
    mean = float(np.rad2deg(np.angle(z.mean())) / 4 % 90)
    dev = np.abs((np.asarray(angles) - mean + 45) % 90 - 45)
    return mean, float(dev.max()) if dev.size else 0.0


def measure_screens(u8: np.ndarray, model, dpi: float, size: int = 1024, grid: int = 3,
                    min_lpi: float = 15, max_lpi: float = 400) -> list[ScreenMeasurement | None]:
    H, W = u8.shape[:2]
    size = min(size, 1 << int(np.log2(min(H, W))))
    ys = np.linspace(0, H - size, grid).astype(int) if H > size else [0]
    xs = np.linspace(0, W - size, grid).astype(int) if W > size else [0]
    readings = [[] for _ in range(model.n)]
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            amounts = model.amounts(crop_linear(u8, y0, x0, size, size))
            for i in range(model.n):
                amt = amounts[..., i]
                if not 0.03 < amt.mean() < 0.97:
                    continue  # empty or solid here: nothing screened to read
                hit = spectrum_peak(amt, dpi, min_lpi, max_lpi)
                if hit:
                    readings[i].append(hit)

    out = []
    for r in readings:
        if not r:
            out.append(None)
            continue
        lpi = np.array([h[0] for h in r])
        angle, spread = _circular_mean_90([h[1] for h in r])
        ruling = float(np.median(lpi))
        out.append(ScreenMeasurement(ruling, angle, float(np.median([h[2] for h in r])), dpi / ruling,
                                     len(r), float(np.abs(lpi - ruling).max()), spread))
    return out
