"""Unsmoothed per-plate rasterization shared by film sheets and PDF image masks.

Nominal plate areas, region cuts unchoked, no press artifacts: what goes on film
or plate. Strips are sized to a pixel budget so an A2 sheet at 2400 dpi (tens of
thousands of pixels wide) still renders in bounded memory.
"""

from __future__ import annotations

import numpy as np

from ..canvas import MM_PER_INCH

PIXELS_PER_STRIP = 2_000_000


def hit_rows(plate, x0_mm: float, y0_mm: float, width_px: int, height_px: int, dpi: float):
    """Yield (r0, X, Y, inked) strips over a pixel grid whose top-left corner sits at (x0_mm, y0_mm)."""
    ppm = dpi / MM_PER_INCH
    thresholds = [plate.shape.threshold(part.area) for part in plate.parts]
    xs = x0_mm + (np.arange(width_px) + 0.5) / ppm
    step = max(1, PIXELS_PER_STRIP // max(1, width_px))
    for r0 in range(0, height_px, step):
        r1 = min(height_px, r0 + step)
        X, Y = np.meshgrid(xs, y0_mm + (np.arange(r0, r1) + 0.5) / ppm)
        row, col, valid, u, v = plate.locate(X, Y)
        spot = plate.shape.spot(u, v)
        inked = np.zeros(X.shape, dtype=bool)
        for part, thr in zip(plate.parts, thresholds):
            layer = valid & (spot <= thr[row, col])
            if part.clip is not None:
                layer &= part.clip(X, Y)
            inked |= layer
        yield r0, X, Y, inked
