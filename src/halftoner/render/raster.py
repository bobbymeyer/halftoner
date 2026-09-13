"""Screen target: supersampled, composited in ink space, flattened to sRGB once.

Each subpixel records which inks landed on it as a bitmask; the bitmask indexes
the ink set's Neugebauer primaries (so overprint overrides apply), and subpixels
are area-averaged in linear light. The box average over analytic coverage is
the anti-aliasing, which keeps the screen from beating against the pixel grid.
Rendering is strip-by-strip so poster sizes fit in memory.
"""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from ..color import linear_to_srgb


def composite(recipe, supersample: int = 4, artifacts: bool = True, strip_rows: int = 48) -> np.ndarray:
    canvas, press = recipe.canvas, recipe.press
    plates = recipe.plates()
    if len(plates) > 8:
        raise ValueError("at most 8 inks")
    primaries = recipe.inks.primaries(recipe.substrate.paper).astype(np.float32)
    offsets = press.offsets(recipe.inks.inks, canvas) if artifacts else None
    slur = press.slur_vector() if artifacts and press.slur > 0 else None
    thresholds = [p.shape.threshold(p.printed if artifacts else p.area) for p in plates]

    W, H = canvas.size_px
    ss = supersample
    sub_mm = 1.0 / (canvas.px_per_mm * ss)
    xs = (np.arange(W * ss) + 0.5) * sub_mm
    out = np.empty((H, W, 3), dtype=np.uint8)

    for r0 in range(0, H, strip_rows):
        r1 = min(H, r0 + strip_rows)
        ys = (np.arange(r0 * ss, r1 * ss) + 0.5) * sub_mm
        X, Y = np.meshgrid(xs, ys)
        mask = np.zeros(X.shape, dtype=np.uint8)
        for k, (plate, thr) in enumerate(zip(plates, thresholds)):
            px, py = X, Y
            if offsets is not None:
                dx, dy = offsets[plate.ink.name](X, Y)
                px, py = X - dx, Y - dy
            inked = _hit(plate, thr, px, py)
            if slur is not None:
                inked |= _hit(plate, thr, px - slur[0], py - slur[1])
            mask |= inked.astype(np.uint8) << k
        lin = primaries[mask].reshape(r1 - r0, ss, W, ss, 3).mean(axis=(1, 3))
        out[r0:r1] = np.round(linear_to_srgb(lin) * 255).astype(np.uint8)
    return out


def _hit(plate, thr, x, y):
    row, col, valid, u, v = plate.locate(x, y)
    return valid & (plate.shape.spot(u, v) <= thr[row, col])


def save_png(pixels: np.ndarray, path, dpi: float) -> None:
    PILImage.fromarray(pixels, mode="RGB").save(path, dpi=(dpi, dpi))
