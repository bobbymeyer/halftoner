"""Screen target: supersampled, composited in ink space, flattened to sRGB once.

Each subpixel records which inks landed on it as a bitmask; the bitmask indexes
the ink set's Neugebauer primaries (so overprint overrides apply), and subpixels
are area-averaged in linear light. The box average over analytic coverage is
the anti-aliasing, which keeps the screen from beating against the pixel grid.
Rendering is strip-by-strip so poster sizes fit in memory.

Press artifacts, when on:
  misregistration  plates (dots and their region cuts) move by the press's offset field
  slur             each hit is smeared along the press direction
  trap gap         region cuts are choked by half the gap, opening paper where regions meet
  density variance each present ink's density is scaled by its low-frequency field
                   (Beer-Lambert on top of the primary, so chosen overprints keep their color)
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
    choke = press.trap_gap / 2 if artifacts else 0.0
    variance = press.density_fields(recipe.inks.inks, canvas) if artifacts else {}
    thresholds = [[p.shape.threshold(part.printed if artifacts else part.area) for part in p.parts] for p in plates]
    if variance:
        paper = np.maximum(primaries[0], 1e-6)
        primary_d = -np.log10(np.clip(primaries / paper, 1e-6, None)).astype(np.float32)
        solid_d = np.array([-np.log10(np.maximum(p.ink.transmittance, 1e-6)) for p in plates], dtype=np.float32)

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
            inked = _hit(plate, thr, px, py, choke)
            if slur is not None:
                inked |= _hit(plate, thr, px - slur[0], py - slur[1], choke)
            mask |= inked.astype(np.uint8) << k
        if variance:
            d = primary_d[mask]
            for k, plate in enumerate(plates):
                delta = variance[plate.ink.name](X, Y).astype(np.float32)
                d += (((mask >> k) & 1) * delta)[..., None] * solid_d[k]
            sub = paper * np.power(10.0, -d, dtype=np.float32)
        else:
            sub = primaries[mask]
        lin = sub.reshape(r1 - r0, ss, W, ss, 3).mean(axis=(1, 3))
        out[r0:r1] = np.round(linear_to_srgb(lin) * 255).astype(np.uint8)
    return out


def _hit(plate, thresholds, x, y, choke: float = 0.0):
    """Inked where any layer's dot covers the point inside that layer's (choked) cut."""
    row, col, valid, u, v = plate.locate(x, y)
    spot = plate.shape.spot(u, v)
    hit = np.zeros(np.shape(x), dtype=bool)
    for part, thr in zip(plate.parts, thresholds):
        layer = valid & (spot <= thr[row, col])
        if part.clip is not None:
            layer &= part.clip(x, y, choke)
        hit |= layer
    return hit


def save_png(pixels: np.ndarray, path, dpi: float) -> None:
    PILImage.fromarray(pixels, mode="RGB").save(path, dpi=(dpi, dpi))
