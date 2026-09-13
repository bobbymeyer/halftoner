"""Film positives: one 1-bit sheet per ink, no anti-aliasing, no press artifacts, mirrored.

Each sheet carries the trim plus a margin with crop marks, registration targets,
a label, and a step wedge screened at the ink's own ruling, angle and shape.
The wedge patches are nominal (uncompensated) areas: print them, measure them,
and feed the pairs to Curve.from_points to close the gain loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageOps

MARGIN_MM = 15.0
WEDGE = (0, 2, 5, 10, 25, 50, 75, 90, 95, 98, 100)
WEDGE_H_MM = 8.0


def _cell_uv(plate, x, y):
    """In-cell coords on the plate's grid, unbounded by the plate's cell array."""
    s, t = plate.to_screen(x, y)
    cs = s / plate.pitch_mm - plate.phase[0]
    ct = t / plate.pitch_mm - plate.phase[1]
    return cs - np.floor(cs) - 0.5, ct - np.floor(ct) - 0.5


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow without FreeType sizing
        return ImageFont.load_default()


def film_sheet(recipe, plate, dpi: float = 1200, wedge: bool = True, strip_rows: int = 256) -> PILImage.Image:
    cv, M = recipe.canvas, MARGIN_MM
    ppm = dpi / 25.4
    W, H = round((cv.width_mm + 2 * M) * ppm), round((cv.height_mm + 2 * M) * ppm)
    thr = plate.shape.threshold(plate.area)
    b = cv.bleed_mm
    n = len(WEDGE)
    wx0 = 2.0  # inset so the wedge clears the crop marks at the trim corners
    patch_w = min(12.0, (cv.width_mm - 2 * wx0) / n)
    wedge_thr = plate.shape.threshold(np.array(WEDGE) / 100)
    wy0 = cv.height_mm + 3.0

    black = np.zeros((H, W), dtype=bool)
    xs = (np.arange(W) + 0.5) / ppm - M
    for r0 in range(0, H, strip_rows):
        r1 = min(H, r0 + strip_rows)
        X, Y = np.meshgrid(xs, (np.arange(r0, r1) + 0.5) / ppm - M)
        art = (X >= -b) & (X < cv.width_mm + b) & (Y >= -b) & (Y < cv.height_mm + b)
        row, col, valid, u, v = plate.locate(X, Y)
        inked = art & valid & (plate.shape.spot(u, v) <= thr[row, col])
        if wedge:
            in_wedge = (Y >= wy0) & (Y < wy0 + WEDGE_H_MM) & (X >= wx0) & (X < wx0 + patch_w * n)
            if in_wedge.any():
                k = np.clip(((X - wx0) / patch_w).astype(int), 0, n - 1)
                wu, wv = _cell_uv(plate, X, Y)
                inked |= in_wedge & (plate.shape.spot(wu, wv) <= wedge_thr[k])
        black[r0:r1] = inked

    img = PILImage.fromarray(np.where(black, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    px = lambda mm: (mm + M) * ppm  # noqa: E731
    line = max(1, round(0.1 * ppm))

    # Crop marks at trim corners, outside the bleed.
    gap, length = max(2.0, b + 1.0), 8.0
    for cx in (0.0, cv.width_mm):
        for cy in (0.0, cv.height_mm):
            sx = -1 if cx == 0 else 1
            sy = -1 if cy == 0 else 1
            draw.line([px(cx + sx * gap), px(cy), px(cx + sx * (gap + length)), px(cy)], fill=0, width=line)
            draw.line([px(cx), px(cy + sy * gap), px(cx), px(cy + sy * (gap + length))], fill=0, width=line)

    # Registration targets centred on each side.
    r = 3.0
    for tx, ty in ((cv.width_mm / 2, -M / 2), (-M / 2, cv.height_mm / 2), (cv.width_mm + M / 2, cv.height_mm / 2)):
        draw.ellipse([px(tx - r), px(ty - r), px(tx + r), px(ty + r)], outline=0, width=line)
        draw.line([px(tx - r - 2), px(ty), px(tx + r + 2), px(ty)], fill=0, width=line)
        draw.line([px(tx), px(ty - r - 2), px(tx), px(ty + r + 2)], fill=0, width=line)

    ink = plate.ink
    label = (
        f"{ink.name}  |  {1 / plate.pitch_mm * 25.4:g} lpi @ {plate.angle_deg:g} deg  |  {plate.shape.name}  |  "
        f"{dpi:g} dpi  |  {recipe.profile or 'no profile'}  |  seed {recipe.press.seed}  |  film positive, mirrored"
    )
    # Largest size up to 2.2 mm cap height that fits the sheet width.
    size = max(8, round(2.2 * ppm))
    font = _font(size)
    while size > 8 and draw.textlength(label, font=font) > W - 4 * ppm:
        size = max(8, int(size * 0.9))
        font = _font(size)
    draw.text((2 * ppm, px(-M + 2)), label, fill=0, font=font)
    if wedge:
        small = _font(max(8, round(min(1.6, patch_w / 3) * ppm)))
        for k, pct in enumerate(WEDGE):
            draw.text((px(wx0 + k * patch_w + 0.5), px(wy0 + WEDGE_H_MM + 0.5)), f"{pct}", fill=0, font=small)

    return ImageOps.mirror(img).convert("1", dither=PILImage.Dither.NONE)


def write_films(recipe, out_dir, stem: str, dpi: float = 1200, wedge: bool = True) -> list[Path]:
    out_dir = Path(out_dir)
    paths = []
    for plate in recipe.plates():
        p = out_dir / f"{stem}_film_{plate.ink.name}.png"
        film_sheet(recipe, plate, dpi=dpi, wedge=wedge).save(p, dpi=(dpi, dpi))
        paths.append(p)
    return paths
