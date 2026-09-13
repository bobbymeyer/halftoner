"""Vector separations: one <path> per ink, coordinates in mm.

Round dots below their join are exact arc subpaths. Everything else (joined
dots, other shapes) is traced as a polygon by marching rays out from the cell
center to the spot-function threshold, clipped to the cell, with the cell
corners always among the rays so clipped edges stay square.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

_RAYS = 48
_BISECT = 18


def _ray_angles():
    base = np.linspace(0, 2 * np.pi, _RAYS, endpoint=False)
    corners = np.array([1, 3, 5, 7]) * np.pi / 4
    return np.unique(np.concatenate([base, corners]))


def _trace(shape, thr: np.ndarray) -> np.ndarray:
    """(cells, rays, 2) polygon vertices in cell units for thresholds `thr` (cells,)."""
    ang = _ray_angles()
    cu, cv = np.cos(ang), np.sin(ang)
    tmax = 0.5 / np.maximum(np.abs(cu), np.abs(cv))
    t = thr[:, None]
    lo = np.zeros((thr.size, ang.size))
    hi = np.broadcast_to(tmax, lo.shape).copy()
    reaches_edge = shape.spot(hi * cu, hi * cv) <= t
    for _ in range(_BISECT):
        mid = (lo + hi) / 2
        inside = shape.spot(mid * cu, mid * cv) <= t
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    r = np.where(reaches_edge, tmax, (lo + hi) / 2)
    return np.stack([r * cu, r * cv], axis=-1)


def plate_path(plate, canvas, use_printed: bool = False, offset=None, precision: int = 2) -> tuple[str, int]:
    """Path data for one plate and the number of dots in it."""
    area = plate.printed if use_printed else plate.area
    X, Y = plate.centers()
    pad = plate.pitch_mm
    keep = (area > 0) & (X > -pad) & (X < canvas.width_mm + pad) & (Y > -pad) & (Y < canvas.height_mm + pad)
    if offset is not None:
        dx, dy = offset(X, Y)
        X, Y = X + dx, Y + dy
    a, cx, cy = area[keep], X[keep], Y[keep]
    fmt = f"{{:.{precision}f}}"
    parts: list[str] = []

    circle = np.zeros(a.shape, dtype=bool)
    if plate.shape.name == "round":
        circle = a < np.pi / 4
        r = np.sqrt(a[circle] / np.pi) * plate.pitch_mm
        for x, y, rr in zip(cx[circle], cy[circle], r):
            R, D = fmt.format(rr), fmt.format(2 * rr)
            parts.append(f"M{fmt.format(x - rr)} {fmt.format(y)}a{R} {R} 0 1 0 {D} 0a{R} {R} 0 1 0 -{D} 0")

    poly = ~circle
    if poly.any():
        verts = _trace(plate.shape, plate.shape.threshold(a[poly])) * plate.pitch_mm
        uax, vax = plate.axes
        px = cx[poly, None] + verts[..., 0] * uax[0] + verts[..., 1] * vax[0]
        py = cy[poly, None] + verts[..., 0] * uax[1] + verts[..., 1] * vax[1]
        for xr, yr in zip(px, py):
            pts = " ".join(f"{fmt.format(x)} {fmt.format(y)}" for x, y in zip(xr, yr))
            parts.append(f"M{pts}Z")
    return "".join(parts), int(a.size)


def write_svg(recipe, path, artifacts: bool = False, precision: int = 2) -> dict[str, int]:
    """Separations as layered groups. The multiply blend is a preview only;
    the accurate composite (with overprint overrides) is the raster target."""
    canvas = recipe.canvas
    offsets = recipe.press.offsets(recipe.inks.inks, canvas) if artifacts else {}
    w, h = canvas.width_mm, canvas.height_mm
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" height="{h:g}mm" viewBox="0 0 {w:g} {h:g}">',
        f'<rect id="paper" width="{w:g}" height="{h:g}" fill="{recipe.substrate.paper}"/>',
    ]
    counts = {}
    for plate in recipe.plates():
        ink = plate.ink
        d, n = plate_path(plate, canvas, use_printed=artifacts, offset=offsets.get(ink.name), precision=precision)
        counts[ink.name] = n
        lines.append(
            f'<g id="{escape(ink.name)}" style="mix-blend-mode:multiply" opacity="{ink.density:g}">'
            f'<path fill="{ink.color}" d="{d}"/></g>'
        )
    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return counts
