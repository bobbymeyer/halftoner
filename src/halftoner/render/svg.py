"""Vector separations as SVG: one group per ink, one <path> per source layer, coordinates in mm.

Round dots below their join are exact arc subpaths; everything else is a traced
polygon (see geometry.py). Masked layers get clip paths that move with the plate.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from ..color import linear_to_hex
from ..sources import Rect
from .geometry import dot_geometry


def plate_path(plate, canvas, use_printed: bool = False, offset=None, precision: int = 2,
               part=None) -> tuple[str, int]:
    """Path data for one plate (or one layer of it) and the number of dots in it."""
    g = dot_geometry(plate, canvas, part, use_printed, offset, tolerance_mm=10.0**-precision)
    fmt = f"{{:.{precision}f}}"
    parts: list[str] = []
    for x, y, r in g.circles:
        R, D = fmt.format(r), fmt.format(2 * r)
        parts.append(f"M{fmt.format(x - r)} {fmt.format(y)}a{R} {R} 0 1 0 {D} 0a{R} {R} 0 1 0 -{D} 0")
    for poly, keep in zip(g.polygons, g.keep):
        parts.append("M" + " ".join(f"{fmt.format(x)} {fmt.format(y)}" for x, y in poly[keep]) + "Z")
    return "".join(parts), g.count


def _region_svg(region, extra: str = "") -> str:
    if isinstance(region, Rect):
        return f'<rect x="{region.x:g}" y="{region.y:g}" width="{region.w:g}" height="{region.h:g}"{extra}/>'
    points = " ".join(f"{x:g},{y:g}" for x, y in region.points)
    return f'<polygon points="{points}"{extra}/>'


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
        offset = offsets.get(ink.name)
        shift = ""
        if offset is not None:
            # Region cuts travel with the plate; the offset at the sheet center stands in for the field.
            tx, ty = (float(np.asarray(v).ravel()[0]) for v in offset(np.array([w / 2]), np.array([h / 2])))
            shift = f' transform="translate({tx:.3f} {ty:.3f})"'
        fill = linear_to_hex(ink.transmittance)  # the ink at its density on white
        paths, counts[ink.name] = [], 0
        for i, part in enumerate(plate.parts):
            d, n = plate_path(plate, canvas, use_printed=artifacts, offset=offset, precision=precision, part=part)
            counts[ink.name] += n
            clip_attr = ""
            if part.regions:
                cid = f"clip-{escape(ink.name)}" + (f"-{i}" if len(plate.parts) > 1 else "")
                lines.append(f'<clipPath id="{cid}">{"".join(_region_svg(r, shift) for r in part.regions)}</clipPath>')
                clip_attr = f' clip-path="url(#{cid})"'
            paths.append(f'<path fill="{fill}"{clip_attr} d="{d}"/>')
        # One path per layer, each with its own cut; multiply previews overprints subtractively.
        lines.append(f'<g id="{escape(ink.name)}" style="mix-blend-mode:multiply">{"".join(paths)}</g>')
    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return counts
