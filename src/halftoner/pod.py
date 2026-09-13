"""Print-on-demand: one recipe, a file per size band.

The recipe is the artifact and size is an input to a run. A band is a standard
print size (grouped by aspect) with a resolution and a coarse ruling cap: an
inkjet reproduces a baked-in screen only while each cell spans several pixels.
Each band renders the design at that size (art scales, press physics doesn't),
caps the ruling, and writes an RGBA PNG with hard alpha by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .canvas import MM_PER_INCH

BANDS_FILE = Path(__file__).parent / "pod_bands.json"
MIN_PX_PER_CELL = 5.0  # default ruling cap: dpi / this


@dataclass(frozen=True)
class SizeBand:
    name: str
    width_mm: float
    height_mm: float
    dpi: float
    max_lpi: float
    family: str = ""


def load_bands(path: str | Path | None = None) -> list[SizeBand]:
    d = json.loads(Path(path or BANDS_FILE).read_text())
    out = []
    for family, bands in d["families"].items():
        for b in bands:
            w, h = (v * MM_PER_INCH for v in b["size_in"]) if "size_in" in b else b["size_mm"]
            out.append(SizeBand(b["name"], w, h, b["dpi"], b.get("max_lpi", b["dpi"] / MIN_PX_PER_CELL), family))
    return out


def bands_for(canvas, bands: list[SizeBand] | None = None, tolerance: float = 0.02) -> list[SizeBand]:
    """Bands whose aspect matches the canvas in either orientation, turned to match it."""
    aspect = canvas.width_mm / canvas.height_mm
    out = []
    for band in load_bands() if bands is None else bands:
        for w, h in ((band.width_mm, band.height_mm), (band.height_mm, band.width_mm)):
            if abs(w / h - aspect) <= tolerance * aspect:
                out.append(replace(band, width_mm=w, height_mm=h))
                break
    return out


def render_bands(recipe, out_dir, stem: str, size: str = "all", bands=None, alpha: str | None = "hard",
                 supersample: int = 4) -> tuple[list[Path], dict[str, tuple[float, float]]]:
    """Render `size` ("all" or a band name) of the bands matching the recipe's aspect.

    `bands` may be a list of SizeBand or a path to a bands JSON file. Returns the files
    written and every ruling cap applied, keyed "band/ink".
    """
    from .policy import cap_ruling
    from .render.raster import composite, save_png

    if isinstance(bands, (str, Path)):
        bands = load_bands(bands)
    matches = bands_for(recipe.canvas, bands)
    if not matches:
        cv = recipe.canvas
        raise ValueError(f"no size band matches a {cv.width_mm:g} x {cv.height_mm:g} mm design; pass bands")
    chosen = matches if size == "all" else [b for b in matches if b.name == size]
    if not chosen:
        raise ValueError(f"no band named {size!r}; this aspect has {', '.join(b.name for b in matches)}")

    out_dir = Path(out_dir)
    paths, caps = [], {}
    for band in chosen:
        job = recipe.at_size(band.width_mm, band.height_mm, dpi=band.dpi)
        job, capped = cap_ruling(job, min(job.substrate.ruling_ceiling_lpi, band.max_lpi))
        caps.update({f"{band.name}/{ink}": change for ink, change in capped.items()})
        path = out_dir / f"{stem}_pod_{band.name}.png"
        save_png(composite(job, supersample=supersample, alpha=alpha), path, band.dpi)
        paths.append(path)
    return paths, caps
