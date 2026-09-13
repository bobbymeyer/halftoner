"""Sources: what a screen fills with.

The screen is a fill. Give it an image and it's a halftone; give it a number
and it's a tint; give it a gradient and it's a vignette. One code path.

Every source declares a `kind`:
  "area" -- values are nominal dot area, 0 = no ink, 1 = solid (tints, vignettes)
  "tone" -- values are linear reflectance, 0 = black, 1 = paper (photographs)
Transfer turns tone into area against a specific ink; area passes through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image as PILImage

from .color import luma_linear, srgb_to_linear

Kind = str
EMPTY = {"area": 0.0, "tone": 1.0}


class Source:
    kind: Kind = "area"

    def sample(self, x_mm: np.ndarray, y_mm: np.ndarray, cell_mm: float, canvas) -> np.ndarray:
        raise NotImplementedError

    def sampling_ratio(self, cell_mm: float, canvas) -> float | None:
        """Source pixels per screen cell along one axis. None when resolution-free."""
        return None


@dataclass
class Constant(Source):
    value: float
    kind: Kind = "area"

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        return np.full(np.shape(x_mm), float(self.value))


@dataclass
class Gradient(Source):
    """Linear ramp from `start` at p0 to `end` at p1 (mm), clamped beyond."""

    start: float
    end: float
    p0: tuple[float, float] = (0.0, 0.0)
    p1: tuple[float, float] | None = None
    kind: Kind = "area"

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        p1 = self.p1 or (canvas.width_mm, 0.0)
        dx, dy = p1[0] - self.p0[0], p1[1] - self.p0[1]
        t = ((x_mm - self.p0[0]) * dx + (y_mm - self.p0[1]) * dy) / (dx * dx + dy * dy)
        return self.start + (self.end - self.start) * np.clip(t, 0, 1)


@dataclass
class Function(Source):
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    kind: Kind = "area"

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        return np.broadcast_to(np.asarray(self.fn(x_mm, y_mm), dtype=np.float64), np.shape(x_mm))


@dataclass
class Image(Source):
    """Photograph as linear reflectance.

    fit: "cover" | "contain" | "stretch", placed within `box` (x, y, w, h mm;
    defaults to the whole canvas including its bleed). `channel` is "luma", "r", "g", "b", or a
    callable over linear RGB (..., 3) returning reflectance.
    """

    path: str | Path
    fit: str = "cover"
    channel: str | Callable = "luma"
    box: tuple[float, float, float, float] | None = None
    kind: Kind = "tone"
    _lin: np.ndarray = field(init=False, repr=False)
    _cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        rgb = srgb_to_linear(np.asarray(PILImage.open(self.path).convert("RGB"), dtype=np.float64) / 255.0)
        if callable(self.channel):
            self._lin = np.asarray(self.channel(rgb), dtype=np.float32)
        elif self.channel == "luma":
            self._lin = luma_linear(rgb).astype(np.float32)
        else:
            self._lin = rgb[..., "rgb".index(self.channel)].astype(np.float32)

    def _placement(self, canvas):
        """(mm per source px x, y) and top-left offset in mm."""
        b = canvas.bleed_mm  # by default a photo runs into the bleed, as art for print should
        bx, by, bw, bh = self.box or (-b, -b, canvas.width_mm + 2 * b, canvas.height_mm + 2 * b)
        ih, iw = self._lin.shape
        if self.fit == "stretch":
            return bw / iw, bh / ih, bx, by
        s = (max if self.fit == "cover" else min)(bw / iw, bh / ih)
        return s, s, bx + (bw - iw * s) / 2, by + (bh - ih * s) / 2

    def sampling_ratio(self, cell_mm, canvas):
        sx, sy, _, _ = self._placement(canvas)
        return cell_mm / max(sx, sy)

    def _prefiltered(self, k: float) -> np.ndarray:
        """Box-filter to about one source sample per cell so tone is an area mean, not a point."""
        if k <= 1.0:
            return self._lin
        key = round(k, 4)
        if key not in self._cache:
            ih, iw = self._lin.shape
            size = (max(1, round(iw / k)), max(1, round(ih / k)))
            img = PILImage.fromarray(self._lin, mode="F").resize(size, PILImage.Resampling.BOX)
            self._cache[key] = np.asarray(img, dtype=np.float32)
        return self._cache[key]

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        sx, sy, ox, oy = self._placement(canvas)
        ih, iw = self._lin.shape
        arr = self._prefiltered(self.sampling_ratio(cell_mm, canvas))
        ah, aw = arr.shape
        px = (np.asarray(x_mm) - ox) / (sx * iw) * aw - 0.5
        py = (np.asarray(y_mm) - oy) / (sy * ih) * ah - 0.5
        inside = (px >= -0.5) & (px <= aw - 0.5) & (py >= -0.5) & (py <= ah - 0.5)
        x0 = np.clip(np.floor(px).astype(int), 0, aw - 1)
        y0 = np.clip(np.floor(py).astype(int), 0, ah - 1)
        x1, y1 = np.clip(x0 + 1, 0, aw - 1), np.clip(y0 + 1, 0, ah - 1)
        fx, fy = np.clip(px - x0, 0, 1), np.clip(py - y0, 0, 1)
        top = arr[y0, x0] * (1 - fx) + arr[y0, x1] * fx
        bot = arr[y1, x0] * (1 - fx) + arr[y1, x1] * fx
        return np.where(inside, top * (1 - fy) + bot * fy, EMPTY["tone"])


# --- regions: fill an arbitrary path -------------------------------------------------


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, x, y):
        return (x >= self.x) & (x < self.x + self.w) & (y >= self.y) & (y < self.y + self.h)

    def signed_distance(self, x, y):
        """mm to the edge: positive inside, negative outside."""
        qx = np.abs(np.asarray(x) - (self.x + self.w / 2)) - self.w / 2
        qy = np.abs(np.asarray(y) - (self.y + self.h / 2)) - self.h / 2
        return -(np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0))


@dataclass(frozen=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def contains(self, x, y):
        """Even-odd rule."""
        inside = np.zeros(np.shape(x), dtype=bool)
        pts = self.points
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            crosses = (y1 > y) != (y2 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (x < xint)
        return inside

    def signed_distance(self, x, y):
        """mm to the nearest edge: positive inside, negative outside."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        d = np.full(np.shape(x), np.inf)
        pts = self.points
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            ex, ey = x2 - x1, y2 - y1
            t = np.clip(((x - x1) * ex + (y - y1) * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1)
            d = np.minimum(d, np.hypot(x - (x1 + t * ex), y - (y1 + t * ey)))
        return np.where(self.contains(x, y), d, -d)


@dataclass
class Masked(Source):
    source: Source
    region: Rect | Polygon

    @property
    def kind(self):
        return self.source.kind

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        # Cells straddling the edge carry the region's value; the renderer then cuts
        # their dots at the true edge, the way a tint was cut from film.
        v = self.source.sample(x_mm, y_mm, cell_mm, canvas)
        near = self.region.signed_distance(x_mm, y_mm) >= -cell_mm * np.sqrt(0.5)
        return np.where(near, v, EMPTY[self.kind])

    def sampling_ratio(self, cell_mm, canvas):
        return self.source.sampling_ratio(cell_mm, canvas)


@dataclass
class Layered(Source):
    """Several sources on one plate, e.g. a photo plus a flat tint panel.

    The recipe runs each layer through the ink's tone chain separately (so a
    tone layer and an area layer can share a plate) and the heaviest dot wins.
    """

    sources: list[Source]
    kind: Kind = "layered"

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        raise TypeError("Layered is resolved per layer by the recipe")

    def sampling_ratio(self, cell_mm, canvas):
        ratios = [r for s in self.sources if (r := s.sampling_ratio(cell_mm, canvas)) is not None]
        return min(ratios) if ratios else None


# --- clipping: where a plate's ink may land, at full resolution ---------------------------


def clip_of(src):
    """fn(x_mm, y_mm, inset_mm=0) -> bool for a plate's source, or None when it covers the sheet.

    Masked sources clip to their region (intersected with any inner mask);
    Layered sources clip to the union of their layers. `inset` chokes the edge,
    which is how a trap gap opens between abutting regions.
    """
    if isinstance(src, Masked):
        inner = clip_of(src.source)
        region = src.region
        if inner is None:
            return lambda x, y, inset=0.0: region.signed_distance(x, y) >= inset
        return lambda x, y, inset=0.0: (region.signed_distance(x, y) >= inset) & inner(x, y, inset)
    if isinstance(src, Layered):
        parts = [clip_of(s) for s in src.sources]
        if any(p is None for p in parts):
            return None
        return lambda x, y, inset=0.0: np.logical_or.reduce([p(x, y, inset) for p in parts])
    return None


def clip_regions(src) -> list | None:
    """Union of regions for vector clip paths (outermost mask of each layer), or None."""
    if isinstance(src, Masked):
        return [src.region]
    if isinstance(src, Layered):
        parts = [clip_regions(s) for s in src.sources]
        if any(p is None for p in parts):
            return None
        return [r for p in parts for r in p]
    return None
