"""Underbase: the plate that goes down first on a dark garment, derived from where the colors print.

Plastisol colors are translucent enough that a black shirt swallows them, so an
underbase (usually white) is printed and flashed first. It isn't drawn, it's
derived: at each point it carries the heaviest printed coverage of the colors
above it, optionally weighted (less white under dark inks), and then

- choked, pulled back from edges so misregistration doesn't leave a white halo.
  Tonal edges are eroded on the underbase's cell grid; region edges are cut at
  full resolution with every color region inset by the choke.
- gain-compensated on its own curve, because white on cotton spreads far more
  than the colors printed onto the flashed base.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .curves import Curve
from .ink import Ink


@dataclass
class Underbase:
    ink: Ink = field(default_factory=lambda: Ink("underbase", "#F4F4F0", angle=22.5, opacity=0.9))
    choke_mm: float = 0.25
    weights: dict[str, float] = field(default_factory=dict)  # per color ink (default 1): how much base under it
    gain: float | Curve = 0.0  # the underbase's own gain; a number is TVI at 50%
    compensate_gain: bool = True

    @property
    def gain_curve(self) -> Curve:
        return Curve.from_spec(self.gain)


MIN_ERODE_CELLS = 1.0
"""Below one cell the disk holds no neighbour, so eroding a tonal edge is a no-op.

The choke is a length in millimetres but tonal edges are eroded on the underbase's
cell grid, so a choke finer than the cell cannot pull the base back at all. Callers
that care (the report's `choke` check) compare against this rather than guessing.
"""


def erode(grid: np.ndarray, radius_cells: float) -> np.ndarray:
    """Grayscale erosion: minimum over a disk of `radius_cells`. Beyond the grid repeats the edge."""
    if radius_cells < MIN_ERODE_CELLS:
        return grid
    r = int(np.floor(radius_cells))
    padded = np.pad(grid, r, mode="edge")
    h, w = grid.shape
    out = grid.copy()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if (di or dj) and di * di + dj * dj <= radius_cells**2:
                out = np.minimum(out, padded[r + di : r + di + h, r + dj : r + dj + w])
    return out
