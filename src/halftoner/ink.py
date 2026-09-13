"""Inks and ink sets. A job is an ink set, not a stack of RGB channels.

The overprint table is where mid-century color lives: two inks were picked
because their overlap was good. Overprints are computed subtractively by
default and can be overridden for any combination. An n-ink job therefore has
2^n Neugebauer primaries (paper included), each individually specifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .cellfill import CellFill
from .color import hex_to_linear, linear_to_hex, luma_linear
from .curves import Curve


@dataclass
class Ink:
    name: str
    color: str  # appearance of a solid on white paper
    density: float = 1.0  # film thickness relative to the swatch: 0 = water, 1 = the swatch, >1 = heavier
    angle: float = 45.0  # degrees, counterclockwise on the page
    curve: Curve = field(default_factory=Curve.identity)  # the ink's own transfer curve
    opacity: float = 0.0  # 0 = transparent (multiplies), 1 = hides what's under it
    source: object | None = None  # per-ink source; falls back to the recipe's
    shape: CellFill | None = None  # per-ink dot shape; falls back to the screen's
    ruling_lpi: float | None = None  # per-ink ruling; falls back to the screen's
    coverage: float | None = None  # target mean printed coverage over the canvas, e.g. a text block's gray

    @property
    def transmittance(self) -> np.ndarray:
        """Linear-RGB filter this ink applies to what's beneath it.

        Beer-Lambert: optical density scales with film thickness, so transmittance
        is the swatch raised to the density rather than a linear fade toward white.
        """
        return np.maximum(hex_to_linear(self.color), 1e-4) ** self.density

    @property
    def solid_reflectance(self) -> float:
        return float(luma_linear(self.transmittance))


def _key(names) -> frozenset[str]:
    return frozenset(names)


@dataclass
class InkSet:
    inks: list[Ink]
    overprint: dict[frozenset[str], str] = field(default_factory=dict)

    def __init__(self, *inks: Ink, overprint: dict | None = None):
        self.inks = list(inks)
        names = [i.name for i in self.inks]
        if len(set(names)) != len(names):
            raise ValueError("ink names must be unique")
        if len(self.inks) > 8:
            raise ValueError("at most 8 inks")
        self.overprint = {}
        for combo, color in (overprint or {}).items():
            k = _key(combo)
            if not k <= set(names) or len(k) < 2:
                raise ValueError(f"overprint {sorted(k)} must name 2+ inks from the set")
            self.overprint[k] = color

    def __iter__(self):
        return iter(self.inks)

    def __len__(self):
        return len(self.inks)

    def primaries(self, paper: str) -> np.ndarray:
        """Linear RGB for every ink combination, indexed by bitmask (bit i = inks[i]).

        Each combination starts from its largest overridden subset (or bare paper)
        and lays the remaining inks on in print order.
        """
        n = len(self.inks)
        paper_lin = hex_to_linear(paper)
        out = np.zeros((1 << n, 3))
        overrides = sorted(self.overprint.items(), key=lambda kv: -len(kv[0]))
        for mask in range(1 << n):
            present = {self.inks[i].name for i in range(n) if mask >> i & 1}
            base, done = paper_lin.copy(), set()
            for combo, color in overrides:
                if combo <= present:
                    base, done = hex_to_linear(color), set(combo)
                    break
            for ink in self.inks:
                if ink.name in present and ink.name not in done:
                    laid = base * ink.transmittance
                    # An opaque ink covers with its own color, whatever is under it (white on a black shirt).
                    base = (1 - ink.opacity) * laid + ink.opacity * ink.transmittance
            out[mask] = base
        return out

    def palette(self, paper: str) -> list[tuple[tuple[str, ...], str, bool]]:
        """(ink names, hex, overridden?) for paper, each ink, and each overprint."""
        prim = self.primaries(paper)
        rows = []
        for mask in range(1 << len(self.inks)):
            names = tuple(self.inks[i].name for i in range(len(self.inks)) if mask >> i & 1)
            rows.append((names, linear_to_hex(prim[mask]), _key(names) in self.overprint))
        return sorted(rows, key=lambda r: len(r[0]))

    def angle_separations(self) -> list[tuple[str, str, float]]:
        """Smallest angular distance for each ink pair, modulo the dot's 90-degree symmetry."""
        out = []
        for a, b in combinations(self.inks, 2):
            d = abs(a.angle - b.angle) % 90
            out.append((a.name, b.name, min(d, 90 - d)))
        return out
