"""Cell fills: dot area -> geometry inside one screen cell.

Every shape is a spot function f(u, v) over the cell, u, v in [-0.5, 0.5).
A point is inked when f <= threshold. The threshold for a requested area is
the area-quantile of f over the cell, computed numerically once per shape, so
area is exact for any shape and join points fall out of the same table rather
than being hand-derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_TABLE_N = 1024


@dataclass(frozen=True, eq=False)
class CellFill:
    name: str
    spot: Callable[[np.ndarray, np.ndarray], np.ndarray]
    kind: str | None = None  # key in SHAPES; None for custom spot functions (not serializable)
    params: dict = field(default_factory=dict)
    _sorted: np.ndarray = field(init=False, repr=False)

    def to_spec(self) -> dict:
        if self.kind is None:
            raise TypeError(f"custom shape {self.name!r} can't be serialized")
        return {"shape": self.kind, **self.params}

    @staticmethod
    def from_spec(spec) -> CellFill:
        if isinstance(spec, CellFill):
            return spec
        if isinstance(spec, str):
            return SHAPES[spec]()
        params = dict(spec)
        return SHAPES[params.pop("shape")](**params)

    def __post_init__(self):
        g = (np.arange(_TABLE_N) + 0.5) / _TABLE_N - 0.5
        u, v = np.meshgrid(g, g)
        object.__setattr__(self, "_sorted", np.sort(self.spot(u, v).ravel()))

    def threshold(self, area) -> np.ndarray:
        """Spot-function threshold that inks exactly `area` of the cell."""
        a = np.clip(np.asarray(area, dtype=np.float64), 0.0, 1.0)
        s = self._sorted
        # f <= s[k] covers (k + 1) / N of the cell.
        k = np.clip(np.ceil(a * s.size).astype(int) - 1, 0, s.size - 1)
        thr = np.where(a <= 0, -np.inf, s[k])
        return np.where(a >= 1, np.inf, thr)

    def area_at(self, threshold: float) -> float:
        return float(np.searchsorted(self._sorted, threshold, side="right") / self._sorted.size)

    def join_points(self) -> list[float]:
        """Areas at which the dot touches the u edges, the v edges, and the corners.

        Edge contact is where neighbouring dots join (twice for chain dots);
        corner contact is where the white between them closes. Ascending, de-duplicated.
        """
        probes = {
            "u": (np.array([0.5, -0.5]), np.array([0.0, 0.0])),
            "v": (np.array([0.0, 0.0]), np.array([0.5, -0.5])),
            "corner": (np.array([0.5, -0.5, 0.5, -0.5]), np.array([0.5, 0.5, -0.5, -0.5])),
        }
        pts = {round(self.area_at(float(self.spot(u, v).min())), 3) for u, v in probes.values()}
        return sorted(p for p in pts if 0 < p < 1)


def Round() -> CellFill:
    """Circular dot. Neighbours join at pi/4 = 78.5%."""
    return CellFill("round", lambda u, v: np.hypot(u, v), "round")


def Square() -> CellFill:
    """Checkerboard dot: corners meet neighbours at 50%."""
    return CellFill("square", lambda u, v: np.abs(u) + np.abs(v), "square")


def Elliptical(ratio: float = 1.6) -> CellFill:
    """Chain dot. Joins along the long axis first, then the short axis."""
    return CellFill(f"elliptical({ratio:g})", lambda u, v: np.hypot(u, v * ratio), "elliptical", {"ratio": ratio})


def Diamond(ratio: float = 1.6) -> CellFill:
    """Elongated rhombus chain dot: sharp-cornered elliptical."""
    return CellFill(f"diamond({ratio:g})", lambda u, v: np.abs(u) + np.abs(v) * ratio, "diamond", {"ratio": ratio})


def Line() -> CellFill:
    """Line screen: bands parallel to the screen's u axis."""
    return CellFill("line", lambda u, v: np.abs(v) + 1e-9 * np.abs(u), "line")


def Custom(name: str, spot: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> CellFill:
    return CellFill(name, spot)


SHAPES = {"round": Round, "square": Square, "elliptical": Elliptical, "diamond": Diamond, "line": Line}
