"""Screens: a rotated grid over the canvas, and the per-ink plate built on it.

A Plate is the separation: one nominal area per cell. Rasterizer, SVG writer
and report all read the same plate, so what's reported is what's drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cellfill import CellFill, Round
from .canvas import MM_PER_INCH


@dataclass
class Screen:
    ruling_lpi: float
    shape: CellFill = field(default_factory=Round)
    origin: tuple[float, float] = (0.0, 0.0)  # mm; canvas corner, not the image
    phase: tuple[float, float] = (0.0, 0.0)  # fraction of a cell


@dataclass
class Plate:
    ink: object
    shape: CellFill
    pitch_mm: float
    angle_deg: float
    origin: tuple[float, float]
    phase: tuple[float, float]
    i0: int
    j0: int
    area: np.ndarray  # (nj, ni) nominal plate area
    printed: np.ndarray  # (nj, ni) area after gain
    bias: float = 1.0  # power on demanded area chosen to hit the ink's coverage target

    @property
    def axes(self):
        a = np.deg2rad(self.angle_deg)
        # Counterclockwise on a y-down page.
        return np.array([np.cos(a), -np.sin(a)]), np.array([np.sin(a), np.cos(a)])

    def to_screen(self, x, y):
        uax, vax = self.axes
        dx, dy = x - self.origin[0], y - self.origin[1]
        return dx * uax[0] + dy * uax[1], dx * vax[0] + dy * vax[1]

    def to_canvas(self, s, t):
        uax, vax = self.axes
        return self.origin[0] + s * uax[0] + t * vax[0], self.origin[1] + s * uax[1] + t * vax[1]

    def centers(self):
        nj, ni = self.area.shape
        s = (np.arange(ni) + self.i0 + 0.5 + self.phase[0]) * self.pitch_mm
        t = (np.arange(nj) + self.j0 + 0.5 + self.phase[1]) * self.pitch_mm
        S, T = np.meshgrid(s, t)
        return self.to_canvas(S, T)

    def locate(self, x, y):
        """Cell-array indices (row, col, valid) and in-cell coords u, v for canvas points."""
        s, t = self.to_screen(x, y)
        cs = s / self.pitch_mm - self.phase[0]
        ct = t / self.pitch_mm - self.phase[1]
        ci, cj = np.floor(cs), np.floor(ct)
        col, row = ci.astype(np.int64) - self.i0, cj.astype(np.int64) - self.j0
        nj, ni = self.area.shape
        valid = (col >= 0) & (col < ni) & (row >= 0) & (row < nj)
        return np.clip(row, 0, nj - 1), np.clip(col, 0, ni - 1), valid, cs - ci - 0.5, ct - cj - 0.5


def cell_range(canvas, pitch_mm, angle_deg, origin, phase, margin_mm):
    probe = Plate(None, None, pitch_mm, angle_deg, origin, phase, 0, 0, np.zeros((1, 1)), np.zeros((1, 1)))
    xs = np.array([-margin_mm, canvas.width_mm + margin_mm])
    ys = np.array([-margin_mm, canvas.height_mm + margin_mm])
    X, Y = np.meshgrid(xs, ys)
    s, t = probe.to_screen(X, Y)
    i0 = int(np.floor(s.min() / pitch_mm - phase[0])) - 1
    i1 = int(np.ceil(s.max() / pitch_mm - phase[0])) + 1
    j0 = int(np.floor(t.min() / pitch_mm - phase[1])) - 1
    j1 = int(np.ceil(t.max() / pitch_mm - phase[1])) + 1
    return i0, j0, i1 - i0, j1 - j0


def lpi_to_pitch_mm(lpi: float) -> float:
    return MM_PER_INCH / lpi
