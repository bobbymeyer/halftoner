"""Dot geometry shared by the vector writers (SVG, PDF), in canvas millimetres.

Round dots below their join are circles. Everything else (joined dots, other
shapes) is traced as a polygon by marching rays out from the cell center to the
spot-function threshold, clipped to the cell, with the cell corners always among
the rays so clipped edges stay square. The ray count follows a chord tolerance,
so fine rulings don't carry more vertices than the plate can resolve. Vertices
on a straight run are flagged so writers can drop them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAX_RAYS = 48
_MIN_RAYS = 16
_BISECT = 18


def _ray_angles(n: int) -> np.ndarray:
    base = np.linspace(0, 2 * np.pi, n, endpoint=False)
    corners = np.array([1, 3, 5, 7]) * np.pi / 4
    # Round before merging: a base ray and a corner ray at the same angle must be one
    # ray, or the corner vertex repeats and straight-run removal deletes both copies.
    return np.unique(np.round(np.concatenate([base, corners]), 9))


def rays_for(pitch_mm: float, tolerance_mm: float | None) -> int:
    """Rays needed so a traced curve strays from the true edge by at most `tolerance_mm`."""
    if tolerance_mm is None:
        return _MAX_RAYS
    reach = pitch_mm * np.sqrt(0.5)  # the farthest a dot's edge gets from its cell center
    step = 2 * np.arccos(max(-1.0, 1.0 - tolerance_mm / reach))
    n = int(np.ceil(2 * np.pi / step / 8)) * 8  # multiples of 8 keep the axis and diagonal rays
    return int(np.clip(n, _MIN_RAYS, _MAX_RAYS))


def _trace(shape, thr: np.ndarray, n_rays: int = _MAX_RAYS) -> np.ndarray:
    """(cells, rays, 2) polygon vertices in cell units for thresholds `thr` (cells,)."""
    ang = _ray_angles(n_rays)
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


_CROSSINGS = 8  # most places a dot's outline meets its cell edge (a clipped square dot has 8)
_EDGE_SAMPLES = 64  # per side, for finding them


def _perimeter(s: np.ndarray) -> np.ndarray:
    """Cell boundary at parameter s in [0, 4), counterclockwise from the middle of the right edge's bottom."""
    s = np.mod(s, 4.0)
    side, t = np.floor(s), s - np.floor(s)
    u = np.select([side == 0, side == 1, side == 2], [np.full_like(t, 0.5), 0.5 - t, np.full_like(t, -0.5)], -0.5 + t)
    v = np.select([side == 0, side == 1, side == 2], [-0.5 + t, np.full_like(t, 0.5), 0.5 - t], np.full_like(t, -0.5))
    return np.stack([u, v], axis=-1)


def _boundary_crossings(shape, thr: np.ndarray, bisect: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Where each cell's dot outline meets the cell edge: (cells, K, 2) points and a (cells, K) valid mask.

    Rays only put vertices at fixed angles, so a joined dot's outline bends at the
    cell edge between rays and would lose its corners. These are those bends.
    """
    n = 4 * _EDGE_SAMPLES
    s = np.arange(n) / _EDGE_SAMPLES
    p = _perimeter(s)
    inside = shape.spot(p[:, 0], p[:, 1])[None, :] <= thr[:, None]  # spot along the edge is the same for every cell
    change = inside != np.roll(inside, -1, axis=1)
    order = np.argsort(~change, axis=1, kind="stable")[:, :_CROSSINGS]
    valid = np.take_along_axis(change, order, axis=1)
    lo = s[order]
    hi = lo + 1.0 / _EDGE_SAMPLES
    lo_inside = np.take_along_axis(inside, order, axis=1)
    t = thr[:, None]
    for _ in range(bisect):
        mid = (lo + hi) / 2
        q = _perimeter(mid)
        same = (shape.spot(q[..., 0], q[..., 1]) <= t) == lo_inside
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    return _perimeter((lo + hi) / 2), valid


@dataclass
class DotGeometry:
    circles: np.ndarray  # (N, 3): cx, cy, r in mm
    polygons: np.ndarray  # (M, R, 2): vertices in mm
    keep: np.ndarray  # (M, R): False for vertices on a straight run between their neighbours

    @property
    def count(self) -> int:
        return len(self.circles) + len(self.polygons)


def dot_geometry(plate, canvas, part=None, use_printed: bool = False, offset=None, margin_mm: float = 0.0,
                 tolerance_mm: float | None = None) -> DotGeometry:
    """Dots for one plate (or one layer of it) whose cells reach within `margin_mm` of the canvas."""
    src = part if part is not None else plate
    area = src.printed if use_printed else src.area
    X, Y = plate.centers()
    pad = plate.pitch_mm + margin_mm
    near = (area > 0) & (X > -pad) & (X < canvas.width_mm + pad) & (Y > -pad) & (Y < canvas.height_mm + pad)
    if offset is not None:
        dx, dy = offset(X, Y)
        X, Y = X + dx, Y + dy
    a, cx, cy = area[near], X[near], Y[near]

    circle = np.zeros(a.shape, dtype=bool)
    if plate.shape.name == "round":
        circle = a < np.pi / 4
    circles = np.column_stack([cx[circle], cy[circle], np.sqrt(a[circle] / np.pi) * plate.pitch_mm])

    poly = ~circle
    if not poly.any():
        return DotGeometry(circles, np.zeros((0, 0, 2)), np.zeros((0, 0), dtype=bool))
    rays = rays_for(plate.pitch_mm, tolerance_mm)
    thr = plate.shape.threshold(a[poly])
    ang = _ray_angles(rays)
    ray_v = _trace(plate.shape, thr, rays)
    cross_v, valid = _boundary_crossings(plate.shape, thr)
    cross_ang = np.mod(np.arctan2(cross_v[..., 1], cross_v[..., 0]), 2 * np.pi)
    on_ray = (np.abs(np.mod(cross_ang[..., None] - ang + np.pi, 2 * np.pi) - np.pi) < 1e-7).any(axis=-1)
    valid &= ~on_ray  # a crossing on a ray is already that ray's vertex
    # Unused slots sit on the first ray edge's midpoint: collinear, so straight-run removal drops them.
    filler = (ray_v[:, 0] + ray_v[:, 1]) / 2
    cross_v = np.where(valid[..., None], cross_v, filler[:, None, :])
    cross_ang = np.where(valid, cross_ang, (ang[0] + ang[1]) / 2)
    all_v = np.concatenate([ray_v, cross_v], axis=1)
    all_ang = np.concatenate([np.broadcast_to(ang, ray_v.shape[:2]), cross_ang], axis=1)
    order = np.argsort(all_ang, axis=1, kind="stable")
    verts = np.take_along_axis(all_v, order[..., None], axis=1) * plate.pitch_mm
    uax, vax = plate.axes
    px = cx[poly, None] + verts[..., 0] * uax[0] + verts[..., 1] * vax[0]
    py = cy[poly, None] + verts[..., 0] * uax[1] + verts[..., 1] * vax[1]
    polygons = np.stack([px, py], axis=-1)

    prev, nxt = np.roll(polygons, 1, axis=1), np.roll(polygons, -1, axis=1)
    cross = ((polygons[..., 0] - prev[..., 0]) * (nxt[..., 1] - polygons[..., 1])
             - (polygons[..., 1] - prev[..., 1]) * (nxt[..., 0] - polygons[..., 0]))
    keep = np.abs(cross) > 1e-6 * plate.pitch_mm**2
    keep[keep.sum(axis=1) < 3] = True  # degenerate dots keep every vertex
    return DotGeometry(circles, polygons, keep)
