"""Monotone 1-D curves on [0, 1]: gain curves, ink transfer curves, anti-curves.

A curve is a piecewise-linear LUT so measured data (a step wedge read with a
densitometer) and analytic models share one type, and inversion is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Curve:
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    spec: object = field(default=None, compare=False, repr=False)  # compact form it was built from

    def __post_init__(self):
        if len(self.xs) != len(self.ys) or len(self.xs) < 2:
            raise ValueError("curve needs >= 2 matching points")
        if any(b < a for a, b in zip(self.xs, self.xs[1:])):
            raise ValueError("curve xs must be non-decreasing")

    def __call__(self, v):
        return np.interp(v, self.xs, self.ys)

    @classmethod
    def identity(cls) -> Curve:
        return cls((0.0, 1.0), (0.0, 1.0))

    @classmethod
    def from_points(cls, points) -> Curve:
        """Measured pairs (nominal, printed). Endpoints are pinned to 0 and 1."""
        given = [[float(a), float(b)] for a, b in points]
        pts = sorted({(0.0, 0.0), (1.0, 1.0), *(tuple(p) for p in given)})
        return cls(tuple(p[0] for p in pts), tuple(p[1] for p in pts), given)

    @classmethod
    def gain(cls, at_50: float, samples: int = 257) -> Curve:
        """Parabolic tone-value increase peaking at the midtone.

        `at_50` is printed-minus-nominal at a 50% dot: 0.15 turns 50% into 65%.
        """
        x = np.linspace(0, 1, samples)
        y = np.clip(x + at_50 * 4 * x * (1 - x), 0, 1)
        return cls(tuple(float(v) for v in x), tuple(float(v) for v in np.maximum.accumulate(y)), float(at_50))

    def inverse(self) -> Curve:
        ys = np.maximum.accumulate(np.asarray(self.ys))
        return Curve(tuple(ys), tuple(self.xs))

    def then(self, other: Curve, samples: int = 257) -> Curve:
        """Composition: apply self, then other."""
        x = np.linspace(0, 1, samples)
        return Curve(tuple(x), tuple(other(self(x))))

    def to_spec(self):
        """JSON form: null (identity), a number (gain at 50%), [[nominal, printed], ...], or {xs, ys}."""
        if self.spec is not None:
            return self.spec
        if self.xs == (0.0, 1.0) and self.ys == (0.0, 1.0):
            return None
        return {"xs": list(self.xs), "ys": list(self.ys)}

    @classmethod
    def from_spec(cls, spec) -> Curve:
        if spec is None:
            return cls.identity()
        if isinstance(spec, Curve):
            return spec
        if isinstance(spec, (int, float)):
            return cls.gain(spec)
        if isinstance(spec, dict):
            return cls(tuple(spec["xs"]), tuple(spec["ys"]))
        return cls.from_points(spec)
