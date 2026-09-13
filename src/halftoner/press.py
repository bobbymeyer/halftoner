"""Press artifacts, seeded. A press walks; it doesn't jitter.

Misregistration moves plates, not RGB channels: each ink gets a constant offset
plus a low-frequency drift across the sheet, and the key plate stays put, so in
a three-ink job two inks can be tight while one walks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import Curve


@dataclass
class Press:
    misregistration: float = 0.0  # mm; per-ink constant offset scale (0.15-0.6 reads right)
    drift: float = 0.35  # low-frequency walk amplitude as a fraction of misregistration
    slur: float = 0.0  # mm of smear along the press direction
    press_angle: float = 90.0  # degrees; 90 = sheet travels down the page
    extra_gain: float | Curve = 0.0  # uncorrected gain on press: 0.15 pushes 50% -> 65%
    key: str | None = None  # plate everything registers to; defaults to the first ink
    seed: int = 0

    @property
    def gain_curve(self) -> Curve | None:
        if isinstance(self.extra_gain, Curve):
            return self.extra_gain
        return Curve.gain(self.extra_gain) if self.extra_gain else None

    def offsets(self, inks, canvas):
        """{ink name: fn(x_mm, y_mm) -> (dx_mm, dy_mm)} plate displacement fields."""
        key = self.key or inks[0].name
        span = max(canvas.width_mm, canvas.height_mm)
        fields = {}
        for idx, ink in enumerate(inks):
            if ink.name == key or self.misregistration <= 0:
                fields[ink.name] = lambda x, y: (np.zeros_like(x), np.zeros_like(y))
                continue
            rng = np.random.default_rng([self.seed, idx])
            theta = rng.uniform(0, 2 * np.pi)
            mag = self.misregistration * rng.uniform(0.25, 1.0)
            const = np.array([np.cos(theta), np.sin(theta)]) * mag
            waves = [
                (rng.uniform(0, 2 * np.pi), 2 * np.pi / (span * rng.uniform(0.8, 2.5)), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi))
                for _ in range(3)
            ]
            amp = self.misregistration * self.drift / len(waves)

            def field(x, y, const=const, waves=waves, amp=amp):
                dx = np.full(np.shape(x), const[0])
                dy = np.full(np.shape(y), const[1])
                for direction, k, px, py in waves:
                    s = x * np.cos(direction) + y * np.sin(direction)
                    dx = dx + amp * np.sin(k * s + px)
                    dy = dy + amp * np.sin(k * s + py)
                return dx, dy

            fields[ink.name] = field
        return fields

    def slur_vector(self) -> tuple[float, float]:
        a = np.deg2rad(self.press_angle)
        return self.slur * np.cos(a), self.slur * np.sin(a)
