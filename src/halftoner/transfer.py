"""The tone chain: source value -> nominal plate area -> printed area.

  tone (linear reflectance) --between base and solid--> demanded area
  area --ink curve--> --tone compression--> --gain compensation--> --limits--> plate area
  plate area --substrate gain--> --press extra gain--> printed area

With n = 1 the tone step is Murray-Davies. Larger n models optical gain
(light scattering in the paper); ~1.5-2 is typical for uncoated stock.

Tone range: an ink moves reflectance between the bare base (paper or garment)
and its own solid as printed (with its opacity, over any underbase).
  "paper"  image white is the base; a dark ink darkens it, and tones darker
           than its solid clip. The classic case: ink on paper.
  "range"  image white and black map to the lighter and darker of base and
           solid. The only sensible reading when a light ink sits on a dark
           garment, and a way to spread a light ink's few tones across a photo.
  "auto"   "range" when the solid is lighter than the base, else "paper".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import Curve

TONE_MODES = ("auto", "paper", "range")


def tone_to_area(reflectance, solid_reflectance: float, n: float = 1.0, base_reflectance: float = 1.0,
                 mode: str = "paper") -> np.ndarray:
    """Dot area that averages to `reflectance` between the base (area 0) and the solid (area 1).

    Solves R^(1/n) = (1 - a) B^(1/n) + a S^(1/n). In "paper" mode `reflectance` is relative
    to the base; in "range" mode 0..1 spans the darker..lighter of base and solid.
    """
    t = np.clip(np.asarray(reflectance, dtype=np.float64), 0.0, 1.0)
    b, s = max(float(base_reflectance), 0.0), max(float(solid_reflectance), 0.0)  # zero is fine: powers are positive
    r = min(b, s) + t * abs(s - b) if mode == "range" else t * b
    k = 1.0 / n
    denom = s**k - b**k
    if abs(denom) < 1e-12:
        return np.zeros(np.shape(t))
    return np.clip((r**k - b**k) / denom, 0.0, 1.0)


def area_to_tone(area, solid_reflectance: float, n: float = 1.0, base_reflectance: float = 1.0) -> np.ndarray:
    """Reflectance relative to the base for a dot area (the inverse of paper mode)."""
    a = np.clip(np.asarray(area, dtype=np.float64), 0.0, 1.0)
    k = 1.0 / n
    b = base_reflectance
    return ((1 - a) * b**k + a * solid_reflectance**k) ** n / b


@dataclass(frozen=True)
class ToneRange:
    base: float  # luminance of the bare substrate
    solid: float  # luminance of a solid of the ink as printed


@dataclass(frozen=True)
class Transfer:
    yule_nielsen_n: float = 1.0
    compensate_gain: bool = True  # prepress inverts the substrate gain; False = uncorrected
    tone_range: str = "auto"  # "auto" | "paper" | "range": see the module docstring

    def __post_init__(self):
        if self.tone_range not in TONE_MODES:
            raise ValueError(f"tone_range must be one of {', '.join(TONE_MODES)}")

    def tone_mode(self, tone: ToneRange) -> str:
        if self.tone_range != "auto":
            return self.tone_range
        return "range" if tone.solid > tone.base else "paper"

    def plate_area(self, values, kind: str, ink, substrate, bias: float = 1.0, tone: ToneRange | None = None) -> np.ndarray:
        """`bias` is a power on demanded area (endpoints fixed); coverage targeting solves for it.

        `tone` is the ink's range on this job; without one, the ink is taken on white paper.
        """
        if kind == "tone":
            tone = tone or ToneRange(1.0, ink.solid_reflectance)
            a = tone_to_area(values, tone.solid, self.yule_nielsen_n, tone.base, self.tone_mode(tone))
        else:
            a = np.clip(values, 0, 1)
        a = a**bias
        empty = a <= 0.0  # no ink on the plate stays no ink through compression
        a = ink.curve(a)
        if substrate.compression:
            lo, hi = substrate.compression
            a = lo + a * (hi - lo)
        if self.compensate_gain:
            a = substrate.gain.inverse()(a)
        a = np.where(a < substrate.min_dot, 0.0, a)  # highlight drop
        a = np.where(a > substrate.max_dot, 1.0, a)  # shadow snap
        return np.where(empty, 0.0, a)

    @staticmethod
    def printed_area(plate, substrate, press_gain: Curve | None = None) -> np.ndarray:
        a = substrate.gain(plate)
        return press_gain(a) if press_gain is not None else a
