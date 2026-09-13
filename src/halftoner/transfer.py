"""The tone chain: source value -> nominal plate area -> printed area.

  tone (linear reflectance) --Yule-Nielsen vs. the ink's solid--> demanded area
  area --ink curve--> --tone compression--> --gain compensation--> --limits--> plate area
  plate area --substrate gain--> --press extra gain--> printed area

With n = 1 the tone step is Murray-Davies. Larger n models optical gain
(light scattering in the paper); ~1.5-2 is typical for uncoated stock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import Curve


def tone_to_area(reflectance, solid_reflectance: float, n: float = 1.0) -> np.ndarray:
    """Dot area that makes an ink with solid reflectance `solid_reflectance` average to `reflectance`.

    Both reflectances are relative to paper. Solves R^(1/n) = 1 - a (1 - Rs^(1/n)).
    """
    r = np.clip(np.asarray(reflectance, dtype=np.float64), 0.0, 1.0)
    rs = float(np.clip(solid_reflectance, 0.0, 0.999))
    return np.clip((1.0 - r ** (1.0 / n)) / (1.0 - rs ** (1.0 / n)), 0.0, 1.0)


def area_to_tone(area, solid_reflectance: float, n: float = 1.0) -> np.ndarray:
    a = np.clip(np.asarray(area, dtype=np.float64), 0.0, 1.0)
    return (1.0 - a * (1.0 - solid_reflectance ** (1.0 / n))) ** n


@dataclass(frozen=True)
class Transfer:
    yule_nielsen_n: float = 1.0
    compensate_gain: bool = True  # prepress inverts the substrate gain; False = uncorrected

    def plate_area(self, values, kind: str, ink, substrate, bias: float = 1.0) -> np.ndarray:
        """`bias` is a power on demanded area (endpoints fixed); coverage targeting solves for it."""
        a = tone_to_area(values, ink.solid_reflectance, self.yule_nielsen_n) if kind == "tone" else np.clip(values, 0, 1)
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
