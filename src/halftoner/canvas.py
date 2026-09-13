"""Physical canvas. Ruling (lpi) is meaningless without a declared physical size."""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4
_UNIT_TO_MM = {"mm": 1.0, "in": MM_PER_INCH, "cm": 10.0}


@dataclass(frozen=True)
class Canvas:
    width_mm: float
    height_mm: float
    dpi: float = 300.0
    bleed_mm: float = 0.0

    @classmethod
    def of(cls, width: float, height: float, unit: str = "mm", dpi: float = 300.0, bleed: float = 0.0) -> Canvas:
        k = _UNIT_TO_MM[unit]
        return cls(width * k, height * k, dpi, bleed * k)

    @property
    def px_per_mm(self) -> float:
        return self.dpi / MM_PER_INCH

    @property
    def size_px(self) -> tuple[int, int]:
        return round(self.width_mm * self.px_per_mm), round(self.height_mm * self.px_per_mm)

    @property
    def origin(self) -> tuple[float, float]:
        """Screen origin defaults to the canvas corner, not the image."""
        return (0.0, 0.0)
