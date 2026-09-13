"""Tone: effective coverage on flat patches, gain pairs, and where dots drop or close.

Coverage is Murray-Davies on the channel where the solid is densest, the way a
densitometer uses the complementary filter. It is *effective* coverage: optical
gain in the paper is included, so a profile built from it should use
yule_nielsen_n = 1 rather than counting that gain twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..render.film import WEDGE, WEDGE_H_MM
from .common import crop_linear, mm_box_to_px


@dataclass
class PatchReading:
    nominal: float | None
    box_mm: tuple[float, float, float, float]
    reflectance: float  # measuring channel, linear
    coverage: float


def wedge_boxes(x_mm: float, y_mm: float, patch_w_mm: float, patch_h_mm: float = WEDGE_H_MM,
                values=WEDGE) -> list[tuple[tuple[float, float, float, float], float]]:
    """Patch boxes for a halftoner film step wedge whose top-left corner is at (x_mm, y_mm) on the scan."""
    return [((x_mm + k * patch_w_mm, y_mm, patch_w_mm, patch_h_mm), v / 100) for k, v in enumerate(values)]


def _mean_rgb(u8, dpi, box_mm, inset: float) -> np.ndarray:
    x, y, w, h = box_mm
    inner = (x + w * inset, y + h * inset, w * (1 - 2 * inset), h * (1 - 2 * inset))
    y0, x0, ph, pw = mm_box_to_px(inner, dpi)
    return crop_linear(u8, y0, x0, ph, pw).reshape(-1, 3).mean(axis=0)


def measure_patches(u8: np.ndarray, dpi: float, patches, paper: np.ndarray, solid: np.ndarray | None = None,
                    inset: float = 0.25) -> list[PatchReading]:
    """patches: [(box_mm, nominal or None)]. A 0 patch overrides paper; a 1.0 patch overrides solid."""
    rgb = [(_mean_rgb(u8, dpi, box, inset), box, nom) for box, nom in patches]
    for mean, _, nom in rgb:
        if nom == 0:
            paper = mean
        if nom == 1:
            solid = mean
    if solid is None:
        raise ValueError("need a 100% patch or a solid ink color to measure coverage against")
    ch = int(np.argmax(-np.log10(np.clip(solid / paper, 1e-4, 1))))
    rp, rs = float(paper[ch]), float(solid[ch])
    return [
        PatchReading(nom, tuple(box), float(mean[ch]), float(np.clip((1 - mean[ch] / rp) / (1 - rs / rp), 0, 1.05)))
        for mean, box, nom in rgb
    ]


def gain_pairs(readings: list[PatchReading]) -> list[list[float]]:
    return [[round(r.nominal, 4), round(min(r.coverage, 1.0), 4)]
            for r in readings if r.nominal is not None and 0 < r.nominal < 1]


def tone_limits(readings: list[PatchReading], drop: float = 0.01, close: float = 0.99) -> tuple[float | None, float | None]:
    """(smallest nominal dot that holds, largest nominal dot whose white stays open)."""
    tints = sorted((r for r in readings if r.nominal is not None and 0 < r.nominal < 1), key=lambda r: r.nominal)
    holds = [r.nominal for r in tints if r.coverage > drop]
    opens = [r.nominal for r in tints if r.coverage < close]
    return (holds[0] if holds else None), (opens[-1] if opens else None)
