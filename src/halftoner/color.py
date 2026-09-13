"""Color encoding helpers. All compositing happens in linear light."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def hex_to_srgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def hex_to_linear(h: str) -> np.ndarray:
    return srgb_to_linear(hex_to_srgb(h))


def linear_to_hex(rgb) -> str:
    s = np.round(linear_to_srgb(rgb) * 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*s)


def cmyk_from_linear(rgb_linear) -> tuple[float, float, float, float]:
    """Naive CMYK from linear RGB: fine for a spot color's proofing alternate, not for color matching."""
    r, g, b = (float(v) for v in linear_to_srgb(rgb_linear))
    k = 1.0 - max(r, g, b)
    if k >= 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return ((1 - r - k) / (1 - k), (1 - g - k) / (1 - k), (1 - b - k) / (1 - k), k)


def luma_linear(rgb_linear: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance from linear RGB (last axis = channels)."""
    return rgb_linear @ np.array([0.2126, 0.7152, 0.0722])
