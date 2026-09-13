"""Recover the ink set from a scan.

At a sufficient scan resolution most pixels are one Neugebauer primary: bare
paper, a single ink, or an overprint. In optical-density space each primary is
a tight spike (as wide as scanner noise); pixels straddling dot edges are a
continuum of linear-light blends between primaries. So:

1. Estimate paper from the lightest pixels (or a margin box).
2. Find density modes: repeatedly take the densest sample and clear its
   neighbourhood. Many more modes than primaries are taken, because on a
   blurred scan the continuum near paper is denser than a small ink spike.
3. Discard modes that are blends of two or three other modes: primaries are
   the extreme points of the gamut, blends lie inside it.
4. Choose the n single inks as the subset whose subtractive combinations land
   nearest to observed modes. The leftover distance per overprint is reported:
   a large one means the overprint isn't simply subtractive (a chosen color,
   an opaque ink, or trapping). Overprints never seen are marked predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..color import hex_to_linear, linear_to_hex, luma_linear
from .common import to_linear

_EPS = 1e-4
MODE_WINDOW_D = 0.05  # neighbourhood radius in density for a primary's spike; about scanner noise
OBSERVED_D = 0.3  # a predicted primary within this of a mode counts as observed
MIX_TOL_D = 0.06  # a mode this close (in density) to a blend of other modes is a mixture, not a primary
MAX_MODES = 48


def density(lin: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Optical density relative to paper, per channel."""
    return -np.log10(np.clip(lin / paper, _EPS, 1.0))


def estimate_paper(u8: np.ndarray, box_px=None, rng=None, sample: int = 400_000) -> np.ndarray:
    if box_px is not None:
        y0, x0, h, w = box_px
        return np.median(to_linear(u8[y0 : y0 + h, x0 : x0 + w]).reshape(-1, 3), axis=0)
    px = _sample(u8, sample, rng)
    luma = luma_linear(px)
    return np.median(px[luma >= np.quantile(luma, 0.98)], axis=0)


def _sample(u8, n, rng) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    flat = u8.reshape(-1, 3)
    idx = rng.choice(flat.shape[0], size=min(n, flat.shape[0]), replace=False)
    return to_linear(flat[idx]).astype(np.float64)


def _densest(points: np.ndarray, r: float, rng, candidates: int = 600, reference: int = 12_000,
             batch: int = 100) -> np.ndarray:
    """Mean of the neighbourhood around the sample with the most neighbours within r."""
    ref = points if len(points) <= reference else points[rng.choice(len(points), reference, replace=False)]
    cand = ref if len(ref) <= candidates else ref[rng.choice(len(ref), candidates, replace=False)]
    counts = np.concatenate([
        (((cand[i : i + batch, None, :] - ref[None]) ** 2).sum(-1) < r * r).sum(1)
        for i in range(0, len(cand), batch)
    ])
    best = cand[np.argmax(counts)]
    return points[np.linalg.norm(points - best, axis=1) < r].mean(0)


def find_modes(x: np.ndarray, count: int, rng, r: float = MODE_WINDOW_D) -> tuple[np.ndarray, np.ndarray]:
    """Up to `count` density modes, densest first, with their neighbour counts."""
    modes, support = [], []
    remaining = x
    for _ in range(count):
        if len(remaining) < 50:
            break
        m = _densest(remaining, r, rng)
        modes.append(m)
        support.append(int((np.linalg.norm(x - m, axis=1) < r).sum()))
        remaining = remaining[np.linalg.norm(remaining - m, axis=1) >= 3 * r]
    return np.array(modes), np.array(support)


def mixture_modes(modes_d: np.ndarray, tol: float = MIX_TOL_D) -> set[int]:
    """Indices of modes that are linear-light blends of two or three other modes."""
    lin = 10 ** (-modes_d)  # relative to paper; blending is linear here
    n = len(lin)
    mixed = set()
    for j in range(n):
        others = np.array([i for i in range(n) if i != j])
        p = lin[j]
        pts = []
        if len(others) >= 2:
            pairs = np.array(list(combinations(others, 2)))
            a, b = lin[pairs[:, 0]], lin[pairs[:, 1]]
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
            pts.append(a + t[:, None] * ab)
        if len(others) >= 3:
            tri = np.array(list(combinations(others, 3)))
            a, b, c = lin[tri[:, 0]], lin[tri[:, 1]], lin[tri[:, 2]]
            M = np.stack([a - c, b - c], axis=2)  # (T, 3, 2)
            MtM = np.einsum("tij,tik->tjk", M, M) + 1e-12 * np.eye(2)
            w = np.linalg.solve(MtM, np.einsum("tij,ti->tj", M, p - c)[..., None])[..., 0]
            inside = (w >= 0).all(1) & (w.sum(1) <= 1)  # edges are already covered by the pairs
            if inside.any():
                pts.append((c + np.einsum("tij,tj->ti", M, w))[inside])
        if pts:
            q = np.concatenate(pts)
            if (np.linalg.norm(-np.log10(np.clip(q, _EPS, None)) - modes_d[j], axis=1) < tol).any():
                mixed.add(j)
    return mixed


def _bits(mask: int, n: int) -> list[int]:
    return [i for i in range(n) if mask >> i & 1]


@dataclass
class InkModel:
    paper: np.ndarray  # linear RGB
    names: list[str]
    solids: np.ndarray  # (n, 3) density of each ink's solid on this paper
    primaries: dict[int, np.ndarray]  # bitmask -> linear RGB (measured if observed, else predicted)
    coverage: dict[int, float]  # bitmask -> fraction of sampled pixels nearest this primary
    residual: dict[int, float]  # observed overprint bitmask -> density distance from subtractive prediction
    observed: set[int] = field(default_factory=set)

    @property
    def n(self) -> int:
        return len(self.names)

    def solid_linear(self, i: int) -> np.ndarray:
        return self.paper * 10 ** (-self.solids[i])

    def ink_hex(self, i: int) -> str:
        """The ink as it would appear on white paper."""
        return linear_to_hex(10 ** (-self.solids[i]))

    def label(self, mask: int) -> str:
        return "+".join(self.names[i] for i in _bits(mask, self.n)) or "paper"

    def amounts(self, lin: np.ndarray) -> np.ndarray:
        """Per-pixel ink amount (..., n): density unmixing, 0 = none, 1 = solid."""
        d = density(lin, self.paper)
        return np.clip(d @ np.linalg.pinv(self.solids.T).T, 0.0, 1.0).astype(np.float32)


def fit_inks(u8: np.ndarray, n_inks: int, paper: np.ndarray, rng=None, sample: int = 150_000,
             colors: list[str] | None = None, names: list[str] | None = None) -> InkModel:
    rng = rng or np.random.default_rng(0)
    if not 1 <= n_inks <= 4:
        raise ValueError("measure supports 1-4 inks")
    names = names or [f"ink{i + 1}" for i in range(n_inks)]
    if len(names) != n_inks:
        raise ValueError(f"{len(names)} names for {n_inks} inks")

    dens = density(_sample(u8, sample, rng), paper)
    k = 1 << n_inks
    modes, support = find_modes(dens, min(MAX_MODES, 4 * k + 8), rng)
    combos = [m for m in range(1, k) if len(_bits(m, n_inks)) >= 2]

    if colors:
        if len(colors) != n_inks:
            raise ValueError(f"{len(colors)} colors for {n_inks} inks")
        solids = np.array([density(hex_to_linear(c), np.ones(3)) for c in colors])
    else:
        paper_mode = int(np.argmin(np.linalg.norm(modes, axis=1)))
        mixed = mixture_modes(modes)
        others = [j for j in range(len(modes)) if j != paper_mode and j not in mixed]
        if len(others) < n_inks:  # too aggressive for this scan: fall back to every mode
            others = [j for j in range(len(modes)) if j != paper_mode]
        if len(others) < n_inks:
            raise ValueError(f"found {len(others)} ink-like modes for {n_inks} inks; is the scan screened and in color?")
        best = None
        for S in combinations(others, n_inks):
            cost = 0.0
            if combos:
                preds = np.array([modes[list(S)][_bits(m, n_inks)].sum(0) for m in combos])
                nearest = np.linalg.norm(preds[:, None, :] - modes[None], axis=2).min(1)
                cost = float(np.minimum(nearest, OBSERVED_D).sum())
            key = (round(cost, 3), -int(support[list(S)].sum()))
            if best is None or key < best[0]:
                best = (key, S)
        S = best[1]
        # Darkest ink first: it's the usual key plate.
        solids = np.array(sorted((modes[j] for j in S), key=lambda d: -d.sum()))

    preds = np.array([solids[_bits(m, n_inks)].sum(0) if m else np.zeros(3) for m in range(k)])
    primaries, residual, observed = {}, {}, set()
    measured = preds.copy()
    for m in range(k):
        dist = np.linalg.norm(modes - preds[m], axis=1)
        j = int(np.argmin(dist))
        if dist[j] <= OBSERVED_D:
            observed.add(m)
            measured[m] = modes[j]
            if m in combos:
                residual[m] = float(dist[j])
        primaries[m] = paper * 10 ** (-measured[m])
    label = np.argmin(np.linalg.norm(dens[:, None, :] - measured[None], axis=2), axis=1)
    counts = np.bincount(label, minlength=k)
    coverage = {m: float(counts[m] / counts.sum()) for m in range(k)}
    return InkModel(np.asarray(paper, dtype=np.float64), names, solids, primaries, coverage, residual, observed)
