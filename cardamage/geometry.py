"""Small polygon helpers; all coordinates are normalised to [0, 1]."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Poly = Sequence[Tuple[float, float]]
_GRID = 256


def poly_area(poly: Poly) -> float:
    """Shoelace area of a normalised polygon (= fraction of the image)."""
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def simplify(poly: Poly, eps: float = 0.002) -> List[Tuple[float, float]]:
    """Douglas-Peucker; eps is relative to image size. Keeps JSON small."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = np.asarray(poly)
    return [(round(float(x), 4), round(float(y), 4)) for x, y in approx]


def _raster(poly: Poly) -> np.ndarray:
    m = np.zeros((_GRID, _GRID), np.uint8)
    pts = (np.clip(np.asarray(poly), 0, 1) * (_GRID - 1)).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m


def overlap_fraction(inner: Poly, outer: Poly) -> float:
    """|inner ∩ outer| / |inner|."""
    a, b = _raster(inner), _raster(outer)
    denom = int(a.sum())
    return float((a & b).sum()) / denom if denom else 0.0


def box_to_poly(box: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
