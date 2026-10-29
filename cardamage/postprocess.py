"""Post-processing shared by the pipeline and the evaluation script."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .backends.base import RawDet
from .geometry import box_to_poly

_G = 160  # raster size for mask overlap; plenty for IoU/containment on normalised polygons


def _mask(d: RawDet) -> np.ndarray:
    m = np.zeros((_G, _G), np.uint8)
    pts = (np.clip(np.asarray(d.poly or box_to_poly(d.box)), 0, 1) * (_G - 1)).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def dedupe(dets: List[RawDet], iou_thr: float = 0.5, containment_thr: float = 0.85) -> List[RawDet]:
    """Drop lower-confidence detections of the SAME label that overlap a kept one.

    RF-DETR is NMS-free, so one damage region often comes back as several overlapping masks. A detection
    is suppressed when mask IoU >= iou_thr, or when >= containment_thr of its own area lies inside a
    higher-confidence kept mask (a small duplicate sitting inside a bigger one). Different labels are
    never merged: a scratch on top of a dent is a legitimate pair. Set a threshold to 0 or None to
    disable that rule.
    """
    if len(dets) < 2 or (not iou_thr and not containment_thr):
        return list(dets)
    kept: List[RawDet] = []
    masks: List[np.ndarray] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        m = _mask(d)
        area = int(m.sum())
        dup = False
        for k, km in zip(kept, masks):
            if k.label != d.label or area == 0:
                continue
            inter = int((m & km).sum())
            union = area + int(km.sum()) - inter
            if (iou_thr and union and inter / union >= iou_thr) or \
               (containment_thr and inter / area >= containment_thr):
                dup = True
                break
        if not dup:
            kept.append(d)
            masks.append(m)
    return kept
