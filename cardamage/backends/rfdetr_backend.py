"""RF-DETR-Seg backend (Apache-2.0). Loads a model directory written by
scripts/finetune_rfdetr.py:

    model_dir/checkpoint_best_total.pth   # weights-only checkpoint
    model_dir/meta.json                   # class names, size, resolution, training provenance

Checkpoints are loaded with rfdetr's safe loader (weights_only=True); we never pass
trust_checkpoint. RF-DETR is NMS-free, so `iou` is unused; `max_det` keeps the top-scoring.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..schema import ModelInfo
from .base import RawDet

CKPT = "checkpoint_best_total.pth"


def mask_to_polygon(mask: np.ndarray, min_pixels: int = 12) -> Optional[List[tuple]]:
    """Largest external contour of a boolean mask, normalised to [0,1]."""
    h, w = mask.shape
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_pixels and len(c) >= 3]
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return [(float(x) / w, float(y) / h) for x, y in c]


class RFDETRBackend:
    def __init__(self, name: str, model_dir: Path, device: str = "cpu"):
        from rfdetr import RFDETR  # heavy import only when this backend is used

        ckpt = model_dir / CKPT
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found - run scripts/finetune_rfdetr.py first")
        meta = json.loads((model_dir / "meta.json").read_text()) if (model_dir / "meta.json").exists() else {}
        self.model = RFDETR.from_checkpoint(ckpt, device=device)
        names = meta.get("class_names") or self.model.class_names
        self.meta = meta
        self.info = ModelInfo(name=name, task="segment", classes=list(names),
                              sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())

    def predict(self, images: Sequence[np.ndarray], conf: float, iou: float,
                max_det: int, device: str, half: bool) -> List[List[RawDet]]:
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
        dets = self.model.predict(rgb, threshold=conf)
        dets = [dets] if not isinstance(dets, list) else dets
        return [self._raw(d, im.shape[1], im.shape[0], max_det) for d, im in zip(dets, images)]

    @staticmethod
    def _raw(d, w: int, h: int, max_det: int) -> List[RawDet]:
        order = np.argsort(-d.confidence)[:max_det] if len(d) else []
        names = d.data.get("class_name")
        out = []
        for i in order:
            if str(names[i]).startswith("__"):  # RF-DETR's background slot ("__background__"): not a class
                continue
            x1, y1, x2, y2 = d.xyxy[i]
            poly = mask_to_polygon(d.mask[i]) if d.mask is not None else None
            out.append(RawDet(label=str(names[i]), conf=float(d.confidence[i]),
                              box=(float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h), poly=poly))
        return out
