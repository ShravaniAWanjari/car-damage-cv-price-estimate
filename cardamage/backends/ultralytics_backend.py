from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..loader import load_yolo
from ..schema import ModelInfo
from .base import RawDet


class UltralyticsBackend:
    """Weights converted by scripts/safe_convert.py. AGPL-3.0 code: see docs/MODEL_SELECTION.md."""

    def __init__(self, name: str, model_dir: Path):
        self.yolo, self.meta = load_yolo(model_dir)
        self.imgsz = int(self.meta["imgsz"] or 640)
        names = self.yolo.model.names
        self.info = ModelInfo(name=name, task=self.meta["task"], sha256=self.meta["source_sha256"],
                              classes=[names[i] for i in sorted(names)])

    def predict(self, images: Sequence[np.ndarray], conf: float, iou: float,
                max_det: int, device: str, half: bool) -> List[List[RawDet]]:
        extra = {"half": True} if half else {}  # passing half=False triggers a deprecation warning
        results = self.yolo.predict(list(images), imgsz=self.imgsz, conf=conf, iou=iou,
                                    max_det=max_det, device=device, verbose=False, **extra)
        return [self._raw(r) for r in results]

    @staticmethod
    def _raw(r) -> List[RawDet]:
        out = []
        for i in range(len(r.boxes)):
            poly = None
            if r.masks is not None and i < len(r.masks.xyn) and len(r.masks.xyn[i]) >= 3:
                poly = [(float(a), float(b)) for a, b in r.masks.xyn[i]]
            out.append(RawDet(label=r.names[int(r.boxes.cls[i])], conf=float(r.boxes.conf[i]),
                              box=tuple(r.boxes.xyxyn[i].tolist()), poly=poly))
        return out
