"""Backend-neutral detection types. A backend turns BGR images into RawDets;
everything downstream (taxonomy, panel assignment, schema, rendering) is shared."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..schema import ModelInfo


@dataclass
class RawDet:
    label: str                                   # model's own label
    conf: float
    box: Tuple[float, float, float, float]       # x1, y1, x2, y2 normalised
    poly: Optional[List[Tuple[float, float]]]    # normalised outline, None for box-only models


class Backend(Protocol):
    info: ModelInfo

    def predict(self, images: Sequence[np.ndarray], conf: float, iou: float,
                max_det: int, device: str, half: bool) -> List[List[RawDet]]:
        """One list of RawDet per input image (BGR uint8)."""
        ...
