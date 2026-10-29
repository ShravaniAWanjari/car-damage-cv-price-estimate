"""Optional car-region filter: drop damage detections that are not on a vehicle.

The damage model never saw a damage-free photo, so on street scenes it can label pedestrians, signs or
shadows as damage (v2 flagged people in yellow clothes as `paint_damage` at 0.9). This runs the
COCO-pretrained RF-DETR-Seg (Apache-2.0; same package and safe weights loader as the damage backend, no
`trust_checkpoint`) once per image and keeps a damage only if it lies on a vehicle.

Rules (per damage, fractions of its own mask area; see gate()):
  * on a person (>= `max_in_person` inside a person mask, not on the vehicle): dropped. Default, safe on wrecks.
  * outside any detected vehicle's box: dropped only if `min_in_vehicle` > 0. Optional and OFF by default: on
    the sample photos it also removed real damage from crushed cars the COCO model did not recognise as cars.
  * photos with no detected vehicle (close-ups of a panel or wheel) are never filtered by the second rule.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .backends.base import RawDet
from .postprocess import _G, _mask
from .schema import ModelInfo

VEHICLE_CLASSES = ("car", "truck", "bus")
_SIZES = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium"}


ON_CAR = 0.5  # a damage with at least this share inside a vehicle MASK counts as on the car (e.g. under a passenger)


@dataclass
class Regions:
    vehicle: np.ndarray   # bool (_G, _G), union of vehicle masks (slightly dilated)
    person: np.ndarray    # bool (_G, _G)
    n_vehicle: int = 0
    vehicle_box: np.ndarray = None  # union of vehicle bounding boxes; generous, covers crushed areas the mask misses

    def __post_init__(self):
        if self.vehicle_box is None:
            self.vehicle_box = self.vehicle


def gate(dets: List[RawDet], reg: Regions, min_in_vehicle: float = 0.0,
         max_in_person: float = 0.5) -> Tuple[List[RawDet], List[Tuple[RawDet, str]]]:
    """Returns (kept, [(removed, reason)]) with reason 'on_person' or 'outside_vehicle'.

    on_person       : >= max_in_person of the damage is inside a person mask and < ON_CAR is on the vehicle mask
    outside_vehicle : (only if min_in_vehicle > 0 and a vehicle was found) < min_in_vehicle of the damage is inside
                      any vehicle bounding box. Risky on wrecks: a crushed car may not be detected as a car at all
                      while another car in the scene is, so real damage then counts as "outside". Off by default.
    """
    kept, removed = [], []
    for d in dets:
        m = _mask(d)
        area = int(m.sum())
        if not area:
            kept.append(d)
            continue
        in_person = int((m & reg.person).sum()) / area
        on_car = int((m & reg.vehicle).sum()) / area
        if max_in_person and in_person >= max_in_person and on_car < ON_CAR:
            removed.append((d, "on_person"))
        elif min_in_vehicle and reg.n_vehicle and int((m & reg.vehicle_box).sum()) / area < min_in_vehicle:
            removed.append((d, "outside_vehicle"))
        else:
            kept.append(d)
    return kept, removed


class VehicleDetector:
    def __init__(self, device: str = "cpu", size: str = "small", conf: float = 0.35, dilate: int = 2):
        import rfdetr
        self.conf, self.dilate = conf, dilate
        self.model = getattr(rfdetr, _SIZES[size])(device=device)  # default = COCO-pretrained, MD5-checked, safe-loaded
        wname = str(getattr(self.model.model_config, "pretrain_weights", "") or "")
        wpath = Path.home() / ".roboflow" / "models" / Path(wname).name
        sha = hashlib.sha256(wpath.read_bytes()).hexdigest() if wpath.is_file() else ""
        self.info = ModelInfo(name=f"coco_rfdetr_seg_{size}", task="segment", sha256=sha,
                              classes=list(VEHICLE_CLASSES) + ["person"])

    def regions(self, images: Sequence[np.ndarray]) -> List[Regions]:
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
        dets = self.model.predict(rgb, threshold=self.conf)
        dets = dets if isinstance(dets, list) else [dets]
        return [self._regions(d) for d in dets]

    def _regions(self, d) -> Regions:
        veh = np.zeros((_G, _G), bool)
        box = np.zeros((_G, _G), bool)
        per = np.zeros((_G, _G), bool)
        n = 0
        names = d.data.get("class_name") if len(d) else []
        for i in range(len(d)):
            name = str(names[i])
            if name not in VEHICLE_CLASSES and name != "person":
                continue
            small = cv2.resize(d.mask[i].astype(np.uint8) * 255, (_G, _G), interpolation=cv2.INTER_AREA) > 127
            if name == "person":
                per |= small
                continue
            h0, w0 = d.mask[i].shape
            x1, y1, x2, y2 = d.xyxy[i]
            box[int(y1 / h0 * _G):int(np.ceil(y2 / h0 * _G)), int(x1 / w0 * _G):int(np.ceil(x2 / w0 * _G))] = True
            veh |= small
            n += 1
        if self.dilate and veh.any():
            veh = cv2.dilate(veh.astype(np.uint8), np.ones((2 * self.dilate + 1,) * 2, np.uint8)).astype(bool)
        return Regions(vehicle=veh, person=per, n_vehicle=n, vehicle_box=box)
