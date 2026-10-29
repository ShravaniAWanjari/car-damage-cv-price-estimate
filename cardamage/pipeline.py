"""Image(s) in -> AnalysisResult (the JEV input) out.

Rendering lives in render.py and is only ever imported by demo-mode callers;
this module has no drawing code, so prod mode cannot emit annotated images.
Model specifics live in cardamage/backends; this module only sees RawDets.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, get_args

import cv2
import numpy as np

from .backends import Backend, RawDet, build_backend
from .config import ModelEntry, Settings, load_registry
from .geometry import box_to_poly, overlap_fraction, poly_area, simplify
from .postprocess import dedupe
from .schema import AnalysisResult, Damage, DamageType, ImageResult, Part

log = logging.getLogger("cardamage")
DAMAGE_TYPES = set(get_args(DamageType))
ImageLike = Union[str, Path, np.ndarray]


def unique_ids(ids: Sequence[str]) -> List[str]:
    """Image ids must be unique within one result (damage ids and demo files derive from them):
    two uploads both called photo.jpg become `photo` and `photo-2`."""
    used: set = set()
    out: List[str] = []
    for i in ids:
        cand, k = i, 2
        while cand in used:
            cand, k = f"{i}-{k}", k + 1
        used.add(cand)
        out.append(cand)
    return out


def pick_device(requested: str) -> Tuple[str, bool]:
    """Returns (device, use_fp16)."""
    if requested != "auto":
        return requested, requested.startswith("cuda")
    import torch
    if torch.cuda.is_available():
        return "cuda:0", True
    return "cpu", False  # MPS segmentation is not reliably supported by Ultralytics


class DamageAnalyzer:
    def __init__(self, settings: Settings):
        self.s = settings
        defaults, registry = load_registry(settings.registry)
        dname = settings.damage_model or defaults["damage_model"]
        pname = settings.parts_model or defaults.get("parts_model")
        self.device, self.half = pick_device(settings.device)
        self.damage, self.damage_entry = self._load(dname, registry, "damage")
        self.parts, self.parts_entry = self._load(pname, registry, "parts") if pname else (None, None)
        self.vehicle = None
        if settings.vehicle_filter:
            from .vehicle import VehicleDetector  # loads the COCO-pretrained model only when asked
            self.vehicle = VehicleDetector(self.device, conf=settings.vehicle_conf)

    def _load(self, name: str, registry: dict, role: str) -> Tuple[Backend, ModelEntry]:
        if name not in registry:
            raise KeyError(f"unknown model {name!r}; known: {sorted(registry)}")
        entry = registry[name]
        if entry.role != role:
            raise ValueError(f"{name} has role {entry.role!r}, expected {role!r}")
        if not entry.license.get("production_ok"):
            msg = f"model {name!r} is not cleared for production use: {entry.license}"
            if self.s.require_cleared:
                raise PermissionError(msg)
            log.warning(msg)
        return build_backend(name, entry, self.device), entry

    # ------------------------------------------------------------------ public
    def analyze(self, images: Sequence[ImageLike], ids: Optional[Sequence[str]] = None
                ) -> Tuple[AnalysisResult, List[np.ndarray]]:
        """Returns the result plus the decoded BGR images (so demo rendering
        does not have to read them twice)."""
        arrs = [self._read(i) for i in images]
        ids = unique_ids(ids if ids else [
            Path(i).stem if isinstance(i, (str, Path)) else f"image_{n}" for n, i in enumerate(images)])
        results: List[ImageResult] = []
        for start in range(0, len(arrs), self.s.batch_size):
            chunk = arrs[start:start + self.s.batch_size]
            t0 = time.perf_counter()
            dmg_raw = [kept for kept, _ in self.gate_raw(chunk, self.predict_raw(chunk))]
            parts_raw = (self.parts.predict(chunk, self.parts_entry.conf, self.s.iou, self.s.max_det,
                                            self.device, self.half) if self.parts else None)
            ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            for k, img in enumerate(chunk):
                h, w = img.shape[:2]
                parts = [Part(label=p.label, confidence=round(p.conf, 4),
                              bbox=tuple(round(v, 4) for v in p.box)) for p in parts_raw[k]] if parts_raw else []
                dmgs = self._damages(dmg_raw[k], ids[start + k], parts_raw[k] if parts_raw else [])
                results.append(ImageResult(image_id=ids[start + k], width=w, height=h,
                                           damages=dmgs, parts=parts, inference_ms=round(ms, 1)))
        out = AnalysisResult(mode=self.s.mode, damage_model=self.damage.info,
                             parts_model=self.parts.info if self.parts else None,
                             vehicle_model=self.vehicle.info if self.vehicle else None, images=results)
        return out, arrs

    def predict_raw(self, imgs: Sequence[np.ndarray], conf: Optional[float] = None) -> List[List[RawDet]]:
        """Backend output for the damage model at `conf` (default: settings/registry).
        Exposed so evaluation can sweep thresholds without the filtering below."""
        c = conf if conf is not None else (self.s.conf if self.s.conf is not None else self.damage_entry.conf)
        return self.damage.predict(imgs, c, self.s.iou, self.s.max_det, self.device, self.half)

    def gate_raw(self, imgs: Sequence[np.ndarray], raws: List[List[RawDet]]):
        """Car-region filter: [(kept, [(removed, reason)])] per image. A no-op (nothing removed) when disabled."""
        if not self.vehicle:
            return [(r, []) for r in raws]
        from .vehicle import gate
        return [gate(r, reg, self.s.vehicle_min_overlap, self.s.person_max_overlap)
                for r, reg in zip(raws, self.vehicle.regions(imgs))]

    def dedupe(self, dets: List[RawDet]) -> List[RawDet]:
        return dedupe(dets, self.s.dedupe_iou, self.s.dedupe_containment)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read(src: ImageLike) -> np.ndarray:
        if isinstance(src, np.ndarray):
            return src
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)  # honours EXIF orientation
        if img is None:
            raise ValueError(f"could not decode image: {src}")
        return img

    def _damages(self, dets: List[RawDet], image_id: str, parts: List[RawDet]) -> List[Damage]:
        tax, class_conf = self.damage_entry.taxonomy, self.damage_entry.class_conf
        part_polys = [(p.label, p.poly or box_to_poly(p.box)) for p in parts]
        out: List[Damage] = []
        for d in sorted(self.dedupe(dets), key=lambda d: -d.conf):
            dtype = tax.get(d.label, d.label if not tax else "other")
            if dtype not in DAMAGE_TYPES:
                dtype = "other"
            if d.conf < class_conf.get(dtype, 0.0):
                continue
            poly = simplify(d.poly) if d.poly and len(d.poly) >= 3 else None
            x1, y1, x2, y2 = d.box
            area = poly_area(poly) if poly else (x2 - x1) * (y2 - y1)
            if area < self.s.min_area_ratio:
                continue
            box = (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))
            panel, ov = self._panel(poly or box_to_poly(box), part_polys)
            out.append(Damage(id=f"{image_id}:{len(out)}", type=dtype, raw_label=d.label,
                              confidence=round(d.conf, 4), bbox=box, polygon=poly,
                              area_ratio_image=round(area, 5), panel=panel, panel_overlap=ov))
        return out

    def _panel(self, poly, part_polys) -> Tuple[Optional[str], Optional[float]]:
        best, best_ov = None, 0.0
        for label, pp in part_polys:
            ov = overlap_fraction(poly, pp)
            if ov > best_ov:
                best, best_ov = label, ov
        if best_ov < self.s.parts_min_overlap:
            return None, None
        return best, round(best_ov, 3)
