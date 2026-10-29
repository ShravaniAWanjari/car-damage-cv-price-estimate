from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]


class ModelEntry(BaseModel):
    dir: str
    role: Literal["damage", "parts"]
    backend: Literal["ultralytics", "rfdetr"] = "ultralytics"
    conf: float = 0.25
    # Per damage-type minimum confidence (keys are taxonomy types, e.g. tire_flat).
    # scripts/eval_model.py prints a calibrated block to paste here.
    class_conf: Dict[str, float] = {}
    source: Optional[str] = None
    taxonomy: Dict[str, str] = {}
    license: Dict[str, object] = {}


class Settings(BaseModel):
    mode: Literal["demo", "prod"] = "demo"
    registry: Path = ROOT / "models.yaml"
    damage_model: Optional[str] = None
    parts_model: Optional[str] = None
    conf: Optional[float] = None       # overrides the per-model default
    iou: float = 0.6
    max_det: int = 50
    min_area_ratio: float = 0.0005     # drop specks smaller than this fraction of the image
    parts_min_overlap: float = 0.3
    dedupe_iou: float = 0.5            # merge same-label detections overlapping this much (0 = off)
    dedupe_containment: float = 0.85   # ... or sitting this far inside a higher-confidence one (0 = off)
    # Optional car-region filter (cardamage/vehicle.py): drop damage that is not on a vehicle / is on a person
    vehicle_filter: bool = False
    vehicle_conf: float = 0.35         # detector threshold for the COCO vehicle/person model
    vehicle_min_overlap: float = 0.0   # >0: also drop damage with less than this fraction inside a detected vehicle's box (risky on wrecks)
    person_max_overlap: float = 0.5    # drop a damage with at least this fraction inside a person mask (0 = off)
    batch_size: int = 8
    device: str = "auto"             # auto | cpu | mps | cuda:0
    require_cleared: bool = False      # refuse weights whose license.production_ok is not true

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env = {"mode": os.getenv("CARDAMAGE_MODE"), "device": os.getenv("CARDAMAGE_DEVICE"),
               "damage_model": os.getenv("CARDAMAGE_DAMAGE_MODEL"),
               "require_cleared": os.getenv("CARDAMAGE_REQUIRE_CLEARED") == "1" or None,
               "vehicle_filter": os.getenv("CARDAMAGE_VEHICLE_FILTER") == "1" or None}
        data = {k: v for k, v in env.items() if v}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def load_registry(path: Path) -> tuple[dict, Dict[str, ModelEntry]]:
    raw = yaml.safe_load(path.read_text())
    return raw.get("defaults", {}), {k: ModelEntry(**v) for k, v in raw["models"].items()}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
