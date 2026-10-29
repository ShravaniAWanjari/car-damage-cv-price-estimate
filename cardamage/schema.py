"""JEV-facing output contract.

`AnalysisResult` is the only thing prod mode emits. Demo mode emits the same
object plus annotated images, so downstream (JEV) code never sees a difference.
Bump SCHEMA_VERSION on any breaking change.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.4.0"  # 0.2.0 added missing_part/paint_chip/flaking/corrosion; 0.3.0 added paint_damage; 0.4.0 vehicle_model (all additive)

# Normalised damage taxonomy handed to JEV. Each model's own labels are mapped
# onto this in models.yaml so JEV is insulated from model swaps.
DamageType = Literal[
    "dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat",
    "broken_part", "missing_part", "paint_chip", "flaking", "corrosion", "paint_damage", "other",
]


class Damage(BaseModel):
    id: str
    type: DamageType
    raw_label: str = Field(description="Label exactly as the model emitted it")
    confidence: float
    # All geometry is normalised to [0, 1] by image width/height.
    bbox: Tuple[float, float, float, float] = Field(description="x1, y1, x2, y2")
    polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None, description="Outline of the damage mask (None for box-only models)"
    )
    area_ratio_image: float = Field(description="Damage area / image area")
    panel: Optional[str] = Field(
        default=None, description="Car part with the largest overlap, if a parts model ran"
    )
    panel_overlap: Optional[float] = Field(
        default=None, description="Fraction of the damage mask lying inside `panel`"
    )


class Part(BaseModel):
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]


class ImageResult(BaseModel):
    image_id: str
    width: int
    height: int
    damages: List[Damage]
    parts: List[Part] = Field(default_factory=list)
    inference_ms: float


class ModelInfo(BaseModel):
    name: str
    task: Literal["detect", "segment"]
    sha256: str
    classes: List[str]


class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    mode: Literal["demo", "prod"]
    damage_model: ModelInfo
    parts_model: Optional[ModelInfo] = None
    vehicle_model: Optional[ModelInfo] = Field(
        default=None, description="Set when the car-region filter ran: damages not on a vehicle were dropped")
    images: List[ImageResult]
