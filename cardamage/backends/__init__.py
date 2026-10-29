from __future__ import annotations

from pathlib import Path

from ..config import ROOT, ModelEntry
from .base import Backend, RawDet

__all__ = ["Backend", "RawDet", "build_backend"]


def build_backend(name: str, entry: ModelEntry, device: str = "cpu") -> Backend:
    """Heavy imports happen here so a prod image only needs the backend it serves."""
    d = Path(entry.dir)
    d = d if d.is_absolute() else ROOT / d
    if entry.backend == "ultralytics":
        from .ultralytics_backend import UltralyticsBackend
        return UltralyticsBackend(name, d)
    if entry.backend == "rfdetr":
        from .rfdetr_backend import RFDETRBackend
        return RFDETRBackend(name, d, device=device)
    raise ValueError(f"unknown backend {entry.backend!r}")
