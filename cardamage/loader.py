"""Load a model converted by scripts/safe_convert.py (safetensors + meta.json).

The architecture is rebuilt by Ultralytics from the YAML recorded in meta.json
and weights are loaded with ``strict=True``; no pickle is ever touched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import load_file
from ultralytics import YOLO
from ultralytics.utils import callbacks
from ultralytics.nn.tasks import DetectionModel, SegmentationModel


def load_yolo(model_dir: Path) -> Tuple[YOLO, dict]:
    meta = json.loads((model_dir / "meta.json").read_text())
    cfg = dict(meta["yaml"])
    names = {int(k): v for k, v in meta["names"].items()}
    cls = SegmentationModel if meta["task"] == "segment" else DetectionModel
    net = cls(cfg, ch=cfg.get("channels", cfg.get("ch", 3)), nc=len(names), verbose=False)
    net.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    net.names = names
    net.eval()
    # YOLO.__init__ insists on loading a file (and fetches a default checkpoint
    # when given none), so wrap our already-built network by hand.
    y = YOLO.__new__(YOLO)
    torch.nn.Module.__init__(y)
    y.callbacks = callbacks.get_default_callbacks()
    y.predictor = y.trainer = y.cfg = y.ckpt = y.ckpt_path = y.metrics = y.model_name = None
    y.model, y.task, y.overrides = net, meta["task"], {"task": meta["task"]}
    return y, meta
