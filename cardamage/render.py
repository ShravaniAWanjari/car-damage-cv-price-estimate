"""Demo-mode only: draw an AnalysisResult onto its source image."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .schema import AnalysisResult, ImageResult

# BGR, one colour per damage type
COLORS: Dict[str, Tuple[int, int, int]] = {
    "dent": (0, 140, 255), "scratch": (0, 220, 255), "crack": (255, 80, 80),
    "glass_shatter": (255, 200, 0), "lamp_broken": (60, 60, 255), "tire_flat": (200, 60, 200),
    "broken_part": (60, 60, 255), "missing_part": (120, 0, 200), "paint_chip": (120, 255, 120),
    "flaking": (80, 200, 160), "paint_damage": (0, 220, 255), "corrosion": (30, 90, 170), "other": (180, 180, 180),
}


def annotate(img: np.ndarray, res: ImageResult, show_parts: bool = True) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1000.0
    th, fs = max(1, round(2 * scale)), max(0.45, 0.6 * scale)
    overlay = out.copy()
    for d in res.damages:  # filled masks first, then text on top
        col = COLORS.get(d.type, COLORS["other"])
        if d.polygon:
            pts = (np.asarray(d.polygon) * [w, h]).astype(np.int32)
            cv2.fillPoly(overlay, [pts], col)
    out = cv2.addWeighted(overlay, 0.40, out, 0.60, 0)
    if show_parts:
        for p in res.parts:
            x1, y1, x2, y2 = (np.asarray(p.bbox) * [w, h, w, h]).astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
    for d in res.damages:
        col = COLORS.get(d.type, COLORS["other"])
        if d.polygon:
            pts = (np.asarray(d.polygon) * [w, h]).astype(np.int32)
            cv2.polylines(out, [pts], True, col, th, cv2.LINE_AA)
        x1, y1, x2, y2 = (np.asarray(d.bbox) * [w, h, w, h]).astype(int)
        if not d.polygon:
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        label = f"{d.type} {d.confidence:.2f}" + (f" @{d.panel}" if d.panel else "")
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(y1, tht + base + 2)
        cv2.rectangle(out, (x1, ty - tht - base - 2), (x1 + tw + 6, ty + 2), col, -1)
        cv2.putText(out, label, (x1 + 3, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    banner = f"{len(res.damages)} damage region(s)" if res.damages else "no damage detected"
    bfs = 0.7 * scale
    (bw, bh), bb = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, bfs, 1)
    cv2.rectangle(out, (0, 0), (bw + 16, bh + bb + 12), (0, 0, 0), -1)
    cv2.putText(out, banner, (8, bh + 6), cv2.FONT_HERSHEY_SIMPLEX, bfs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def annotated_names(ids: List[str]) -> List[str]:
    """File names for the annotated images: URL/file-system safe and unique (image ids come from user file names)."""
    used, out = set(), []
    for i in ids:
        base = re.sub(r"[^\w.-]+", "_", i).strip("._") or "image"
        name, k = base, 2
        while name in used:
            name, k = f"{base}-{k}", k + 1
        used.add(name)
        out.append(name + ".jpg")
    return out


def write_demo_outputs(result: AnalysisResult, images: List[np.ndarray], out_dir: Path,
                       show_parts: bool = True) -> List[Path]:
    """annotated/*.jpg + report.html next to result.json."""
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)
    paths, cards = [], []
    names = annotated_names([r.image_id for r in result.images])
    for img, res, name in zip(images, result.images, names):
        p = ann_dir / name
        cv2.imwrite(str(p), annotate(img, res, show_parts), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
        rows = "".join(
            f"<tr><td>{html.escape(d.type)}</td><td>{d.confidence:.2f}</td>"
            f"<td>{d.area_ratio_image * 100:.2f}%</td><td>{html.escape(d.panel or '-')}</td></tr>"
            for d in res.damages) or "<tr><td colspan=4>no damage detected</td></tr>"
        cards.append(
            f"<section><h3>{html.escape(res.image_id)} <small>{res.inference_ms:.0f} ms</small></h3>"
            f"<img src='annotated/{html.escape(name)}'>"
            f"<table><tr><th>type</th><th>conf</th><th>area</th><th>panel</th></tr>{rows}</table>"
            f"<details><summary>JEV input (JSON)</summary><pre>"
            f"{html.escape(json.dumps(res.model_dump(mode='json'), indent=1))}</pre></details></section>")
    (out_dir / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Car damage demo</title><style>"
        "body{font:14px system-ui;margin:24px;background:#111;color:#eee}"
        "section{display:inline-block;vertical-align:top;width:560px;margin:0 16px 24px 0}"
        "img{max-width:100%;border-radius:6px}table{border-collapse:collapse;margin-top:6px}"
        "td,th{border:1px solid #444;padding:2px 8px;text-align:left}pre{max-height:260px;overflow:auto;"
        "background:#1b1b1b;padding:8px}small{color:#999}</style>"
        f"<h2>Car damage demo &middot; model <code>{html.escape(result.damage_model.name)}</code></h2>"
        + "".join(cards))
    return paths
