"""python -m cardamage run --mode demo|prod --input <files|dirs> --out <dir>

demo: result.json + annotated/*.jpg + report.html
prod: result.json only (render.py is never imported)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect(inputs):
    files = []
    for i in inputs:
        p = Path(i)
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cardamage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse images")
    r.add_argument("--mode", choices=["demo", "prod"], default=None)
    r.add_argument("--input", nargs="+", required=True)
    r.add_argument("--out", default="outputs/run")
    r.add_argument("--registry", help="alternate models.yaml")
    r.add_argument("--damage-model")
    r.add_argument("--parts-model")
    r.add_argument("--conf", type=float)
    r.add_argument("--device")
    r.add_argument("--require-cleared", action="store_true", default=None)
    r.add_argument("--vehicle-filter", action="store_true", default=None,
                   help="drop damage not on a vehicle / on a person (COCO-pretrained RF-DETR-Seg; also CARDAMAGE_VEHICLE_FILTER=1)")
    r.add_argument("--stdout", action="store_true", help="print the JEV JSON instead of only writing it")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    s = Settings.from_env(mode=a.mode, damage_model=a.damage_model, parts_model=a.parts_model,
                          conf=a.conf, device=a.device, require_cleared=a.require_cleared,
                          vehicle_filter=a.vehicle_filter,
                          registry=Path(a.registry) if a.registry else None)
    files = _collect(a.input)
    if not files:
        print("no images found", file=sys.stderr)
        return 2

    from .pipeline import DamageAnalyzer  # heavy import (torch) only when needed
    result, imgs = DamageAnalyzer(s).analyze(files)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump_json(indent=1)
    (out / "result.json").write_text(payload)
    if s.mode == "demo":
        from .render import write_demo_outputs  # only imported in demo mode
        write_demo_outputs(result, imgs, out, show_parts=True)
    if a.stdout:
        print(payload)
    n = sum(len(i.damages) for i in result.images)
    print(f"[{s.mode}] {len(files)} image(s), {n} damage region(s) -> {out}"
          + (f"  (open {out / 'report.html'})" if s.mode == "demo" else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
