"""Supervisely-format export (img/ + ann/*.json) -> one COCO-seg json for prepare_coco_splits.py.

    python scripts/convert_supervisely.py --root "data/raw/hitl_car_damages/Car parts dataset/File1" \
        --out data/work/hitl_damage_all.json
Classes are mapped onto the schema taxonomy (CLASS_MAP). Interior holes are dropped (COCO polygons
can't hold them cheaply; counted and reported). Near-duplicate photos (dHash Hamming <= --dup-dist)
share a `group`, so prepare_coco_splits.py keeps them in one split (no near-duplicate leakage).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import DAMAGE_TYPES  # noqa: E402

CLASS_MAP = {"Scratch": "scratch", "Dent": "dent", "Cracked": "crack", "Broken part": "broken_part",
             "Missing part": "missing_part", "Paint chip": "paint_chip", "Flaking": "flaking",
             "Corrosion": "corrosion"}


def dhash(path: Path) -> int:
    g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def cluster(hashes, dist):
    parent = list(range(len(hashes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    arr = np.array([[(h >> k) & 1 for k in range(64)] for h in hashes], dtype=np.uint8)
    for i in range(len(hashes)):
        d = (arr[i + 1:] != arr[i]).sum(1)
        for j in np.nonzero(d <= dist)[0]:
            parent[find(i + 1 + j)] = find(i)
    return [find(i) for i in range(len(hashes))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dup-dist", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.root)
    unknown, holes, tiny = Counter(), 0, 0
    images, anns, paths = [], [], []
    for annf in sorted((root / "ann").glob("*.json")):
        d = json.loads(annf.read_text())
        imgf = root / "img" / annf.name[:-5]
        if not imgf.exists():
            print("missing image for", annf.name); continue
        w, h = d["size"]["width"], d["size"]["height"]
        with Image.open(imgf) as im:
            assert im.size == (w, h), f"{imgf.name}: ann size {(w, h)} != image {im.size}"
        iid = len(images) + 1
        images.append(dict(id=iid, file_name=imgf.name, width=w, height=h))
        paths.append(imgf)
        for o in d["objects"]:
            name = CLASS_MAP.get(o["classTitle"])
            if name is None or o["geometryType"] != "polygon":
                unknown[o["classTitle"]] += 1; continue
            pts = np.asarray(o["points"]["exterior"], dtype=float)
            holes += bool(o["points"].get("interior"))
            if len(pts) < 3:
                tiny += 1; continue
            x, y = pts[:, 0], pts[:, 1]
            area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
            bw, bh = x.max() - x.min(), y.max() - y.min()
            if area < 4 or bw < 2 or bh < 2:
                tiny += 1; continue
            anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=name,
                             iscrowd=0, area=float(area), bbox=[float(x.min()), float(y.min()), float(bw), float(bh)],
                             segmentation=[[round(float(v), 1) for v in pts.flatten()]]))
    groups = cluster([dhash(p) for p in paths], a.dup_dist)
    for im, g in zip(images, groups):
        im["group"] = f"g{g}"
    used = sorted({x["category_id"] for x in anns})
    ids = {n: i + 1 for i, n in enumerate(used)}  # contiguous 1..K (RF-DETR maps class slots from these)
    for x in anns:
        x["category_id"] = ids[x["category_id"]]
    cats = [dict(id=ids[n], name=n) for n in used]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(images=images, annotations=anns, categories=cats)))
    ng = len(set(groups))
    print(f"{len(images)} images in {ng} groups ({len(images) - ng} near-duplicates merged), {len(anns)} polygons, "
          f"holes dropped on {holes}, tiny skipped {tiny}, unmapped {dict(unknown)}")
    print("per class:", Counter(c['name'] for x in anns for c in cats if c['id'] == x['category_id']).most_common())


if __name__ == "__main__":
    main()
