"""Remap a COCO-seg dataset to a merged taxonomy and cluster tiny marks into regions.

    python scripts/remap_classes.py --src data/hitl_damage_v1 --out data/hitl_damage_v2 --spec configs/taxonomy_v2.yaml

Every split (train/valid/test) is remapped the same way, so the evaluation target changes with the taxonomy:
v2 numbers are NOT comparable to v1 numbers (coarser classes, region-level instances). Report both.
Images are symlinked, not copied. Photo groups are preserved (no new leakage).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import ANN, DAMAGE_TYPES, SPLITS, load_coco, validate  # noqa: E402

W = 512  # working resolution (long side) for the region clustering


def regions(polys, w, h, frac, min_area_pct):
    """Union the polygons of ONE class in one image and close gaps <= 2*radius -> list of polygons (original px)."""
    s = W / max(w, h)
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    m = np.zeros((sh, sw), np.uint8)
    for seg in polys:
        for poly in seg:
            pts = (np.asarray(poly, float).reshape(-1, 2) * s).round().astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    if frac:
        r = max(1, round(frac * W))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.morphologyEx(cv2.copyMakeBorder(m, r, r, r, r, cv2.BORDER_CONSTANT), cv2.MORPH_CLOSE, k)[r:-r, r:-r]
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) * 100 / (sw * sh) < min_area_pct or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append((c / s).round(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.spec).read_text())
    bad = [c for c in spec["classes"] if c not in DAMAGE_TYPES]
    if bad:
        sys.exit(f"classes not in the schema taxonomy: {bad}")
    src_to_new = {s: n for n, srcs in spec["classes"].items() for s in srcs}
    ids = {n: i + 1 for i, n in enumerate(spec["classes"])}
    src, out = Path(a.src), Path(a.out)
    before, after = Counter(), Counter()
    for split in SPLITS:
        coco = load_coco(src, split)
        if coco is None:
            continue
        old = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {i["id"]: i for i in coco["images"]}
        grouped = {}
        for ann in coco["annotations"]:
            new = src_to_new.get(old[ann["category_id"]])
            if new is None:
                continue
            before[new] += 1
            grouped.setdefault((ann["image_id"], new), []).append(ann["segmentation"])
        anns = []
        for (iid, new), segs in sorted(grouped.items()):
            im = imgs[iid]
            frac = spec.get("region_merge", {}).get(new, 0)
            polys = regions(segs, im["width"], im["height"], frac, spec.get("min_region_area_pct", 0.0)) if frac \
                else [np.asarray(p, float).reshape(-1, 2) for seg in segs for p in seg]
            for p in polys:
                x, y = p[:, 0], p[:, 1]
                area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
                if area < 4:
                    continue
                anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=ids[new], iscrowd=0, area=float(area),
                                 bbox=[float(x.min()), float(y.min()), float(x.max() - x.min()), float(y.max() - y.min())],
                                 segmentation=[[round(float(v), 1) for v in p.flatten()]]))
                after[new] += 1
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for im in coco["images"]:
            dst = d / im["file_name"]
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to((src / split / im["file_name"]).resolve())
        (d / ANN).write_text(json.dumps(dict(images=coco["images"], annotations=anns,
                                             categories=[dict(id=i, name=n) for n, i in ids.items()])))
    prov = src / "PROVENANCE.md"
    (out / "PROVENANCE.md").write_text((prov.read_text() if prov.exists() else "") +
        f"\n\n## Derived dataset: {spec['name']}\nRemapped from `{src}` with `{a.spec}` (merged classes, tiny marks clustered "
        f"into regions). Same images, same groups, same splits. Class counts (instances): before {dict(before)} -> after {dict(after)}.\n")
    shutil.copy2(a.spec, out / "taxonomy.yaml")
    print(f"{'class':14s} {'source instances':>17s} {'-> regions/instances':>21s}")
    for n in ids:
        print(f"{n:14s} {before[n]:17d} {after[n]:21d}")
    rep = validate(out)
    print(rep.render())
    print("valid:", rep.ok)


if __name__ == "__main__":
    main()
