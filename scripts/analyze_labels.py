"""Label audit for a COCO-seg split: how big/thin each class is at the model's input size, how classes
overlap each other in the annotations, and same-class duplicates. Use it to decide a taxonomy BEFORE
training - and run it on train/valid only; do not let test statistics steer design decisions.

    python scripts/analyze_labels.py --dataset-dir data/hitl_damage_v1 --splits train valid --input-size 384
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import load_coco  # noqa: E402

G = 192


def raster(seg, w, h):
    m = np.zeros((G, G), np.uint8)
    for poly in seg:
        pts = (np.asarray(poly, float).reshape(-1, 2) / [w, h] * (G - 1)).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--input-size", type=int, default=384, help="model input side in px (small=384, medium=432)")
    ap.add_argument("--out", default=None, help="write JSON here")
    a = ap.parse_args()

    stats = defaultdict(lambda: defaultdict(list))
    overlap = defaultdict(Counter)   # overlap[a][b] = #instances of a mostly covered by some instance of b
    dup = Counter()
    n_inst = Counter()
    per_img = defaultdict(list)
    for split in a.splits:
        coco = load_coco(Path(a.dataset_dir), split)
        names = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {i["id"]: i for i in coco["images"]}
        by_img = defaultdict(list)
        for ann in coco["annotations"]:
            by_img[ann["image_id"]].append(ann)
        for iid, anns in by_img.items():
            im = imgs[iid]
            w, h = im["width"], im["height"]
            masks = [(names[x["category_id"]], raster(x["segmentation"], w, h)) for x in anns]
            for x, (n, m) in zip(anns, masks):
                bw, bh = x["bbox"][2], x["bbox"][3]
                scale = a.input_size / max(w, h)
                stats[n]["area_pct"].append(100 * m.sum() / (G * G))
                stats[n]["min_side_px"].append(min(bw, bh) * scale)
                stats[n]["max_side_px"].append(max(bw, bh) * scale)
                n_inst[n] += 1
                per_img[n].append(iid)
            for i, (ni, mi) in enumerate(masks):
                ai = mi.sum()
                for j, (nj, mj) in enumerate(masks):
                    if i == j or ai == 0:
                        continue
                    inter = (mi & mj).sum()
                    if ni == nj:
                        if j > i and inter / max(1, (mi | mj).sum()) >= 0.7:
                            dup[ni] += 1
                    elif inter / ai >= 0.5:
                        overlap[ni][nj] += 1
                        break  # count each instance once per other-class partner: first hit only
    names_sorted = sorted(n_inst, key=lambda n: -n_inst[n])
    print(f"splits {a.splits}; model input {a.input_size}px\n")
    print(f"{'class':14s} {'n':>5s} {'imgs':>5s} {'median area%':>13s} {'<0.2% area':>11s} {'median min-side px':>19s} {'min-side<6px':>13s} {'dup(IoU>.7)':>12s}")
    out = {}
    for n in names_sorted:
        s = stats[n]
        area, mn = np.array(s["area_pct"]), np.array(s["min_side_px"])
        row = dict(n=n_inst[n], images=len(set(per_img[n])), median_area_pct=float(np.median(area)),
                   frac_tiny_area=float((area < 0.2).mean()), median_min_side_px=float(np.median(mn)),
                   frac_min_side_lt6=float((mn < 6).mean()), dup_pairs=int(dup[n]))
        out[n] = row
        print(f"{n:14s} {row['n']:5d} {row['images']:5d} {row['median_area_pct']:13.2f} {row['frac_tiny_area']:11.0%} "
              f"{row['median_min_side_px']:19.1f} {row['frac_min_side_lt6']:13.0%} {row['dup_pairs']:12d}")
    print("\nAnnotation overlap: % of row-class instances >=50% covered by an instance of the column class (same image)")
    print(f"{'':14s}" + "".join(f"{n[:9]:>10s}" for n in names_sorted))
    for ni in names_sorted:
        print(f"{ni:14s}" + "".join(f"{(100 * overlap[ni][nj] / n_inst[ni]) if ni != nj else 0:10.0f}" for nj in names_sorted))
    out["_overlap_pct"] = {ni: {nj: 100 * overlap[ni][nj] / n_inst[ni] for nj in names_sorted if nj != ni} for ni in names_sorted}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
