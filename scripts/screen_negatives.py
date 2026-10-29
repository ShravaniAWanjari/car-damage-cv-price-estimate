"""Screen negative candidates: run a damage model over them and write review montages.

A negative must be damage-free; a model firing is NOT proof of damage (these are the hard negatives we want) and a
silent model is not proof of none, so the montages are for a human to look at. Review order is highest-scoring first.

    python scripts/screen_negatives.py --root data/raw/openimages_negatives --model rfdetr_seg_s_v2
Writes root/screen.json (per image: index, id, group, regions, max_conf) and root/review/montage_XX.jpg (labelled #index).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.config import Settings  # noqa: E402
from cardamage.pipeline import DamageAnalyzer  # noqa: E402
from cardamage.render import annotate  # noqa: E402


def montage(items, path, cols=5, w=360):
    tiles = []
    for k, im, res in items:
        a = annotate(im, res, show_parts=False)
        t = cv2.resize(a, (w, max(1, int(a.shape[0] * w / a.shape[1]))))
        t = t[:int(w * 1.1)]
        cv2.rectangle(t, (0, t.shape[0] - 26), (110, t.shape[0]), (0, 0, 0), -1)
        cv2.putText(t, f"#{k}", (6, t.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros((h, w, 3), np.uint8)] * (-len(tiles) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/openimages_negatives")
    ap.add_argument("--model", default="rfdetr_seg_s_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--review-top", type=int, default=80, help="highest-scoring images to review one by one")
    ap.add_argument("--review-random", type=int, default=20, help="plus a random sample of the rest, to estimate how many slip through")
    a = ap.parse_args()
    root = Path(a.root)
    rows = list(csv.DictReader(open(root / "candidates.csv", newline="")))
    an = DamageAnalyzer(Settings.from_env(mode="prod", damage_model=a.model, device=a.device))
    out, seen = [], {}
    for s in range(0, len(rows), 8):
        chunk = rows[s:s + 8]
        imgs = [cv2.imread(str(root / "images" / f"{r['ImageID']}.jpg")) for r in chunk]
        res, _ = an.analyze(imgs, [r["ImageID"] for r in chunk])
        for j, (r, im, ir) in enumerate(zip(chunk, imgs, res.images)):
            k = s + j
            seen[k] = (im, ir)
            out.append(dict(index=k, id=r["ImageID"], group=r["group"], regions=len(ir.damages),
                            max_conf=max([d.confidence for d in ir.damages], default=0.0),
                            types=dict(Counter(d.type for d in ir.damages))))
    (root / "screen.json").write_text(json.dumps(out, indent=1))
    for g in ("car", "street"):
        o = [x for x in out if x["group"] == g]
        fired = sum(x["regions"] > 0 for x in o)
        print(f"{g:6s}: {len(o)} images; >=1 region at conf 0.30: {fired} ({100 * fired / len(o):.0f}%); "
              f"max_conf>=0.5: {sum(x['max_conf'] >= 0.5 for x in o)}; by class {dict(sum((Counter(x['types']) for x in o), Counter()))}")
    order = sorted(out, key=lambda x: (-x["max_conf"], -x["regions"]))
    top = order[:a.review_top]
    rest = random.Random(1).sample(order[a.review_top:], a.review_random)
    rev = root / "review"
    rev.mkdir(exist_ok=True)
    for n, chunk in enumerate([top[i:i + 20] for i in range(0, len(top), 20)] + [rest]):
        path = rev / f"montage_{n:02d}.jpg"
        montage([(x["index"], *seen[x["index"]]) for x in chunk], path)
        print("wrote", path, "indices", [x["index"] for x in chunk])


if __name__ == "__main__":
    main()
