"""Turn screened Open Images candidates into hard negatives (images with NO annotations).

    python scripts/build_negatives.py plan  --exclude 367 111 81          # split + montages of unreviewed held-out images
    python scripts/build_negatives.py build --exclude 367 111 81 [...]    # datasets

  data/hitl_damage_v4 = v3 + negatives in train/valid (v3 test unchanged: same damaged photos, so v3 and v4 compare)
  data/neg_test       = held-out negatives only; measure the false-alarm rate with scripts/false_alarm_rate.py
Indices are the #numbers on the review montages (row order in candidates.csv / screen.json).
Split is per group (car / street) and seeded; images never appear in two splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import ANN, validate  # noqa: E402


def reviewed_indices(screen):
    """The images a human already looked at (same rule as screen_negatives.py: top-scoring 80 + seeded random 20)."""
    order = sorted(screen, key=lambda x: (-x["max_conf"], -x["regions"]))
    return {x["index"] for x in order[:80]} | {x["index"] for x in random.Random(1).sample(order[80:], 20)}


def plan(a):
    root = Path(a.root)
    screen = json.loads((root / "screen.json").read_text())
    by = {x["index"]: x for x in screen}
    ok = [x for x in screen if x["index"] not in set(a.exclude)]
    rng = random.Random(a.seed)
    split = {"test": [], "valid": [], "train": []}
    for g, nt, nv, nr in (("car", a.test_car, a.valid_car, a.train_car), ("street", a.test_street, a.valid_street, a.train_street)):
        pool = [x for x in ok if x["group"] == g]
        rng.shuffle(pool)
        pool = [x for x in pool if x["index"] not in set(a.drop)]   # after the shuffle: earlier reviews stay valid
        split["test"] += pool[:nt]
        split["valid"] += pool[nt:nt + nv]
        split["train"] += pool[nt + nv:nt + nv + nr]
    out = {s: [x["index"] for x in v] for s, v in split.items()}
    (root / "negatives_plan.json").write_text(json.dumps(out))
    print({s: len(v) for s, v in out.items()}, "| unused reserve:", len(ok) - sum(len(v) for v in out.values()))
    seen = reviewed_indices(screen)
    todo = [i for s in ("test", "valid") for i in out[s] if i not in seen]
    print(f"{len(todo)} held-out images (test+valid) not yet reviewed -> montages in {root / 'review'}")
    tiles = []
    for i in todo:
        im = cv2.imread(str(root / "images" / f"{by[i]['id']}.jpg"))
        t = cv2.resize(im, (300, max(1, int(im.shape[0] * 300 / im.shape[1]))))[:330]
        cv2.rectangle(t, (0, t.shape[0] - 24), (80, t.shape[0]), (0, 0, 0), -1)
        cv2.putText(t, f"#{i}", (5, t.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    for n in range(0, len(tiles), 30):
        chunk = tiles[n:n + 30]
        h = max(t.shape[0] for t in chunk)
        chunk = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in chunk]
        chunk += [np.zeros((h, 300, 3), np.uint8)] * (-len(chunk) % 6)
        p = root / "review" / f"heldout_{n // 30:02d}.jpg"
        cv2.imwrite(str(p), np.vstack([np.hstack(chunk[i:i + 6]) for i in range(0, len(chunk), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("wrote", p, "indices", [t for t in todo[n:n + 30]])


def link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build(a):
    root, base = Path(a.root), Path(a.base)
    out, nt = Path(a.out), Path(a.neg_test_out)
    for p in (out, nt):
        if p.exists():
            sys.exit(f"{p} exists; remove it (symlinks + json only) or choose another path")
    screen = {x["index"]: x for x in json.loads((root / "screen.json").read_text())}
    plan_ = json.loads((root / "negatives_plan.json").read_text())
    bad = (set(a.exclude) | set(a.drop)) & {i for v in plan_.values() for i in v}
    if bad:
        sys.exit(f"excluded indices {sorted(bad)} are still in the plan: re-run `plan` with the same --exclude/--drop")
    cand = {r["ImageID"]: r for r in csv.DictReader(open(root / "candidates.csv", newline=""))}
    categories = json.loads((base / "train" / ANN).read_text())["categories"]

    def negatives(split):
        return [dict(file_name=f"oi_{screen[i]['id']}.jpg", src=root / "images" / f"{screen[i]['id']}.jpg", id=screen[i]["id"],
                     group=f"oi_{screen[i]['id']}", kind=screen[i]["group"]) for i in plan_[split]]

    def write(dest: Path, split: str, images, anns, extra):
        d = dest / split
        d.mkdir(parents=True, exist_ok=True)
        nid = max([i["id"] for i in images], default=0)
        for n in extra:
            nid += 1
            link(n["src"], d / n["file_name"])
            im = cv2.imread(str(n["src"]))
            images.append(dict(id=nid, file_name=n["file_name"], width=im.shape[1], height=im.shape[0], group=n["group"]))
        (d / ANN).write_text(json.dumps(dict(images=images, annotations=anns, categories=categories)))

    for split in ("train", "valid", "test"):
        coco = json.loads((base / split / ANN).read_text())
        (out / split).mkdir(parents=True, exist_ok=True)
        for p in (base / split).glob("*"):
            if p.name != ANN:
                link(p, out / split / p.name)
        write(out, split, list(coco["images"]), list(coco["annotations"]), negatives(split) if split != "test" else [])
    write(nt, "test", [], [], negatives("test"))
    shutil.copy(base / "taxonomy.yaml", out / "taxonomy.yaml")

    used = [n for s in plan_ for n in negatives(s)]
    for dest, ids in ((out, [n for s in ("train", "valid") for n in negatives(s)]), (nt, negatives("test"))):
        with open(dest / "ATTRIBUTION.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "split_group", "Author", "License", "OriginalLandingURL", "Title"])
            for n in ids:
                c = cand[n["id"]]
                w.writerow([n["file_name"], n["kind"], c["Author"], c["License"], c["OriginalLandingURL"], c["Title"]])
    note = f"""

## Added for v4: hard negatives (damage-free) from Open Images V7 (validation split)
| | |
|---|---|
| Source | Open Images V7 validation images, via the public S3 mirror; metadata from storage.googleapis.com/openimages (CSV sha256 recorded in the build log) |
| Licence | Per-image `License` column = **CC BY 2.0** for every image used (attribution required: see ATTRIBUTION.csv with author + Flickr landing URL). Annotations are CC BY 4.0 (not used: these images have NO damage labels). Google gives no warranty on each image's licence status. |
| Selection | Human-verified `Car` boxes, non-depictions: {sum(n['kind']=='car' for n in used)} car-focused (one dominant car, >=30% of frame) and {sum(n['kind']=='street' for n in used)} street scenes (>=1 person, small cars). Excluded after review: indices {sorted(set(a.exclude) | set(a.drop))} (damaged vehicles, 3D renders, and images that are not vehicle exteriors: interiors, controls, animals, objects). |
| Screening | Reviewed by eye: the 80 images where v2 fired most strongly + 20 random + every held-out (valid/test) image. Train negatives outside that were NOT all reviewed; a few damaged cars may remain. |
| Split | train {len(plan_['train'])} / valid {len(plan_['valid'])} (in hitl_damage_v4) and test {len(plan_['test'])} (held out in data/neg_test); seed {a.seed}. |
| Domain note | Mostly car-show, vintage, studio and event photos plus street scenes; not typical claims photos. Negatives teach "people, bikes, karts, healthy cars are not damage"; they do not teach what a claims photo of an undamaged car looks like. |
"""
    (out / "PROVENANCE.md").write_text((base / "PROVENANCE.md").read_text() + note)
    nt.mkdir(parents=True, exist_ok=True)
    (nt / "PROVENANCE.md").write_text("# Held-out negatives (development only)\nNo damage labels: every image is a hard negative. Use scripts/false_alarm_rate.py.\n" + note)
    r = validate(out)
    print(f"\n== validate {out}\n{r.render()}\n-> {'OK' if r.ok else 'ERRORS'}")
    print(f"negative test set: {nt / 'test'} ({len(plan_['test'])} images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build"])
    ap.add_argument("--root", default="data/raw/openimages_negatives")
    ap.add_argument("--base", default="data/hitl_damage_v3")
    ap.add_argument("--out", default="data/hitl_damage_v4")
    ap.add_argument("--neg-test-out", default="data/neg_test")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="removed before the split shuffle")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="removed after the shuffle (keeps earlier review results valid)")
    ap.add_argument("--test-car", type=int, default=40)
    ap.add_argument("--test-street", type=int, default=20)
    ap.add_argument("--valid-car", type=int, default=20)
    ap.add_argument("--valid-street", type=int, default=10)
    ap.add_argument("--train-car", type=int, default=100)
    ap.add_argument("--train-street", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else build(a)
