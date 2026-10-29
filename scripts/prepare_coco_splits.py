"""Turn ONE annotated COCO-seg export (CVAT / Label Studio / Roboflow ...) into the
train/valid/test layout, splitting by GROUP so photos of the same claim never straddle splits.

    python scripts/prepare_coco_splits.py --coco all.json --images photos/ --out data/damage_v1 \
        --group-key claim_id            # field on each image entry
    # or derive the group from the file name:   --group-regex '^(CLM\\d+)_'
    # damage-free photos: images with zero annotations are kept as negatives.

Greedy, class-aware: groups are placed (largest first) in the split furthest below its
target instance share. Writes a PROVENANCE.md stub you must complete.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import ANN, validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True, help="folder containing the files named in the COCO json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--group-key", default=None)
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    a = ap.parse_args()

    coco = json.loads(Path(a.coco).read_text())
    imgs = {i["id"]: i for i in coco["images"]}
    anns = defaultdict(list)
    for x in coco["annotations"]:
        anns[x["image_id"]].append(x)

    def group(i):
        if a.group_key and i.get(a.group_key) is not None:
            return str(i[a.group_key])
        if a.group_regex:
            m = re.search(a.group_regex, Path(i["file_name"]).name)
            if m:
                return m.group(1)
        return f"__single_{i['id']}"  # no group info: image is its own group (leak risk; warned below)

    groups = defaultdict(list)
    for i in imgs.values():
        i["group"] = group(i)
        groups[i["group"]].append(i["id"])
    singles = sum(1 for g in groups if g.startswith("__single_"))
    if singles:
        print(f"WARNING: {singles} image(s) had no group; same-claim photos among them can leak across splits",
              file=sys.stderr)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    total = Counter(cat_names[x["category_id"]] for x in coco["annotations"])
    target = {s: {c: n * r for c, n in total.items()} for s, r in zip(("train", "valid", "test"), a.ratios)}
    img_target = {s: len(imgs) * r for s, r in zip(("train", "valid", "test"), a.ratios)}
    got = {s: Counter() for s in target}
    got_imgs = Counter()

    def gcount(g):
        return Counter(cat_names[x["category_id"]] for iid in groups[g] for x in anns[iid])

    order = list(groups)
    random.Random(a.seed).shuffle(order)
    order.sort(key=lambda g: -sum(gcount(g).values()))
    assign = {}
    for g in order:
        gc, n_img = gcount(g), len(groups[g])
        def deficit(s):
            # relative shortfall (1 = empty, <=0 = full), so image counts and every class count
            # pull with comparable strength regardless of absolute size
            tot = sum(gc.values())
            cls = sum(k / tot * (target[s][c] - got[s][c]) / max(target[s][c], 1e-9) for c, k in gc.items()) if tot else 0.0
            img = (img_target[s] - got_imgs[s]) / max(img_target[s], 1e-9)
            return cls + img
        s = max(target, key=deficit)
        assign[g] = s
        got[s].update(gc)
        got_imgs[s] += n_img

    out = Path(a.out)
    for s in target:
        (out / s).mkdir(parents=True, exist_ok=True)
        ids = [iid for g, sp in assign.items() if sp == s for iid in groups[g]]
        sub = dict(images=[imgs[i] for i in ids], categories=coco["categories"],
                   annotations=[x for i in ids for x in anns[i]])
        for im in sub["images"]:
            src, dst = Path(a.images) / im["file_name"], out / s / Path(im["file_name"]).name
            if dst.exists():
                dst.unlink()
            (dst.symlink_to(src.resolve()) if a.link else shutil.copy2(src, dst))
            im["file_name"] = dst.name
        (out / s / ANN).write_text(json.dumps(sub))
        print(f"{s}: {len(sub['images'])} images, {len(sub['annotations'])} instances")
    prov = out / "PROVENANCE.md"
    if not prov.exists():
        prov.write_text("# Provenance\n\nTODO: for every image source record: origin, who owns it, licence/consent, "
                        "whether it may be used to train a commercial model, and annotator/guideline version.\n")
        print(f"wrote {prov} - complete it before training")
    rep = validate(out)
    print(rep.render())


if __name__ == "__main__":
    main()
