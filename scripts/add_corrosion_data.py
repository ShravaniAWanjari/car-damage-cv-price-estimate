"""Build dataset v3 = v2 + extra corrosion photos, plus a separate external corrosion test set.

Source: a Roboflow COCO-segmentation export (e.g. car-rust-corrosion). Only images whose labels are ALL free-form
polygons are used: Roboflow users often draw plain rectangles, which would teach the model box-shaped masks.

    python scripts/add_corrosion_data.py --src data/raw/roboflow_car_rust_corrosion/train \
        --base data/hitl_damage_v2 --out data/hitl_damage_v3 --ext-out data/ext_corrosion_test

  v3/train, v3/valid = v2 + new photos (split by group, near-duplicates stay together)
  v3/test            = v2 test, unchanged (so v2 and v3 are scored on the same photos)
  ext/test           = ~8 new photos no model trained on v3 sees; ext/valid = v3 valid, only there so
                       eval_model.py can calibrate thresholds on it (score only the corrosion row on ext/test)
Images that look like near-duplicates of any v2 photo are dropped, not split.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import ANN, SPLITS, validate  # noqa: E402

HASH_DIST = 10  # of 256 bits (matches the dedup check used when v1 was built)


def is_rect(seg) -> bool:
    """A single 4-corner, axis-aligned polygon: a bounding box drawn as a 'polygon'."""
    if not isinstance(seg, list) or len(seg) != 1 or len(seg[0]) != 8:
        return False
    xs, ys = seg[0][0::2], seg[0][1::2]
    return len({round(x) for x in xs}) <= 2 and len({round(y) for y in ys}) <= 2


def dhash(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (17, 16))
    return (g[:, 1:] > g[:, :-1]).flatten()


def poly_geom(seg, w, h):
    """Clip polygons to the image; return (segmentation, bbox xywh, area) or None if degenerate."""
    parts, xs, ys, area = [], [], [], 0.0
    for p in seg:
        pts = np.asarray(p, float).reshape(-1, 2)
        pts[:, 0] = pts[:, 0].clip(0, w)
        pts[:, 1] = pts[:, 1].clip(0, h)
        if len(pts) < 3:
            continue
        x, y = pts[:, 0], pts[:, 1]
        a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if a < 1:
            continue
        parts.append([round(float(v), 2) for v in pts.flatten()])
        xs += list(x); ys += list(y); area += a
    if not parts:
        return None
    return parts, [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], area


def link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_split(out: Path, split: str, images, annotations, categories):
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    (d / ANN).write_text(json.dumps(dict(images=images, annotations=annotations, categories=categories)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with _annotations.coco.json + images (Roboflow export)")
    ap.add_argument("--base", default="data/hitl_damage_v2")
    ap.add_argument("--out", default="data/hitl_damage_v3")
    ap.add_argument("--ext-out", default="data/ext_corrosion_test")
    ap.add_argument("--class-name", default="corrosion", help="class in the source export to import")
    ap.add_argument("--n-ext", type=int, default=8, help="images held out as the external test")
    ap.add_argument("--n-valid", type=int, default=6, help="images added to valid")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, base, out, ext = Path(a.src), Path(a.base), Path(a.out), Path(a.ext_out)
    for p in (out, ext):
        if p.exists():
            sys.exit(f"{p} already exists; remove it (it only holds symlinks and json) or choose another path")

    rf = json.loads((src / ANN).read_text())
    cid_src = next(c["id"] for c in rf["categories"] if c["name"] == a.class_name)
    imgs = {i["id"]: i for i in rf["images"]}
    anns = defaultdict(list)
    for x in rf["annotations"]:
        if x["category_id"] == cid_src:
            anns[x["image_id"]].append(x)

    # 1. eligible = labelled, every label free-form
    stats = dict(total=len(imgs), no_label=0, has_rectangle=0)
    eligible = {}
    for iid, im in imgs.items():
        xs = anns.get(iid, [])
        if not xs:
            stats["no_label"] += 1
        elif any(is_rect(x["segmentation"]) for x in xs):
            stats["has_rectangle"] += 1
        else:
            eligible[iid] = im

    # 2. drop near-duplicates of any v2 photo, group near-duplicates among themselves
    base_coco = {s: json.loads((base / s / ANN).read_text()) for s in SPLITS}
    base_hash = []
    for s, c in base_coco.items():
        for im in c["images"]:
            base_hash.append((s, im["file_name"], dhash(cv2.imread(str((base / s / im["file_name"]).resolve())))))
    hashes, leaks = {}, []
    for iid, im in list(eligible.items()):
        h = dhash(cv2.imread(str(src / im["file_name"])))
        hit = next(((s, f) for s, f, bh in base_hash if (bh != h).sum() <= HASH_DIST), None)
        if hit:
            leaks.append((im["file_name"], hit)); del eligible[iid]
        else:
            hashes[iid] = h
    ids = list(hashes)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            if (hashes[ids[x]] != hashes[ids[y]]).sum() <= HASH_DIST:
                parent[find(ids[x])] = find(ids[y])
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)

    # 3. assign groups: ext test, then valid, rest train (seeded)
    order = sorted(groups.values(), key=lambda g: g[0])
    random.Random(a.seed).shuffle(order)
    where, count = {}, defaultdict(int)
    for g in order:
        s = "ext" if count["ext"] < a.n_ext else "valid" if count["valid"] < a.n_valid else "train"
        for i in g:
            where[i] = s
        count[s] += len(g)

    # 4. write v3 (+ ext)
    cats = base_coco["train"]["categories"]
    cid = next(c["id"] for c in cats if c["name"] == "corrosion")
    new = {"train": ([], []), "valid": ([], []), "ext": ([], [])}
    n_poly = defaultdict(int)
    for k, iid in enumerate(sorted(ids, key=lambda i: imgs[i]["file_name"])):
        im, s = imgs[iid], where[iid]
        name = f"rfcorr_{k:03d}{Path(im['file_name']).suffix.lower()}"
        w, h = im["width"], im["height"]
        keep = []
        for x in anns[iid]:
            g = poly_geom(x["segmentation"], w, h)
            if g:
                keep.append(g)
        if not keep:
            continue
        new[s][0].append(dict(id=-1, file_name=name, width=w, height=h, group=f"rf_{find(iid)}", _src=str(src / im["file_name"])))
        for seg, bbox, area in keep:
            new[s][1].append(dict(image_name=name, category_id=cid, iscrowd=0, area=round(area, 1),
                                  bbox=[round(v, 1) for v in bbox], segmentation=seg))
        n_poly[s] += len(keep)

    def merged(split):
        c = base_coco[split]
        (out / split).mkdir(parents=True, exist_ok=True)
        for p in (base / split).glob("*"):
            if p.name != ANN:
                link(p, out / split / p.name)
        return list(c["images"]), list(c["annotations"])

    def add_new(split, ims, ans, key):
        nid = max([i["id"] for i in ims], default=0)
        nan = max([x["id"] for x in ans], default=0)
        ids_by_name = {}
        for im in new[key][0]:
            nid += 1
            ids_by_name[im["file_name"]] = nid
            dst_dir = (out if key != "ext" else ext) / ("test" if key == "ext" else split)
            dst_dir.mkdir(parents=True, exist_ok=True)
            link(Path(im["_src"]), dst_dir / im["file_name"])
            ims.append({k: v for k, v in im.items() if k != "_src"} | {"id": nid})
        for x in new[key][1]:
            nan += 1
            ans.append({k: v for k, v in x.items() if k != "image_name"} | {"id": nan, "image_id": ids_by_name[x["image_name"]]})
        return ims, ans

    for split in ("train", "valid"):
        ims, ans = merged(split)
        ims, ans = add_new(split, ims, ans, split)
        write_split(out, split, ims, ans, cats)
    ims, ans = merged("test")  # v2 test, unchanged
    write_split(out, "test", ims, ans, cats)

    ims_e, ans_e = add_new("test", [], [], "ext")
    write_split(ext, "test", ims_e, ans_e, cats)
    (ext / "valid").mkdir(parents=True, exist_ok=True)
    for p in (out / "valid").glob("*"):  # ext/valid = v3 valid, for threshold calibration only
        if p.name != ANN:
            link(p, ext / "valid" / p.name)
    shutil.copy(out / "valid" / ANN, ext / "valid" / ANN)

    shutil.copy(base / "taxonomy.yaml", out / "taxonomy.yaml")
    note = f"""

## Added for v3: extra corrosion photos (Roboflow Universe "car-rust-corrosion")
| | |
|---|---|
| Source | https://universe.roboflow.com/workspace-cars/car-rust-corrosion (export v1, 2023-11-20, "Provided by a Roboflow user", workspace `workspace-cars`) |
| Licence | **CC BY 4.0** per the export's README.dataset.txt: attribution required. Applies to the annotator's labels; **the photos appear to be web-sourced and their own copyright is unverified**: development and demo only. |
| Selection | {stats['total']} images; kept only the {len(eligible)} whose labels are all free-form polygons (dropped {stats['has_rectangle']} with rectangle labels, {stats['no_label']} unlabelled, {len(leaks)} near-duplicates of v2 photos). 'car-rust-corrosion' parent class ignored; 'corrosion' imported. |
| Split | {count['train']} train / {count['valid']} valid / {count['ext']} external test (held out of v3 entirely: see data/ext_corrosion_test), polygons {dict(n_poly)}; near-duplicates stay in one split; seed {a.seed}. |
| Domain note | Mostly close-ups of rust on a single panel, unlike the wide accident shots elsewhere in the set. |
"""
    (out / "PROVENANCE.md").write_text((base / "PROVENANCE.md").read_text() + note)
    ext.mkdir(parents=True, exist_ok=True)
    (ext / "PROVENANCE.md").write_text(
        "# External corrosion test (development only)\n" + note +
        "\n`test/` = the held-out photos above. `valid/` = a copy of hitl_damage_v3/valid so eval_model.py can calibrate "
        "thresholds; score only the **corrosion** row here (other classes have no labels on these photos).\n")

    print(f"eligible {len(eligible)} of {stats['total']} (rectangles {stats['has_rectangle']}, unlabelled {stats['no_label']}, "
          f"near-dup of v2 {len(leaks)}) -> train {count['train']} / valid {count['valid']} / ext {count['ext']}; polygons {dict(n_poly)}")
    for f, hit in leaks:
        print(f"  dropped {f}: near-duplicate of v2 {hit}")
    r = validate(out)  # ext has no train split by design, so it is not validated as a full dataset
    print(f"\n== validate {out}\n{r.render()}\n-> {'OK' if r.ok else 'ERRORS'}")
    print(f"external test written to {ext}/test ({len(ims_e)} images, {len(ans_e)} polygons)")


if __name__ == "__main__":
    main()
