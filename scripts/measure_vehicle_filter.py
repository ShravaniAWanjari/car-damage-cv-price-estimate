"""What does the car-region filter (cardamage/vehicle.py) remove, and does it cost real detections?

  samples  (unlabelled street/accident photos): detections kept/removed + a montage to eyeball them
  split    (labelled COCO split): the same counts, mask AP50 before/after, and how many REMOVED detections
           matched a ground-truth region (IoU >= 0.5), i.e. true positives the filter would throw away

    python scripts/measure_vehicle_filter.py --model rfdetr_seg_s_v2 --samples samples \
        --dataset-dir data/hitl_damage_v2 --split test --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_model import ap_single  # noqa: E402
from eval_model import dets_from_raw, per_class_ap, raw_split, run_eval, to_rle  # noqa: E402

from cardamage.config import Settings  # noqa: E402
from cardamage.dataset import load_coco  # noqa: E402
from cardamage.geometry import box_to_poly  # noqa: E402
from cardamage.pipeline import DamageAnalyzer  # noqa: E402
from cardamage.vehicle import VehicleDetector, gate  # noqa: E402

SHOW = 0.30
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def regions_for(vd, imgs, bs=4):
    out = []
    for s in range(0, len(imgs), bs):
        out += vd.regions(imgs[s:s + bs])
    return out


def shown(an, dets):
    """What the pipeline would hand to the output stage at the demo threshold."""
    return an.dedupe([d for d in dets if d.conf >= SHOW])


def summarise(name, an, raws, regs, args):
    c_kept, c_removed, reasons = Counter(), Counter(), Counter()
    n_no_vehicle = n_all_removed = n_img_removed = 0
    per_img = []
    for raw, reg in zip(raws, regs):
        kept, removed = gate(shown(an, raw), reg, args.min_overlap, args.max_person)
        for d in kept:
            c_kept[d.label] += 1
        for d, why in removed:
            c_removed[d.label] += 1; reasons[why] += 1
        n_no_vehicle += reg.n_vehicle == 0
        n_img_removed += bool(removed)
        n_all_removed += bool(removed) and not kept
        per_img.append((kept, removed))
    tot_k, tot_r = sum(c_kept.values()), sum(c_removed.values())
    print(f"\n== {name}: {len(raws)} images, {tot_k + tot_r} regions shown at conf>={SHOW} without the filter")
    print(f"   removed {tot_r} ({100 * tot_r / max(1, tot_k + tot_r):.0f}%): {dict(reasons)}; kept {tot_k}")
    print(f"   images with a detected vehicle: {len(raws) - n_no_vehicle}/{len(raws)} (no vehicle found -> kept as close-up); "
          f"images losing >=1 region: {n_img_removed}; images losing all regions: {n_all_removed}")
    print("   per class removed/kept: " + ", ".join(f"{k} {c_removed[k]}/{c_kept[k] + c_removed[k]}"
                                                    for k in sorted(set(c_kept) | set(c_removed))))
    return per_img, dict(images=len(raws), removed=tot_r, kept=tot_k, reasons=dict(reasons), images_with_vehicle=len(raws) - n_no_vehicle,
                         images_losing_any=n_img_removed, images_losing_all=n_all_removed,
                         removed_by_class=dict(c_removed), kept_by_class=dict(c_kept))


def montage(imgs, per_img, ids, path, limit=12, w=520):
    tiles = []
    for im, (kept, removed), iid in zip(imgs, per_img, ids):
        if not removed or len(tiles) >= limit:
            continue
        h0, w0 = im.shape[:2]
        o = im.copy()
        th = max(2, max(h0, w0) // 250)
        for d, col in [(d, (0, 200, 0)) for d in kept] + [(d, (0, 0, 255)) for d, _ in removed]:
            pts = (np.asarray(d.poly or box_to_poly(d.box)) * [w0, h0]).astype(np.int32)
            cv2.polylines(o, [pts], True, col, th, cv2.LINE_AA)
        for d, why in removed:
            x, y = int(d.box[0] * w0), int(d.box[1] * h0)
            cv2.putText(o, f"X {d.label} {d.conf:.2f} ({why})", (x, max(20, y)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, w0 / 1400), (0, 0, 255), max(1, th // 2), cv2.LINE_AA)
        t = cv2.resize(o, (w, int(h0 * w / w0)))
        cv2.putText(t, iid, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return None
    cols = 3
    hmax = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles[i:i + cols]]
        row += [np.zeros((hmax, w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def tp_removed(coco, raw_by_id, removed_by_id, an):
    """Of the detections the filter removed (conf >= SHOW), how many overlap a ground-truth region of the same class (IoU>=.5)?"""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    gts = {}
    for a in coco["annotations"]:
        gts.setdefault((a["image_id"], names[a["category_id"]]), []).append(a)
    imgs = {i["id"]: i for i in coco["images"]}
    hit = total = 0
    for iid, removed in removed_by_id.items():
        w, h = imgs[iid]["width"], imgs[iid]["height"]
        for d, _ in removed:
            total += 1
            g = gts.get((iid, d.label), [])
            if not g:
                continue
            grle = [mu.merge(mu.frPyObjects(a["segmentation"], h, w)) for a in g]
            drle = to_rle(d.poly or box_to_poly(d.box), w, h)
            if mu.iou([drle], grle, [0] * len(grle)).max() >= 0.5:
                hit += 1
    return hit, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vehicle-conf", type=float, default=0.35)
    ap.add_argument("--min-overlap", type=float, default=0.0, help="outside-vehicle rule threshold; 0 (default) = off, people rule only")
    ap.add_argument("--max-person", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/vehicle_filter")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    an = DamageAnalyzer(Settings.from_env(mode="prod", damage_model=args.model, device=args.device, max_det=100))
    vd = VehicleDetector(args.device, conf=args.vehicle_conf)
    report = {}

    if args.samples:
        files = sorted(f for f in Path(args.samples).iterdir() if f.suffix.lower() in EXTS)
        imgs = [cv2.imread(str(f)) for f in files]
        raws = [r for s in range(0, len(imgs), 4) for r in an.predict_raw(imgs[s:s + 4], conf=SHOW)]
        per_img, report["samples"] = summarise("samples", an, raws, regions_for(vd, imgs), args)
        m = montage(imgs, per_img, [f.stem for f in files], out / "samples_removed.jpg")
        print(f"   montage of images with removals (green kept, red removed): {m}")

    if args.dataset_dir:
        d = Path(args.dataset_dir)
        coco = load_coco(d, args.split)
        names = {c["id"]: c["name"] for c in coco["categories"]}
        raw = raw_split(an, d, args.split, coco, 0.01)            # {image_id: (w, h, [RawDet])}
        ids = [i["id"] for i in coco["images"]]
        imgs = [cv2.imread(str((d / args.split / i["file_name"]).resolve())) for i in coco["images"]]
        regs = regions_for(vd, imgs)
        per_img, report["split"] = summarise(f"{d.name}/{args.split}", an, [raw[i][2] for i in ids], regs, args)
        gated = {i: (raw[i][0], raw[i][1], gate(raw[i][2], r, args.min_overlap, args.max_person)[0]) for i, r in zip(ids, regs)}
        res = {}
        for label, rr in (("without filter", raw), ("with filter", gated)):
            dets = dets_from_raw(an, coco, rr, an.dedupe)
            E, Ea = run_eval(coco, dets, "segm", iou_thrs=[0.5]), run_eval(coco, dets, "segm", True, [0.5])
            res[label] = dict(mask_AP50=ap_single(E), agnostic_AP50=ap_single(Ea),
                              per_class={n: v["AP50"] for n, v in per_class_ap(E, names).items()})
        print("\n   mask AP50 (IoU 0.5)        " + "".join(f"{k:>16s}" for k in res))
        print(f"   {'all classes':27s}" + "".join(f"{v['mask_AP50']:16.3f}" for v in res.values()))
        print(f"   {'any damage (agnostic)':27s}" + "".join(f"{v['agnostic_AP50']:16.3f}" for v in res.values()))
        for n in names.values():
            print(f"   {n:27s}" + "".join(f"{v['per_class'].get(n, float('nan')):16.3f}" for v in res.values()))
        rem = {i: r for i, (_, r) in zip(ids, per_img) if r}
        hit, tot = tp_removed(coco, raw, rem, an)
        print(f"\n   removed detections that matched a ground-truth region (IoU>=.5, same class): {hit} of {tot}")
        report["split"].update(ap=res, removed_matching_gt=hit, removed_total=tot)

    (out / "vehicle_filter.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out / 'vehicle_filter.json'}")


if __name__ == "__main__":
    main()
