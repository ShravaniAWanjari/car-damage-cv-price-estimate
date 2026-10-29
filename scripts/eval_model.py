"""Evaluate one or more models from models.yaml on a held-out COCO-seg split.

    python scripts/eval_model.py --dataset-dir data/damage_v1 --split test \
        --models abdullahg7_v8s_seg rfdetr_seg_m --out outputs/eval

Reports (per model):
  * mask / box AP and AP50 (COCO protocol, all classes + per class)
  * class-agnostic mask AP50 ("is there damage here?", independent of label taxonomy)
  * per-class precision / recall / F1 at mask-IoU 0.5 and the F1-optimal confidence,
    emitted as a ready-to-paste `class_conf:` block for models.yaml
  * image level: false-alarm rate on damage-free photos, any-damage recall on damaged ones

Always evaluate on data the model never saw (and, for fair comparison with community
baselines, on photos that are not from CarDD). Box-only models are scored with their
boxes as masks.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.config import Settings  # noqa: E402
from cardamage.dataset import DAMAGE_TYPES, load_coco  # noqa: E402
from cardamage.geometry import box_to_poly  # noqa: E402


def to_rle(poly_norm, w, h):
    m = np.zeros((h, w), np.uint8)
    pts = (np.asarray(poly_norm) * [w, h]).round().astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    rle = mu.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode()
    return rle


def raw_split(analyzer, dataset_dir: Path, split: str, coco: dict, conf: float):
    """Backend output per image, before any post-processing: {image_id: (w, h, [RawDet])}."""
    out, bs = {}, analyzer.s.batch_size
    images = coco["images"]
    for s in range(0, len(images), bs):
        chunk = images[s:s + bs]
        arrs = [cv2.imread(str(dataset_dir / split / i["file_name"])) for i in chunk]
        for im, a, raws in zip(chunk, arrs, analyzer.predict_raw(arrs, conf=conf)):
            out[im["id"]] = (a.shape[1], a.shape[0], raws)
    return out


def dets_from_raw(analyzer, coco: dict, raw: dict, dedupe_fn=None):
    """COCO-style detections from `raw`; `dedupe_fn` post-processes each image's RawDets (None: no dedupe)."""
    tax = analyzer.damage_entry.taxonomy
    type_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    dets = []
    for iid, (w, h, raws) in raw.items():
        for d in (dedupe_fn(raws) if dedupe_fn else raws):
            t = tax.get(d.label, d.label if not tax else "other")
            poly = d.poly if d.poly and len(d.poly) >= 3 else box_to_poly(d.box)
            x1, y1, x2, y2 = d.box
            dets.append(dict(image_id=iid, category_id=type_to_id.get(t, -1), score=float(d.conf),
                             segmentation=to_rle(poly, w, h),
                             bbox=[x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]))
    return dets


def predict_split(analyzer, dataset_dir: Path, split: str, coco: dict, conf: float, dedupe: bool = True):
    raw = raw_split(analyzer, dataset_dir, split, coco, conf)
    # same post-processing the pipeline applies
    return dets_from_raw(analyzer, coco, raw, analyzer.dedupe if dedupe else None)


def _coco_from(d: dict) -> COCO:
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def run_eval(gt: dict, dets: list, iou_type: str, agnostic: bool = False, iou_thrs=None):
    """`iou_thrs` overrides the COCO 0.50:0.95 grid (diagnostics only); E.stats is then not meaningful,
    read E.eval['precision'] instead."""
    gt = json.loads(json.dumps(gt))
    dets = json.loads(json.dumps(dets))
    # category_id -1 = a class the eval set does not annotate: counts only in the class-agnostic score
    if not agnostic:
        dets = [d for d in dets if d["category_id"] != -1]
    if agnostic:
        gt["categories"] = [dict(id=1, name="damage")]
        for a in gt["annotations"]:
            a["category_id"] = 1
        for d in dets:
            d["category_id"] = 1
    for i, a in enumerate(gt["annotations"]):
        a.setdefault("iscrowd", 0)
        a.setdefault("id", i + 1)
        a.setdefault("area", a["bbox"][2] * a["bbox"][3])
    if not dets:
        return None
    cg = _coco_from(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes(dets)
        E = COCOeval(cg, cd, iou_type)
        if iou_thrs is not None:
            E.params.iouThrs = np.asarray(iou_thrs, float)
        E.evaluate(); E.accumulate(); E.summarize()
    return E


def per_class_ap(E, names):
    p = E.eval["precision"]  # T,R,K,A,M
    out = {}
    for k, cid in enumerate(E.params.catIds):
        def m(arr):
            arr = arr[arr > -1]
            return float(arr.mean()) if arr.size else float("nan")
        out[names[cid]] = dict(AP=m(p[:, :, k, 0, -1]), AP50=m(p[0, :, k, 0, -1]))
    return out


def match_operating_points(gt: dict, dets: list, names: dict):
    """Greedy mask-IoU>=0.5 matching per image+class. Returns per-class [(score, tp)] and n_gt."""
    dets = [d for d in dets if d["category_id"] != -1]
    gts = defaultdict(list)
    for a in gt["annotations"]:
        gts[(a["image_id"], a["category_id"])].append(a)
    byk = defaultdict(list)
    for d in dets:
        byk[(d["image_id"], d["category_id"])].append(d)
    recs, n_gt = defaultdict(list), defaultdict(int)
    for (iid, cid), g in gts.items():
        n_gt[cid] += len(g)
    img_hits = defaultdict(list)  # image -> [(score, cid, tp)]
    for key in set(gts) | set(byk):
        g, d = gts.get(key, []), sorted(byk.get(key, []), key=lambda x: -x["score"])
        used = set()
        if g and d:
            grle = [a["segmentation"] for a in g]
            ims = next(i for i in gt["images"] if i["id"] == key[0])
            grle = [mu.merge(mu.frPyObjects(s, ims["height"], ims["width"])) if isinstance(s, list) else s for s in grle]
            iou = mu.iou([x["segmentation"] for x in d], grle, [0] * len(grle))
        for j, x in enumerate(d):
            tp = False
            if g:
                order = np.argsort(-iou[j])
                for gi in order:
                    if iou[j][gi] >= 0.5 and gi not in used:
                        used.add(gi); tp = True; break
            recs[key[1]].append((x["score"], tp))
            img_hits[key[0]].append((x["score"], key[1], tp))
    return recs, n_gt, img_hits


def best_f1_thresholds(recs, n_gt, names):
    out = {}
    for cid, name in names.items():
        r = sorted(recs.get(cid, []), key=lambda t: -t[0])
        n = n_gt.get(cid, 0)
        tp = fp = 0
        best = dict(f1=0.0, thr=None, p=0.0, r=0.0)
        for s, is_tp in r:
            tp += is_tp; fp += (not is_tp)
            p, rec = tp / (tp + fp), (tp / n if n else 0.0)
            f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
            if f1 > best["f1"]:
                best = dict(f1=f1, thr=s, p=p, r=rec)
        best["n_gt"] = n
        out[name] = best
    return out


def metrics_at_thresholds(recs, n_gt, names, thr_by_name):
    """P/R/F1 per class at FIXED thresholds (e.g. calibrated on another split)."""
    out = {}
    for cid, name in names.items():
        thr, n = thr_by_name.get(name), n_gt.get(cid, 0)
        r = recs.get(cid, [])
        tp = sum(1 for sc, t in r if thr is not None and sc >= thr and t)
        fp = sum(1 for sc, t in r if thr is not None and sc >= thr and not t)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n if n else 0.0
        out[name] = dict(p=p, r=rec, f1=2 * p * rec / (p + rec) if p + rec else 0.0, thr=thr, n_gt=n)
    return out


def image_level(gt, img_hits, thr_by_cid):
    pos_imgs = {a["image_id"] for a in gt["annotations"]}
    neg = [i["id"] for i in gt["images"] if i["id"] not in pos_imgs]
    def fires(iid):
        return any(s >= thr_by_cid.get(cid, 1.1) for s, cid, _ in img_hits.get(iid, []))
    return dict(
        n_damage_free=len(neg),
        false_alarm_rate_on_damage_free=(sum(fires(i) for i in neg) / len(neg)) if neg else None,
        any_damage_recall=sum(fires(i) for i in pos_imgs) / len(pos_imgs) if pos_imgs else None)


def evaluate_model(name, a, coco, calib=None):
    """`calib` = COCO dict of the split used to pick per-class thresholds (None: tune on the eval split itself)."""
    from cardamage.pipeline import DamageAnalyzer
    s = Settings.from_env(mode="prod", damage_model=name, device=a.device, max_det=a.max_det,
                          registry=Path(a.registry) if a.registry else None)
    if a.no_dedupe:
        s.dedupe_iou = s.dedupe_containment = 0.0
    an = DamageAnalyzer(s)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    use_dd = not a.no_dedupe
    dets = predict_split(an, Path(a.dataset_dir), a.split, coco, a.sweep_conf, use_dd)
    res = dict(model=name, n_images=len(coco["images"]), n_detections_at_sweep_conf=len(dets), dedupe=use_dd)
    for it in ("segm", "bbox"):
        E = run_eval(coco, dets, it)
        res[it] = None if E is None else dict(AP=E.stats[0], AP50=E.stats[1], AP75=E.stats[2],
                                              per_class=per_class_ap(E, names))
    Eg = run_eval(coco, dets, "segm", agnostic=True)
    res["class_agnostic_segm_AP50"] = None if Eg is None else Eg.stats[1]
    recs, n_gt, img_hits = match_operating_points(coco, dets, names)
    if calib is None:
        res["thresholds_tuned_on"] = f"{a.split} (the same split: optimistic)"
        best = best_f1_thresholds(recs, n_gt, names)
        thr_by_name = {n: b["thr"] for n, b in best.items()}
    else:
        cdets = predict_split(an, Path(a.dataset_dir), a.calibrate_split, calib, a.sweep_conf, use_dd)
        crecs, cn_gt, _ = match_operating_points(calib, cdets, names)
        cbest = best_f1_thresholds(crecs, cn_gt, names)
        thr_by_name = {n: b["thr"] for n, b in cbest.items()}
        best = metrics_at_thresholds(recs, n_gt, names, thr_by_name)
        res["thresholds_tuned_on"] = f"{a.calibrate_split} (frozen, then scored on {a.split})"
    if a.bootstrap:
        res["bootstrap_ci95"] = bootstrap_ci(coco, dets, names, a.bootstrap)
    res["operating_points_iou50"] = best
    thr = {cid: (thr_by_name[n] if thr_by_name.get(n) is not None else 1.1) for cid, n in names.items()}
    res["image_level_at_f1_thresholds"] = image_level(coco, img_hits, thr)
    res["suggested_class_conf"] = {n: round(t, 3) for n, t in thr_by_name.items() if t is not None}
    return res


def bootstrap_ci(gt: dict, dets: list, names: dict, n_boot: int, seed: int = 0):
    """95% CI by resampling IMAGES with replacement (AP is not decomposable, so each draw is a full COCOeval)."""
    rng = np.random.default_rng(seed)
    by_img_a, by_img_d = defaultdict(list), defaultdict(list)
    for x in gt["annotations"]:
        by_img_a[x["image_id"]].append(x)
    for d in dets:
        by_img_d[d["image_id"]].append(d)
    ids = [i["id"] for i in gt["images"]]
    imgs = {i["id"]: i for i in gt["images"]}
    draws = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        g = dict(images=[], annotations=[], categories=gt["categories"])
        dd = []
        for new, old in enumerate(pick, 1):
            g["images"].append({**imgs[old], "id": new})
            for x in by_img_a[old]:
                g["annotations"].append({**x, "image_id": new, "id": len(g["annotations"]) + 1})
            for d in by_img_d[old]:
                dd.append({**d, "image_id": new})
        E = run_eval(g, dd, "segm")
        if E is None:
            continue
        draws["maskAP"].append(E.stats[0]); draws["maskAP50"].append(E.stats[1])
        for n, v in per_class_ap(E, names).items():
            if not np.isnan(v["AP50"]):
                draws[f"AP50[{n}]"].append(v["AP50"])
        Eg = run_eval(g, dd, "segm", agnostic=True)
        if Eg is not None:
            draws["agnosticAP50"].append(Eg.stats[1])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def fmt(x):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:5.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sweep-conf", type=float, default=0.01, help="low conf so AP sees the full ranking")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--calibrate-split", default="valid",
                    help="split on which per-class thresholds are chosen, then frozen for --split ('none' = tune on --split itself)")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N", help="N image-resamples for 95%% confidence intervals (e.g. 200)")
    ap.add_argument("--no-dedupe", action="store_true", help="evaluate raw detections without duplicate suppression")
    ap.add_argument("--registry", default=None, help="alternate models.yaml")
    ap.add_argument("--model-dir", action="append", default=[], metavar="NAME=PATH",
                    help="evaluate a fine-tuned RF-DETR model dir that is not in models.yaml (repeatable)")
    a = ap.parse_args()

    if a.model_dir:  # ad-hoc entries -> temporary registry (inherits the rest of models.yaml)
        import tempfile
        import yaml
        reg_path = Path(a.registry) if a.registry else Path(__file__).resolve().parents[1] / "models.yaml"
        reg = yaml.safe_load(reg_path.read_text())
        for spec in a.model_dir:
            n, _, d = spec.partition("=")
            reg["models"][n] = dict(dir=str(Path(d).resolve()), backend="rfdetr", role="damage", conf=0.3, taxonomy={},
                                    license={"production_ok": False, "weights": "ad-hoc (--model-dir)"})
        tmp = Path(tempfile.mkdtemp()) / "registry.yaml"
        tmp.write_text(yaml.safe_dump(reg))
        a.registry = str(tmp)
    coco = load_coco(Path(a.dataset_dir), a.split)
    if coco is None:
        sys.exit(f"{a.dataset_dir}/{a.split} has no annotations")
    calib = None
    if a.calibrate_split.lower() != "none":
        if a.calibrate_split == a.split:
            print(f"note: --calibrate-split equals --split ({a.split}); thresholds are tuned on the data they are scored on")
        else:
            calib = load_coco(Path(a.dataset_dir), a.calibrate_split)
            if calib is None:
                print(f"warning: no '{a.calibrate_split}' split; falling back to tuning thresholds on '{a.split}'")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allres = [evaluate_model(m, a, coco, calib) for m in a.models]
    (out / f"eval_{a.split}.json").write_text(json.dumps(allres, indent=1))

    print(f"\n{a.split}: {len(coco['images'])} images | thresholds: {allres[0]['thresholds_tuned_on']} | "
          f"duplicate suppression: {'on' if allres[0]['dedupe'] else 'off'}\n")
    print(f"{'model':28s} {'maskAP':>7s} {'maskAP50':>9s} {'boxAP':>6s} {'agnAP50':>8s} {'FA@neg':>7s} {'anyRec':>7s}")
    for r in allres:
        il = r["image_level_at_f1_thresholds"]
        print(f"{r['model']:28s} {fmt(r['segm'] and r['segm']['AP']):>7s} {fmt(r['segm'] and r['segm']['AP50']):>9s} "
              f"{fmt(r['bbox'] and r['bbox']['AP']):>6s} {fmt(r['class_agnostic_segm_AP50']):>8s} "
              f"{fmt(il['false_alarm_rate_on_damage_free']):>7s} {fmt(il['any_damage_recall']):>7s}")
    for r in allres:
        print(f"\n== {r['model']}: per-class (mask IoU 0.5, P/R/F1 at the chosen thresholds)")
        for n, b in r["operating_points_iou50"].items():
            ap_ = (r["segm"] or {}).get("per_class", {}).get(n, {})
            print(f"  {n:14s} n_gt={b['n_gt']:4d}  AP={fmt(ap_.get('AP'))} AP50={fmt(ap_.get('AP50'))}  "
                  f"P={b['p']:.2f} R={b['r']:.2f} F1={b['f1']:.2f}  conf>={b['thr'] if b['thr'] is None else round(b['thr'], 3)}")
        if r.get("bootstrap_ci95"):
            ci = r["bootstrap_ci95"]
            print("  95% CI (image bootstrap): " + ", ".join(f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in ci.items()
                                                       if k in ("maskAP", "maskAP50", "agnosticAP50")))
            print("  per-class AP50 95% CI:    " + ", ".join(f"{k[5:-1]} [{v[0]:.2f}, {v[1]:.2f}]" for k, v in ci.items()
                                                       if k.startswith("AP50[")))
        if r["suggested_class_conf"]:
            print("  models.yaml ->  class_conf: " + json.dumps(r["suggested_class_conf"]))
    print(f"\nwrote {out / f'eval_{a.split}.json'}")


if __name__ == "__main__":
    main()
