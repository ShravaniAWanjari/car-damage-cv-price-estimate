"""Diagnostics for a fine-tuned model on one split. NOT an acceptance report: the acceptance bar stays
mask AP50 at IoU 0.5 from eval_model.py. These numbers answer "why is it low?", so always show them next to it.

  1. per-class AP at looser/stricter overlap, masks and boxes (is the model finding the damage but
     disagreeing with annotators about its exact extent?)
  2. how much of the output is stacked (predictions mostly covered by another prediction)
  3. what stricter duplicate suppression would do to AP and to the number of regions shown

    python scripts/diagnose_model.py --dataset-dir data/hitl_damage_v2 --split test --model rfdetr_seg_s_v2 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_model import dets_from_raw, per_class_ap, raw_split, run_eval  # noqa: E402

from cardamage.config import Settings  # noqa: E402
from cardamage.dataset import load_coco  # noqa: E402
from cardamage.pipeline import DamageAnalyzer  # noqa: E402
from cardamage.postprocess import _mask, dedupe  # noqa: E402

SHOW_CONF = 0.30  # what the demo displays


def ap_single(E):
    """Mean AP over classes at the single IoU in E.params (area 'all', maxDets 100); nan if no detections."""
    if E is None:
        return float("nan")
    p = E.eval["precision"][0, :, :, 0, -1]
    return float(p[p > -1].mean())


def ap_table(coco, dets, names, thrs=(0.3, 0.5)):
    """{(kind, thr): {class or 'ALL(agnostic)': AP}} where AP is at that single IoU."""
    out = {}
    for kind, it in (("mask", "segm"), ("box", "bbox")):
        for t in thrs:
            E = run_eval(coco, dets, it, iou_thrs=[t])
            Ea = run_eval(coco, dets, it, agnostic=True, iou_thrs=[t])
            row = {n: v["AP"] for n, v in per_class_ap(E, names).items()} if E else {}
            if Ea:
                row["ALL(agnostic)"] = ap_single(Ea)
            out[(kind, t)] = row
    return out


def stacking(raw, conf=SHOW_CONF):
    """Among displayed predictions (conf >= `conf`, after the default dedupe): how many are mostly covered by another."""
    n = same = cross = involved = 0
    for _, (_, _, raws) in raw.items():
        ds = dedupe([d for d in raws if d.conf >= conf])
        ms = [_mask(d) for d in ds]
        hit = set()
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = int(ms[i].sum()), int(ms[j].sum())
                if not a or not b:
                    continue
                if int((ms[i] & ms[j]).sum()) / min(a, b) >= 0.5:
                    hit |= {i, j}
                    if ds[i].label == ds[j].label:
                        same += 1
                    else:
                        cross += 1
        n += len(ds)
        involved += len(hit)
    return dict(shown=n, involved=involved, same_label_pairs=same, cross_label_pairs=cross)


def drop_cross_class(dets, containment=0.85):
    """Experiment only: drop a detection mostly inside a higher-confidence one of ANOTHER label."""
    kept, masks = [], []
    for d in sorted(dets, key=lambda x: -x.conf):
        m = _mask(d)
        area = int(m.sum())
        if area and any(k.label != d.label and int((m & km).sum()) / area >= containment
                        for k, km in zip(kept, masks)):
            continue
        kept.append(d)
        masks.append(m)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--out", default="outputs/eval_v2")
    a = ap.parse_args()

    an = DamageAnalyzer(Settings.from_env(mode="prod", damage_model=a.model, device=a.device, max_det=100,
                                          registry=Path(a.registry) if a.registry else None))
    coco = load_coco(Path(a.dataset_dir), a.split)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    raw = raw_split(an, Path(a.dataset_dir), a.split, coco, 0.01)   # predict once, post-process many ways
    n_img = len(coco["images"])

    print(f"{a.model} on {a.split} ({n_img} images)\n")
    dets = dets_from_raw(an, coco, raw, an.dedupe)                   # = what eval_model.py scores
    tab = ap_table(coco, dets, names)
    cols = [("mask", 0.3), ("mask", 0.5), ("box", 0.3), ("box", 0.5)]
    print("1. AP at a single overlap threshold (default dedupe). mask@0.5 is the acceptance metric.")
    print(f"   {'class':16s}" + "".join(f"{k}@{t:<4}  " for k, t in cols))
    for n in [*names.values(), "ALL(agnostic)"]:
        print(f"   {n:16s}" + "".join(f"{tab[c].get(n, float('nan')):>9.3f} " for c in cols))

    st = stacking(raw)
    print(f"\n2. Stacking among the {st['shown']} regions shown at conf>={SHOW_CONF}: {st['involved']} "
          f"({100 * st['involved'] / max(1, st['shown']):.0f}%) are >=50% covered by another one "
          f"(same-label pairs {st['same_label_pairs']}, cross-label pairs {st['cross_label_pairs']})")

    variants = {
        "no dedupe": None,
        "default (IoU .5 / contain .85)": lambda r: dedupe(r, 0.5, 0.85),
        "stricter (IoU .3 / contain .6)": lambda r: dedupe(r, 0.3, 0.6),
        "default + drop cross-label inside": lambda r: drop_cross_class(dedupe(r, 0.5, 0.85)),
        "stricter + drop cross-label inside": lambda r: drop_cross_class(dedupe(r, 0.3, 0.6)),
    }
    print("\n3. Suppression variants (mask AP50 / agnostic AP50 / box AP50; regions per image shown at conf>=.30)")
    res = {}
    for label, fn in variants.items():
        d = dets_from_raw(an, coco, raw, fn)
        E, Ea, Eb = run_eval(coco, d, "segm", iou_thrs=[0.5]), run_eval(coco, d, "segm", True, [0.5]), \
            run_eval(coco, d, "bbox", iou_thrs=[0.5])
        f = ap_single  # all-area, maxDets=100, the single IoU set above
        shown = sum(len([x for x in (fn(r) if fn else r) if x.conf >= SHOW_CONF]) for _, _, r in raw.values()) / n_img
        res[label] = dict(mask_AP50=f(E), agnostic_AP50=f(Ea), box_AP50=f(Eb), regions_per_image=shown)
        print(f"   {label:36s} {f(E):.3f}  {f(Ea):.3f}  {f(Eb):.3f}   {shown:.1f}/img")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"diagnose_{a.split}.json").write_text(json.dumps(dict(
        ap_by_overlap={f"{k}@{t}": v for (k, t), v in tab.items()}, stacking=st, suppression_variants=res), indent=1))
    print(f"\nwrote {out / f'diagnose_{a.split}.json'}")


if __name__ == "__main__":
    main()
