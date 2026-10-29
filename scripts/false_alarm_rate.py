"""False-alarm rate on damage-free photos: the share of images where the model reports >= 1 damage.

    python scripts/false_alarm_rate.py --dataset-dir data/neg_test --split test --model rfdetr_seg_s_v2 [--model-dir NAME=PATH]

Reports, per confidence threshold, with and without the car-region (person) filter: images flagged, Wilson 95% CI,
mean regions per image and the classes responsible. Every image in the set must be damage-free (see its PROVENANCE.md);
a damaged photo slipping in would count as a false alarm, so the rate is, if anything, slightly pessimistic.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.config import Settings  # noqa: E402
from cardamage.dataset import load_coco  # noqa: E402
from cardamage.pipeline import DamageAnalyzer  # noqa: E402
from cardamage.vehicle import VehicleDetector, gate  # noqa: E402

THRESHOLDS = (0.3, 0.5, 0.7)


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/neg_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="rfdetr_seg_s_v2")
    ap.add_argument("--model-dir", action="append", default=[], metavar="NAME=PATH", help="fine-tuned RF-DETR dir not in models.yaml")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    registry = None
    if a.model_dir:
        import tempfile
        import yaml
        reg = yaml.safe_load((Path(__file__).resolve().parents[1] / "models.yaml").read_text())
        for spec in a.model_dir:
            n, _, d = spec.partition("=")
            reg["models"][n] = dict(dir=str(Path(d).resolve()), backend="rfdetr", role="damage", conf=0.3, taxonomy={},
                                    license={"production_ok": False, "weights": "ad-hoc"})
        registry = Path(tempfile.mkdtemp()) / "registry.yaml"
        registry.write_text(yaml.safe_dump(reg))
    d = Path(a.dataset_dir)
    coco = load_coco(d, a.split)
    kinds = {r["file"]: r["split_group"] for r in csv.DictReader(open(d / "ATTRIBUTION.csv", newline=""))} if (d / "ATTRIBUTION.csv").exists() else {}
    an = DamageAnalyzer(Settings.from_env(mode="prod", damage_model=a.model, device=a.device, max_det=100, registry=registry))
    vd = VehicleDetector(a.device)
    files = [i["file_name"] for i in coco["images"]]
    imgs = [cv2.imread(str((d / a.split / f).resolve())) for f in files]
    raws = [r for s in range(0, len(imgs), 8) for r in an.predict_raw(imgs[s:s + 8], conf=min(THRESHOLDS))]
    regs = [r for s in range(0, len(imgs), 4) for r in vd.regions(imgs[s:s + 4])]
    n = len(files)
    print(f"{a.model} on {d.name}/{a.split}: {n} damage-free images ({dict(Counter(kinds.get(f, '?') for f in files))})")
    for label, filt in (("no filter", False), ("person filter", True)):
        print(f"\n[{label}]")
        for t in THRESHOLDS:
            fired, regions, by_cls = [], [], Counter()
            for f, raw, reg in zip(files, raws, regs):
                dets = [x for x in raw if x.conf >= t]
                dets = gate(dets, reg)[0] if filt else dets
                dets = an.dedupe(dets)
                fired.append(bool(dets)); regions.append(len(dets))
                by_cls.update(x.label for x in dets)
            k = sum(fired)
            lo, hi = wilson(k, n)
            per = {g: f"{sum(x for x, f in zip(fired, files) if kinds.get(f) == g)}/{sum(1 for f in files if kinds.get(f) == g)}"
                   for g in sorted(set(kinds.values()))}
            print(f"  conf>={t}: {k}/{n} images flagged = {100 * k / n:.0f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%); "
                  f"{sum(regions) / n:.1f} regions/img; by group {per}; classes {dict(by_cls.most_common())}")


if __name__ == "__main__":
    main()
