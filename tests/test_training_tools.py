import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardamage.backends.base import RawDet  # noqa: E402
from cardamage.backends.rfdetr_backend import RFDETRBackend, mask_to_polygon  # noqa: E402
from cardamage.config import ModelEntry, Settings  # noqa: E402
from cardamage.dataset import validate  # noqa: E402


def _write_split(root: Path, split: str, items, cats=("dent", "scratch")):
    d = root / split
    d.mkdir(parents=True)
    images, anns = [], []
    for n, (fname, group, with_ann) in enumerate(items, 1):
        cv2.imwrite(str(d / fname), np.zeros((100, 100, 3), np.uint8))
        images.append(dict(id=n, file_name=fname, width=100, height=100, group=group))
        if with_ann:
            anns.append(dict(id=n, image_id=n, category_id=1, iscrowd=0, bbox=[10, 10, 20, 20], area=400,
                             segmentation=[[10, 10, 30, 10, 30, 30, 10, 30]]))
    (d / "_annotations.coco.json").write_text(json.dumps(
        dict(images=images, annotations=anns, categories=[dict(id=i + 1, name=c) for i, c in enumerate(cats)])))


def _ds(tmp_path, train, valid, test=None, provenance=True, cats=("dent", "scratch")):
    root = tmp_path / "ds"
    _write_split(root, "train", train, cats)
    _write_split(root, "valid", valid, cats)
    if test:
        _write_split(root, "test", test, cats)
    if provenance:
        (root / "PROVENANCE.md").write_text("all ours")
    return root


def test_validate_clean_dataset(tmp_path):
    r = validate(_ds(tmp_path, [("a.jpg", "c1", True)], [("b.jpg", "c2", True)], [("c.jpg", "c3", True)]))
    assert r.ok, r.errors


def test_validate_flags_leak_missing_provenance_and_bad_category(tmp_path):
    root = _ds(tmp_path, [("a.jpg", "claim1", True)], [("b.jpg", "claim1", True)],
               provenance=False, cats=("dent", "banana"))
    r = validate(root)
    text = " | ".join(r.errors)
    assert "LEAK" in text and "PROVENANCE" in text and "banana" in text


def test_validate_rejects_missing_segmentation(tmp_path):
    root = _ds(tmp_path, [("a.jpg", "c1", True)], [("b.jpg", "c2", True)])
    p = root / "train" / "_annotations.coco.json"
    j = json.loads(p.read_text())
    j["annotations"][0]["segmentation"] = []
    p.write_text(json.dumps(j))
    assert any("segmentation" in e for e in validate(root).errors)


def test_prepare_splits_never_splits_a_group(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    images, anns, k = [], [], 0
    for claim in range(30):
        for shot in range(3):
            k += 1
            f = f"CLM{claim:03d}_{shot}.jpg"
            cv2.imwrite(str(src / f), np.zeros((64, 64, 3), np.uint8))
            images.append(dict(id=k, file_name=f, width=64, height=64))
            anns.append(dict(id=k, image_id=k, category_id=1 + claim % 2, iscrowd=0, bbox=[5, 5, 20, 20], area=400,
                             segmentation=[[5, 5, 25, 5, 25, 25, 5, 25]]))
    (tmp_path / "all.json").write_text(json.dumps(dict(images=images, annotations=anns,
        categories=[dict(id=1, name="dent"), dict(id=2, name="scratch")])))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_coco_splits.py"), "--coco", str(tmp_path / "all.json"),
                    "--images", str(src), "--out", str(out), "--group-regex", r"^(CLM\d+)_"], check=True, capture_output=True)
    seen = {}
    for s in ("train", "valid", "test"):
        for im in json.loads((out / s / "_annotations.coco.json").read_text())["images"]:
            seen.setdefault(im["group"], set()).add(s)
    assert len(seen) == 30 and all(len(v) == 1 for v in seen.values())
    assert not [e for e in validate(out).errors if "LEAK" in e]


def test_mask_to_polygon_normalised():
    m = np.zeros((50, 100), bool)
    m[10:30, 20:60] = True
    poly = mask_to_polygon(m)
    xs, ys = zip(*poly)
    assert min(xs) == pytest.approx(0.20, abs=0.02) and max(xs) == pytest.approx(0.59, abs=0.02)
    assert min(ys) == pytest.approx(0.20, abs=0.02) and max(ys) == pytest.approx(0.58, abs=0.02)
    assert mask_to_polygon(np.zeros((10, 10), bool)) is None


def test_rfdetr_backend_missing_checkpoint_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="finetune_rfdetr"):
        RFDETRBackend("x", tmp_path)


def test_class_conf_and_taxonomy_filtering():
    from cardamage.pipeline import DamageAnalyzer
    an = DamageAnalyzer.__new__(DamageAnalyzer)
    an.s = Settings()
    an.damage_entry = ModelEntry(dir="x", role="damage", taxonomy={"Dent": "dent", "Wreck": "not_a_type"},
                                 class_conf={"dent": 0.5})
    sq = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)]
    box = (0.1, 0.1, 0.4, 0.4)
    dets = [RawDet("Dent", 0.4, box, sq), RawDet("Dent", 0.7, box, sq), RawDet("Wreck", 0.9, box, sq)]
    out = an._damages(dets, "img", [])
    assert [(d.type, d.confidence) for d in out] == [("other", 0.9), ("dent", 0.7)]  # 0.4 dent dropped by class_conf


def test_supervisely_converter_maps_classes_and_groups_duplicates(tmp_path):
    root = tmp_path / "sv"
    (root / "img").mkdir(parents=True)
    (root / "ann").mkdir()
    rng = np.random.default_rng(0)
    base = (rng.random((80, 120, 3)) * 255).astype(np.uint8)
    other = (rng.random((80, 120, 3)) * 255).astype(np.uint8)
    obj = lambda cls: dict(classTitle=cls, geometryType="polygon",
                           points=dict(exterior=[[10, 10], [60, 10], [60, 50], [10, 50]], interior=[]))
    for name, im, cls in [("a.png", base, "Scratch"), ("b.png", base, "Cracked"), ("c.png", other, "Paint chip")]:
        cv2.imwrite(str(root / "img" / name), im)
        (root / "ann" / f"{name}.json").write_text(json.dumps(
            dict(size=dict(width=120, height=80), objects=[obj(cls)])))
    out = tmp_path / "all.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_supervisely.py"), "--root", str(root),
                    "--out", str(out)], check=True, capture_output=True)
    coco = json.loads(out.read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    assert sorted(names.values()) == ["crack", "paint_chip", "scratch"]
    assert sorted(names) == [1, 2, 3]  # contiguous ids
    g = {im["file_name"]: im["group"] for im in coco["images"]}
    assert g["a.png"] == g["b.png"] != g["c.png"]  # identical photos share a group


def _det(label, conf, x1, y1, x2, y2):
    return RawDet(label, conf, (x1, y1, x2, y2), [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_dedupe_merges_same_label_overlaps_only():
    from cardamage.postprocess import dedupe
    big = _det("dent", 0.9, 0.1, 0.1, 0.6, 0.6)
    near_dup = _det("dent", 0.5, 0.12, 0.12, 0.6, 0.62)         # IoU well above 0.5 -> dropped
    inside = _det("dent", 0.4, 0.2, 0.2, 0.4, 0.4)              # fully inside the big one -> dropped (containment)
    other_label = _det("scratch", 0.6, 0.12, 0.12, 0.58, 0.58)  # same area, different label -> kept
    far = _det("dent", 0.7, 0.7, 0.7, 0.95, 0.95)               # elsewhere -> kept
    out = dedupe([near_dup, inside, other_label, big, far])
    assert sorted((d.label, d.conf) for d in out) == [("dent", 0.7), ("dent", 0.9), ("scratch", 0.6)]
    assert len(dedupe([near_dup, big], iou_thr=0, containment_thr=0)) == 2  # rules can be disabled


def test_pipeline_applies_dedupe_before_output():
    from cardamage.pipeline import DamageAnalyzer
    an = DamageAnalyzer.__new__(DamageAnalyzer)
    an.s = Settings()
    an.damage_entry = ModelEntry(dir="x", role="damage", taxonomy={})
    dets = [_det("dent", 0.9, 0.1, 0.1, 0.6, 0.6), _det("dent", 0.5, 0.11, 0.11, 0.6, 0.6)]
    assert [d.confidence for d in an._damages(dets, "i", [])] == [0.9]


def test_frozen_thresholds_are_applied_not_retuned():
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_model import best_f1_thresholds, metrics_at_thresholds
    names = {1: "dent"}
    calib = {1: [(0.9, True), (0.6, True), (0.3, False)]}
    thr = {n: b["thr"] for n, b in best_f1_thresholds(calib, {1: 2}, names).items()}
    assert thr["dent"] == 0.6                                    # best F1 on the calibration split
    held = {1: [(0.95, True), (0.5, True), (0.7, False)]}        # on another split, 0.6 keeps 0.95(TP) + 0.7(FP)
    m = metrics_at_thresholds(held, {1: 3}, names, thr)["dent"]
    assert (m["p"], round(m["r"], 3), m["thr"]) == (0.5, 0.333, 0.6)


def test_remap_classes_merges_and_clusters_regions(tmp_path):
    root = tmp_path / "v1"
    cats = [dict(id=1, name="scratch"), dict(id=2, name="paint_chip"), dict(id=3, name="dent")]
    for split in ("train", "valid", "test"):
        d = root / split
        d.mkdir(parents=True)
        cv2.imwrite(str(d / f"{split}.jpg"), np.zeros((200, 400, 3), np.uint8))
        sq = lambda x, y, s=8: [[x, y, x + s, y, x + s, y + s, x, y + s]]
        anns = [  # three tiny marks within 3% of each other (-> one region), one far away, one dent
            dict(id=1, image_id=1, category_id=1, iscrowd=0, bbox=[100, 100, 8, 8], area=64, segmentation=sq(100, 100)),
            dict(id=2, image_id=1, category_id=2, iscrowd=0, bbox=[112, 100, 8, 8], area=64, segmentation=sq(112, 100)),
            dict(id=3, image_id=1, category_id=1, iscrowd=0, bbox=[124, 102, 8, 8], area=64, segmentation=sq(124, 102)),
            dict(id=4, image_id=1, category_id=1, iscrowd=0, bbox=[300, 20, 8, 8], area=64, segmentation=sq(300, 20)),
            dict(id=5, image_id=1, category_id=3, iscrowd=0, bbox=[20, 20, 60, 60], area=3600, segmentation=sq(20, 20, 60)),
        ]
        (d / "_annotations.coco.json").write_text(json.dumps(dict(
            images=[dict(id=1, file_name=f"{split}.jpg", width=400, height=200, group=split)], annotations=anns, categories=cats)))
    (root / "PROVENANCE.md").write_text("test")
    spec = tmp_path / "spec.yaml"
    spec.write_text("name: t\nclasses: {dent: [dent], paint_damage: [scratch, paint_chip]}\n"
                    "region_merge: {paint_damage: 0.03}\nmin_region_area_pct: 0.0\n")
    out = tmp_path / "v2"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "remap_classes.py"), "--src", str(root), "--out", str(out),
                    "--spec", str(spec)], check=True, capture_output=True)
    c = json.loads((out / "train" / "_annotations.coco.json").read_text())
    names = {x["id"]: x["name"] for x in c["categories"]}
    got = sorted(names[a["category_id"]] for a in c["annotations"])
    assert got == ["dent", "paint_damage", "paint_damage"]  # 4 tiny marks -> 2 regions (3 merged + 1 far)
    assert (out / "train" / "train.jpg").is_symlink() and (out / "PROVENANCE.md").exists()


def test_corrosion_import_helpers():
    sys.path.insert(0, str(ROOT / "scripts"))
    from add_corrosion_data import is_rect, poly_geom
    assert is_rect([[10, 10, 50, 10, 50, 40, 10, 40]])                       # drawn box
    assert not is_rect([[10, 10, 50, 12, 55, 40, 12, 38]])                    # free-form quad
    assert not is_rect([[10, 10, 50, 10, 50, 40, 10, 40], [0, 0, 5, 0, 5, 5]])  # multi-part
    seg, bbox, area = poly_geom([[-5, -5, 20, -5, 20, 10, -5, 10]], 100, 100)   # clipped to the image
    assert bbox == [0, 0, 20, 10] and area == 200
    assert poly_geom([[1, 1, 1, 1, 1, 1]], 100, 100) is None                    # degenerate


def test_rfdetr_background_slot_is_not_a_detection():
    class Fake:
        confidence = np.array([0.9, 0.8])
        xyxy = np.array([[10, 10, 50, 50], [20, 20, 60, 60]], float)
        data = {"class_name": np.array(["__background__", "dent"])}
        mask = None
        def __len__(self):
            return 2
    out = RFDETRBackend._raw(Fake(), 100, 100, 10)
    assert [d.label for d in out] == ["dent"]
