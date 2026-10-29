"""COCO-segmentation dataset layout + validation shared by the training and eval scripts.

    dataset/
      PROVENANCE.md                 # where every image came from + its licence (required)
      train/  _annotations.coco.json + images
      valid/  _annotations.coco.json + images
      test/   _annotations.coco.json + images   (held out; used only by eval_model.py)

Categories must be names from cardamage.schema.DamageType. Image entries may carry a
`group` (or `claim_id`) field - photos of the same claim/vehicle - which is used to
check the splits do not leak.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, get_args

from .schema import DamageType

SPLITS = ("train", "valid", "test")
ANN = "_annotations.coco.json"
DAMAGE_TYPES = set(get_args(DamageType))
MIN_TRAIN_INSTANCES = 50


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, dict] = field(default_factory=dict)
    categories: Dict[int, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = []
        for split, st in self.stats.items():
            lines.append(f"[{split}] images={st['images']} (empty={st['empty_images']}) "
                         f"instances={st['instances']}  per-class={st['per_class']}")
        lines += [f"WARNING: {w}" for w in self.warnings] + [f"ERROR:   {e}" for e in self.errors]
        return "\n".join(lines)


def load_coco(dataset_dir: Path, split: str) -> Optional[dict]:
    p = dataset_dir / split / ANN
    return json.loads(p.read_text()) if p.exists() else None


def group_of(img: dict) -> Optional[str]:
    g = img.get("group", img.get("claim_id"))
    return None if g is None else str(g)


def validate(dataset_dir: Path) -> Report:
    dataset_dir = Path(dataset_dir)
    r = Report()
    if not any(dataset_dir.glob("PROVENANCE*")):
        r.errors.append("no PROVENANCE.md: record the source and licence of every image before training "
                        "(this is the licensing blocker from docs/MODEL_SELECTION.md)")
    cats_by_split: Dict[str, Dict[int, str]] = {}
    groups_by_split: Dict[str, set] = defaultdict(set)
    names_by_split: Dict[str, set] = defaultdict(set)
    for split in SPLITS:
        coco = load_coco(dataset_dir, split)
        if coco is None:
            (r.errors if split != "test" else r.warnings).append(f"missing {split}/{ANN}")
            continue
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        cats_by_split[split] = cats
        bad = sorted(n for n in cats.values() if n not in DAMAGE_TYPES)
        if bad:
            r.errors.append(f"[{split}] categories not in schema taxonomy {sorted(DAMAGE_TYPES)}: {bad}")
        imgs = {i["id"]: i for i in coco.get("images", [])}
        if len(imgs) != len(coco.get("images", [])):
            r.errors.append(f"[{split}] duplicate image ids")
        per_class, with_ann = Counter(), set()
        for a in coco.get("annotations", []):
            img = imgs.get(a["image_id"])
            if img is None:
                r.errors.append(f"[{split}] annotation {a.get('id')} references unknown image")
                continue
            if a["category_id"] not in cats:
                r.errors.append(f"[{split}] annotation {a.get('id')} unknown category")
                continue
            seg = a.get("segmentation")
            if not isinstance(seg, list) or not seg or any(len(p) < 6 for p in seg):
                r.errors.append(f"[{split}] annotation {a.get('id')} has no polygon segmentation "
                                f"(RLE/empty not supported by this pipeline)")
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > img["width"] + 1 or y + h > img["height"] + 1:
                r.errors.append(f"[{split}] annotation {a.get('id')} bbox outside image")
                continue
            per_class[cats[a["category_id"]]] += 1
            with_ann.add(a["image_id"])
        missing = [i["file_name"] for i in imgs.values() if not (dataset_dir / split / i["file_name"]).exists()]
        if missing:
            r.errors.append(f"[{split}] {len(missing)} image files missing, e.g. {missing[:3]}")
        for i in imgs.values():
            names_by_split[split].add(Path(i["file_name"]).name)
            g = group_of(i)
            if g is not None:
                groups_by_split[split].add(g)
        r.stats[split] = dict(images=len(imgs), empty_images=len(imgs) - len(with_ann),
                              instances=sum(per_class.values()), per_class=dict(per_class))
        if split == "train":
            for n in cats.values():
                if per_class[n] < MIN_TRAIN_INSTANCES:
                    r.warnings.append(f"class {n!r} has only {per_class[n]} train instances "
                                      f"(< {MIN_TRAIN_INSTANCES}); expect weak recall")
            if imgs and (len(imgs) - len(with_ann)) / len(imgs) < 0.05:
                r.warnings.append("<5% negative (damage-free) train images; add undamaged cars "
                                  "to reduce the false positives seen with the community models")
    # same category ids/names everywhere
    ref = next(iter(cats_by_split.values()), None)
    for split, cats in cats_by_split.items():
        if ref is not None and cats != ref:
            r.errors.append(f"[{split}] category ids/names differ from the first split")
    if ref:
        r.categories = ref
    for split, st in r.stats.items():
        if split != "train":
            for n in r.categories.values():
                if st["per_class"].get(n, 0) == 0:
                    r.warnings.append(f"[{split}] has no instances of {n!r}; its metrics will be undefined")
    # leakage
    present = [s for s in SPLITS if s in names_by_split]
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            shared = groups_by_split[a] & groups_by_split[b]
            if shared:
                r.errors.append(f"LEAK: {len(shared)} group(s) appear in both {a} and {b}, e.g. {sorted(shared)[:3]}")
            dup = names_by_split[a] & names_by_split[b]
            if dup:
                r.errors.append(f"LEAK: {len(dup)} file name(s) in both {a} and {b}, e.g. {sorted(dup)[:3]}")
    if not any(groups_by_split.values()):
        r.warnings.append("images carry no `group`/`claim_id`: cannot verify that photos of one claim/vehicle "
                          "stay in a single split (use scripts/prepare_coco_splits.py)")
    return r
