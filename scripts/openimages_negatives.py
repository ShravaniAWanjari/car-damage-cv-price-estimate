"""Candidate hard negatives (damage-free cars, street scenes with people) from Open Images V7 validation.

    python scripts/openimages_negatives.py select   --root data/raw/openimages_negatives
    python scripts/openimages_negatives.py download --root data/raw/openimages_negatives

select   reads meta/*.csv, keeps images licensed CC BY (per-image `License` column), not depictions, with
         human-verified Car / Person boxes, and writes candidates.csv (with author + landing URL for attribution):
           car   : one dominant, un-grouped car (>= 30% of the frame)        -> normal "healthy car" views
           street: >= 1 person and only small cars (< 25% of the frame)      -> pedestrians/traffic context
download fetches only the candidates from the Open Images S3 mirror (polite: 4 threads, retries).
Nothing here says the cars are undamaged: screen the images (see scripts/build_negatives.py) before using them.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

CAR, PERSON = "/m/0k4j", "/m/01g317"
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
S3 = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
UA = {"User-Agent": "cardamage-dev/0.1 (internal research)"}


def select(root: Path, n_car: int, n_street: int, seed: int):
    meta = root / "meta"
    lic = Counter()
    info = {}
    with open(meta / "validation-images-with-rotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            lic[r["License"]] += 1
            if r["License"] == CC_BY:
                info[r["ImageID"]] = r
    print("licence column:", dict(lic.most_common(4)), "| kept CC BY:", len(info))
    cars, persons = defaultdict(list), Counter()
    with open(meta / "validation-annotations-bbox.csv", newline="") as f:
        for r in csv.DictReader(f):
            iid, lab = r["ImageID"], r["LabelName"]
            if iid not in info or lab not in (CAR, PERSON) or r["Confidence"] != "1" or r["IsDepiction"] == "1":
                continue
            if lab == PERSON:
                if r["IsGroupOf"] != "1":
                    persons[iid] += 1
            else:
                area = (float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"]))
                cars[iid].append((area, r["IsGroupOf"] == "1"))
    car_pool, street_pool = [], []
    for iid, cs in cars.items():
        big = max(a for a, g in cs)
        grouped = any(g for _, g in cs)
        if big >= 0.30 and not grouped and len(cs) <= 3:
            car_pool.append((iid, "car", big, persons[iid]))
        elif persons[iid] >= 1 and big < 0.25:
            street_pool.append((iid, "street", big, persons[iid]))
    rng = random.Random(seed)
    print(f"pools: car {len(car_pool)}, street {len(street_pool)}")
    pick = rng.sample(car_pool, min(n_car, len(car_pool))) + rng.sample(street_pool, min(n_street, len(street_pool)))
    with open(root / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ImageID", "group", "car_area", "n_person", "License", "Author", "OriginalLandingURL", "Title"])
        for iid, g, a, p in pick:
            i = info[iid]
            w.writerow([iid, g, round(a, 3), p, i["License"], i["Author"], i["OriginalLandingURL"], i["Title"]])
    print(f"wrote {len(pick)} candidates -> {root / 'candidates.csv'}")


def fetch(iid: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return "cached", 0
    for k in range(3):
        try:
            data = urlopen(Request(S3.format(iid), headers=UA), timeout=40).read()
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if im is None or min(im.shape[:2]) < 200:
                return "bad", 0
            dst.write_bytes(data)
            return "ok", len(data)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return "failed", 0


def download(root: Path):
    ids = [r["ImageID"] for r in csv.DictReader(open(root / "candidates.csv", newline=""))]
    out = root / "images"
    out.mkdir(exist_ok=True)
    stat, total = Counter(), 0
    with ThreadPoolExecutor(4) as ex:
        for s, n in ex.map(lambda i: fetch(i, out / f"{i}.jpg"), ids):
            stat[s] += 1
            total += n
    print(f"downloaded {dict(stat)}; {total / 1e6:.0f} MB new")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "download"])
    ap.add_argument("--root", default="data/raw/openimages_negatives")
    ap.add_argument("--n-car", type=int, default=250)
    ap.add_argument("--n-street", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    select(root, a.n_car, a.n_street, a.seed) if a.cmd == "select" else download(root)
