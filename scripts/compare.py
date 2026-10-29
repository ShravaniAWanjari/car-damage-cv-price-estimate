"""Run every converted candidate over a folder of images; write overlays, a
contact sheet per image and a latency/size summary.

    python scripts/compare.py --images samples --out outputs/compare --device cpu
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.loader import load_yolo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="outputs/compare")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    imgs = sorted(Path(a.images).glob("*.jpg"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    tiles = {p.name: {} for p in imgs}
    for mdir in sorted(Path(a.models).iterdir()):
        if a.only and mdir.name not in a.only:
            continue
        y, meta = load_yolo(mdir)
        imgsz = meta["imgsz"] or 640
        params = sum(p.numel() for p in y.model.parameters()) / 1e6
        y.predict(str(imgs[0]), imgsz=imgsz, device=a.device, verbose=False)  # warm-up
        lat, counts = [], {}
        for p in imgs:
            t = time.perf_counter()
            r = y.predict(str(p), imgsz=imgsz, conf=a.conf, device=a.device, verbose=False)[0]
            lat.append((time.perf_counter() - t) * 1000)
            counts[p.name] = len(r.boxes)
            plot = r.plot(line_width=2, font_size=14)
            tiles[p.name][mdir.name] = plot
        summary[mdir.name] = dict(task=meta["task"], imgsz=imgsz, params_m=round(params, 1),
                                  median_ms=round(statistics.median(lat), 1),
                                  imgs_with_detection=sum(1 for v in counts.values() if v),
                                  total_detections=sum(counts.values()), counts=counts)
        print(mdir.name, {k: v for k, v in summary[mdir.name].items() if k != "counts"}, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # contact sheet per image: one tile per model, labelled
    for name, per in tiles.items():
        row = []
        for mname, im in per.items():
            h = 420; w = int(im.shape[1] * h / im.shape[0])
            t = cv2.resize(im, (w, h))
            cv2.rectangle(t, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(t, mname, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(t)
        if not row:
            continue
        cols = 3
        rows = [row[i:i + cols] for i in range(0, len(row), cols)]
        W = max(sum(t.shape[1] for t in r) for r in rows)
        canvas = np.zeros((420 * len(rows), W, 3), np.uint8)
        for ri, r in enumerate(rows):
            x = 0
            for t in r:
                canvas[ri * 420:(ri + 1) * 420, x:x + t.shape[1]] = t; x += t.shape[1]
        cv2.imwrite(str(out / f"sheet_{name}"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
