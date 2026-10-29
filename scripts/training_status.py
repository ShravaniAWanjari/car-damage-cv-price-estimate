"""Per-epoch table (and ETA) for a running or finished fine-tune, read from the run dir's metrics.csv.

    python scripts/training_status.py outputs/train/rfdetr_seg_s            # once
    python scripts/training_status.py outputs/train/rfdetr_seg_s --watch 30  # refresh every 30 s
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path


def rows(run: Path):
    f = run / "metrics.csv"
    if not f.exists():
        return {}
    by = {}
    for r in csv.DictReader(f.open()):
        if r.get("epoch") not in (None, ""):
            by.setdefault(int(float(r["epoch"])), {}).update({k: v for k, v in r.items() if v not in ("", None)})
    return by


def pick(r, *keys):
    for k in keys:
        if k in r:
            return float(r[k])
    return None


def show(run: Path, total):
    by = rows(run)
    if not by:
        print(f"no metrics yet in {run} (the first epoch has to finish)"); return
    mtime = (run / "metrics.csv").stat().st_mtime
    print(f"{'epoch':>5} {'mask mAP50':>11} {'mask mAP50-95':>14} {'box mAP50':>10} {'train loss':>11}")
    best = 0.0
    for e in sorted(by):
        r = by[e]
        m50, m95, b50 = (pick(r, "val/segm_mAP_50"), pick(r, "val/segm_mAP_50_95"), pick(r, "val/mAP_50"))
        loss = pick(r, "train/loss", "train/loss_epoch")
        best = max(best, m50 or 0)
        f = lambda v: "     n/a" if v is None else f"{v:8.4f}"
        print(f"{e + 1:>5} {f(m50):>11} {f(m95):>14} {f(b50):>10} {f(loss):>11}" + ("  <- best" if m50 == best and m50 else ""))
    done = len(by)
    ckpt = run / "last.ckpt"
    if total and done >= 2:
        st = (run / "metrics.csv").stat()
        first = getattr(st, "st_birthtime", None)  # metrics.csv is created when epoch 1 finishes
        if first and mtime > first:
            per = (mtime - first) / (done - 1)
            print(f"\n{done}/{total} epochs, ~{per / 60:.1f} min/epoch, ~{per * (total - done) / 3600:.1f} h left")
    elif total:
        print(f"\n{done}/{total} epochs; the ETA appears after epoch 2")
    print(f"last update {time.strftime('%H:%M:%S', time.localtime(mtime))} | resume ckpt: {'yes' if ckpt.exists() else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--total-epochs", type=int, default=None, help="for the ETA line")
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    a = ap.parse_args()
    run = Path(a.run_dir)
    while True:
        if a.watch:
            os.system("clear")
        show(run, a.total_epochs)
        if not a.watch:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
