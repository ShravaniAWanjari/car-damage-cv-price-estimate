"""Fine-tune RF-DETR-Seg (Apache-2.0) on our COCO-seg damage dataset.

    python scripts/finetune_rfdetr.py --dataset-dir data/damage_v1 --size medium --name rfdetr_seg_m \
        --epochs 60 --batch-size 4 --grad-accum 4 --device cuda

Layout/rules: see cardamage/dataset.py. Prepare data with scripts/prepare_coco_splits.py.
Effective batch ~16 (batch-size x grad-accum) is the upstream recommendation: 4x4 on a 16 GB T4,
16x1 on an A100. Output model dir (models/<name>/) is what `backend: rfdetr` in models.yaml loads.

Starter weights are Roboflow's COCO-pretrained RF-DETR-Seg checkpoints, downloaded + MD5-checked by
the rfdetr package. Train-time checkpoints are written by rfdetr in output_dir; we publish only the
weights-only `checkpoint_best_total.pth`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardamage.dataset import ANN, SPLITS, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--name", default=None, help="model dir name under models/ (default rfdetr_seg_<size>)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None, help="override input size (multiple of the model patch size)")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: rfdetr auto)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--resume", default=None, help="checkpoint.pth in the run dir to continue from")
    ap.add_argument("--models-dir", default=str(ROOT / "models"), help="where the exported model dir is written")
    ap.add_argument("--run-dir", default=None, help="training scratch dir (default outputs/train/<name>)")
    ap.add_argument("--progress", choices=["tqdm", "rich", "none"], default="tqdm",
                    help="live progress bar (rfdetr disables it unless asked); 'none' for quiet logs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="validate the dataset and print the plan only")
    ap.add_argument("--force", action="store_true", help="train even if validation reports errors (don't)")
    a = ap.parse_args()

    ds = Path(a.dataset_dir).resolve()
    name = a.name or f"rfdetr_seg_{a.size[0]}"
    model_dir = Path(a.models_dir) / name
    run_dir = Path(a.run_dir) if a.run_dir else ROOT / "outputs" / "train" / name

    rep = validate(ds)
    print(rep.render())
    if not rep.ok and not a.force:
        sys.exit("\ndataset validation failed; fix the errors above (or --force at your own risk)")
    plan = dict(size=a.size, name=name, epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
                effective_batch=a.batch_size * a.grad_accum, lr=a.lr, resolution=a.resolution or "model default",
                classes=list(rep.categories.values()))
    print("\nplan:", json.dumps(plan, indent=1))
    if a.dry_run:
        return

    import rfdetr
    cls = getattr(rfdetr, SIZES[a.size])
    kw = {"resolution": a.resolution} if a.resolution else {}
    if a.device:
        kw["device"] = a.device
    model = cls(**kw)

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Lightning's accelerator defaults to "auto" and would pick MPS/CUDA even when --device cpu was asked for
    accel = {"cpu": "cpu", "mps": "mps", "cuda": "gpu"}.get((a.device or "").split(":")[0])
    model.train(dataset_dir=str(ds), **({"accelerator": accel} if accel else {}), epochs=a.epochs, batch_size=a.batch_size, grad_accum_steps=a.grad_accum,
                lr=a.lr, output_dir=str(run_dir), num_workers=a.num_workers, early_stopping=a.early_stopping,
                seed=a.seed, **({"progress_bar": a.progress} if a.progress != "none" else {}),
                **({"resume": a.resume} if a.resume else {}))
    mins = (time.time() - t0) / 60

    best = run_dir / "checkpoint_best_total.pth"
    if not best.exists():
        sys.exit(f"training finished but {best} was not written; inspect {run_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, model_dir / "checkpoint_best_total.pth")

    from rfdetr import RFDETR
    class_names = list(RFDETR.from_checkpoint(model_dir / "checkpoint_best_total.pth", device="cpu").class_names)
    meta = dict(
        task="segment", rfdetr_class=SIZES[a.size], class_names=class_names, plan=plan,
        resolution=getattr(model.model_config, "resolution", None),
        rfdetr_version=metadata.version("rfdetr"), trained_minutes=round(mins, 1),
        dataset=dict(dir=str(ds), stats=rep.stats,
                     annotations_sha256={s: sha256(ds / s / ANN) for s in SPLITS if (ds / s / ANN).exists()},
                     provenance_sha256={p.name: sha256(p) for p in ds.glob("PROVENANCE*")}),
        checkpoint_sha256=sha256(model_dir / "checkpoint_best_total.pth"),
        base_weights="Roboflow COCO-pretrained " + SIZES[a.size] + " (Apache-2.0)",
    )
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {model_dir}  ({mins:.1f} min)\n")
    print("Add to models.yaml (flip production_ok only after legal clears the training data in PROVENANCE.md):\n")
    print(f"""  {name}:
    dir: {model_dir}
    backend: rfdetr
    role: damage
    conf: 0.30
    taxonomy: {{}}            # labels already use schema names
    class_conf: {{}}          # paste the block printed by scripts/eval_model.py
    license:
      weights: Apache-2.0 (RF-DETR) fine-tuned by us
      data: see {ds}/PROVENANCE.md
      production_ok: false""")
    print(f"\nthen: python scripts/eval_model.py --dataset-dir {ds} --split test --models abdullahg7_v8s_seg {name}")


if __name__ == "__main__":
    main()
