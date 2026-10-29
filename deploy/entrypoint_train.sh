#!/usr/bin/env bash
# Fine-tune Job entrypoint: validate -> train (auto-resume) -> export -> evaluate on the held-out split.
# Configured only through env vars (defaults below). Idempotent: safe to re-run after a pod restart/eviction.
set -euo pipefail

APP="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-/data}"          # COCO-seg layout, see cardamage/dataset.py (mounted read-only is fine)
OUTPUT_DIR="${OUTPUT_DIR:-/output}"          # persistent volume: runs/, models/, eval/
SIZE="${SIZE:-medium}"                        # nano | small | medium | large
MODEL_NAME="${MODEL_NAME:-rfdetr_seg_${SIZE:0:1}}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"                 # effective batch = BATCH_SIZE x GRAD_ACCUM (~16 recommended)
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
RESOLUTION="${RESOLUTION:-}"                  # empty = model default
EARLY_STOPPING="${EARLY_STOPPING:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

RUN_DIR="$OUTPUT_DIR/runs/$MODEL_NAME"
MODELS_DIR="$OUTPUT_DIR/models"
EVAL_DIR="$OUTPUT_DIR/eval/$MODEL_NAME"
mkdir -p "$RUN_DIR" "$MODELS_DIR" "$EVAL_DIR"
cd "$APP"

echo "== config: size=$SIZE name=$MODEL_NAME epochs=$EPOCHS batch=${BATCH_SIZE}x${GRAD_ACCUM} lr=$LR device=$DEVICE dataset=$DATASET_DIR output=$OUTPUT_DIR"

if [ "$DEVICE" = "cuda" ]; then
  command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "nvidia-smi not found"
  python - <<'PY'
import sys, torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| cuda available:", ok,
      "|", torch.cuda.get_device_name(0) if ok else "no GPU visible")
sys.exit(0 if ok else "CUDA requested but no GPU is visible: check the GPU resource request, node driver, "
         "and that the torch CUDA build (TORCH_INDEX_URL) matches the driver")
PY
fi

if [ -f "$MODELS_DIR/$MODEL_NAME/meta.json" ]; then
  echo "== $MODELS_DIR/$MODEL_NAME already exported; skipping training"
else
  args=(--dataset-dir "$DATASET_DIR" --size "$SIZE" --name "$MODEL_NAME" --epochs "$EPOCHS"
        --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --lr "$LR" --num-workers "$NUM_WORKERS"
        --device "$DEVICE" --models-dir "$MODELS_DIR" --run-dir "$RUN_DIR")
  [ -n "$RESOLUTION" ] && args+=(--resolution "$RESOLUTION")
  [ "$EARLY_STOPPING" = "1" ] && args+=(--early-stopping)
  if [ -f "$RUN_DIR/last.ckpt" ]; then
    echo "== resuming from $RUN_DIR/last.ckpt"
    args+=(--resume "$RUN_DIR/last.ckpt")
  fi
  python scripts/finetune_rfdetr.py "${args[@]}"
fi

rc=0
if [ "$SKIP_EVAL" != "1" ] && [ -f "$DATASET_DIR/test/_annotations.coco.json" ]; then
  echo "== evaluating on the held-out test split"
  python scripts/eval_model.py --dataset-dir "$DATASET_DIR" --split test --models "$MODEL_NAME" \
      --model-dir "$MODEL_NAME=$MODELS_DIR/$MODEL_NAME" --out "$EVAL_DIR" --device "$DEVICE" | tee "$EVAL_DIR/eval_test.txt" || rc=3
else
  echo "== no test split (or SKIP_EVAL=1): skipping evaluation"
fi
[ "$rc" -ne 0 ] && echo "!! EVALUATION FAILED (exit $rc); the trained model is intact in $MODELS_DIR/$MODEL_NAME"
echo "== done: model in $MODELS_DIR/$MODEL_NAME, eval in $EVAL_DIR"
exit "$rc"
