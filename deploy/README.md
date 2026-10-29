# GPU fine-tune Job (RF-DETR-Seg)

Runs `scripts/finetune_rfdetr.py` on a GPU node: validate dataset -> train (auto-resume) -> export model -> evaluate on the held-out split.

```
deploy/train.Dockerfile        image (python:3.11 + pinned torch/rfdetr, pretrained weights baked in, non-root)
deploy/entrypoint_train.sh     the job's steps, configured by env vars; idempotent
deploy/k8s/pvc.yaml            dataset volume + output volume
deploy/k8s/data-loader-pod.yaml  throwaway pod to copy the dataset onto the volume
deploy/k8s/job-finetune.yaml   the Job (GPU request, read-only dataset mount, /dev/shm, resume on retry)
requirements-job.txt           pinned deps for the image (no ultralytics/AGPL code)
```

## 1. Build and push the image

```bash
docker build -f deploy/train.Dockerfile -t REGISTRY/cardamage-train:TAG \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .     # pick the CUDA build your node driver supports
docker push REGISTRY/cardamage-train:TAG
```
`PRETRAIN_SIZES` (default `"medium large"`) controls which Roboflow COCO-pretrained starter weights are baked in, so the Job needs no internet.
Without `TORCH_INDEX_URL` you get the default PyPI torch wheel; the node's NVIDIA driver must be new enough for it - the entrypoint prints
the torch CUDA build and fails fast if no GPU is visible.

## 2. Storage and data

```bash
kubectl apply -f deploy/k8s/pvc.yaml                       # set storageClassName first
kubectl apply -f deploy/k8s/data-loader-pod.yaml
kubectl -n car-damage wait --for=condition=Ready pod/cardamage-data-loader
tar -C data/damage_v1 -ch . | kubectl -n car-damage exec -i cardamage-data-loader -- tar -C /data -x
kubectl -n car-damage delete pod cardamage-data-loader
```
The dataset must already pass `python scripts/finetune_rfdetr.py --dataset-dir <dir> --dry-run` locally (COCO-seg, `PROVENANCE.md`,
no train/valid/test leakage - see `cardamage/dataset.py`). The Job re-validates and refuses to train otherwise. `tar -h` follows
symlinks, so a dataset made with `prepare_coco_splits.py --link` uploads real files.

## 3. Run

Edit the image and `env` block in `deploy/k8s/job-finetune.yaml` (size, epochs, batch), then:

```bash
kubectl create -f deploy/k8s/job-finetune.yaml          # create, not apply: the name is generated
kubectl -n car-damage get jobs,pods -l component=finetune
kubectl -n car-damage logs -f job/<generated-name>
```

| env | default | notes |
|---|---|---|
| `SIZE` | medium | nano / small / medium / large |
| `MODEL_NAME` | rfdetr_seg_m | output dir name; must match the `models.yaml` entry you'll use |
| `EPOCHS` | 60 | total, also when resuming |
| `BATCH_SIZE` x `GRAD_ACCUM` | 4 x 4 | effective ~16: 4x4 on a 16 GB GPU, 16x1 on 40/80 GB |
| `LR` | 1e-4 | |
| `NUM_WORKERS` | 4 | keep `/dev/shm` mounted; workers crash without it |
| `RESOLUTION` | model default | optional override |
| `EARLY_STOPPING` / `SKIP_EVAL` | 0 / 0 | |

**Failure handling:** `backoffLimit: 1` retries once; the retry resumes from `/output/runs/<name>/last.ckpt` (written every epoch) and skips
training if `/output/models/<name>/meta.json` already exists. An evaluation failure exits 3 but leaves the trained model intact.

## 4. Results

Outputs on the `cardamage-train-output` volume:
`models/<name>/{checkpoint_best_total.pth,meta.json}` (what `backend: rfdetr` loads), `eval/<name>/eval_test.{json,txt}` (mask/box AP, per-class P/R/F1,
a `class_conf` block to paste into `models.yaml`), `runs/<name>/` (metrics.csv, checkpoints). To fetch them, mount the output claim in a throwaway pod
(copy `data-loader-pod.yaml`, change `claimName`) and `kubectl cp`. Then copy `models/<name>` into this repo's `models/` and register it in `models.yaml`
(`production_ok` stays `false` until legal clears the data in `PROVENANCE.md`).

## What was and wasn't verified (read this)

Verified on the dev machine (CPU, no Docker): `entrypoint_train.sh` end to end on a read-only dataset copy - validation, training, **kill and resume from
`last.ckpt`**, export, evaluation via `--model-dir`, skip-on-rerun, and fail-fast when `DEVICE=cuda` has no GPU. Both manifests parse as YAML and the
Job's volumes/mounts are consistent.

**Not verified** (no Docker, kubectl, cluster or GPU available): the image build, the pinned `torch==2.14.1`/`torchvision==0.29.1` CUDA wheels on Linux, any GPU
training run, `kubectl` schema validation, the resource requests (CPU 6 / memory 24-32 Gi are guesses), and whether the cluster allows the securityContext /
`/dev/shm` emptyDir as written. Expect to adjust `nodeSelector`/`tolerations`, storage classes and resources on the first run; check
`kubectl create --dry-run=server -f deploy/k8s/job-finetune.yaml` before submitting. Multi-GPU is not configured (single `nvidia.com/gpu: 1`).
Do not train on Apple MPS (memory blow-up); this job is CUDA/CPU only.
