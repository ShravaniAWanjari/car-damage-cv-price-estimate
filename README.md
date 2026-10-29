# cardamage - YOLO pipeline stage (damage detection -> JEV input)

Image(s) in, JEV input JSON out. Two modes, one code path:

| | demo | prod |
|---|---|---|
| `result.json` (JEV input) | yes | yes (identical payload) |
| annotated images + `report.html` | yes | **never** (renderer isn't imported) |

## Run

```bash
source .venv311/bin/activate   # Python 3.11 (rfdetr needs >=3.10)
python -m cardamage run --mode demo --input samples --out outputs/demo   # open outputs/demo/report.html
python -m cardamage run --mode prod --input samples --out outputs/prod   # result.json only
python -m cardamage run --mode prod --input a.jpg --stdout               # print JSON for piping
```

API (same modes via env):

```bash
CARDAMAGE_MODE=demo uvicorn cardamage.api:app --port 8000
curl -F files=@samples/s25.jpg localhost:8000/analyze
```

**Demo website** (demo mode only): `CARDAMAGE_MODE=demo CARDAMAGE_DEVICE=cpu python -m uvicorn cardamage.api:app --port 8000`, then open http://localhost:8000.
Drop photos or click a sample; it shows the overlay drawn in the browser from the JSON polygons (type filter chips, min-confidence slider),
the server-rendered annotated image, a result table and the JEV-input JSON (copy/download). `GET /schema` returns the JSON Schema of that
payload for the JEV side. In prod mode `/`, `/api/*`, `/demo` and `/samples` do not exist (tested in `tests/test_api.py`). Pick the model with
`CARDAMAGE_DAMAGE_MODEL=rfdetr_seg_s_v2`. Running it during a CPU training run slows both; stop it while training.

Other flags: `--damage-model <name from models.yaml>`, `--parts-model aitools_parts` (experimental, unreliable), `--conf`, `--device cpu|cuda:0`, `--require-cleared` (refuse weights not licence-cleared; also `CARDAMAGE_REQUIRE_CLEARED=1`).

## Output contract (`cardamage/schema.py`, v0.1.0)

Per image: `damages[]` with `type` (dent, scratch, crack, glass_shatter, lamp_broken, tire_flat, broken_part, other), `raw_label`, `confidence`, normalised `bbox` and `polygon`, `area_ratio_image`, optional `panel`; plus model name/sha256/classes and per-image latency. Model labels are mapped to the stable taxonomy in `models.yaml`, so swapping models doesn't change what JEV sees. Severity and Go/No-Go are JEV's job; we only provide geometry + type + confidence.

## Layout

- `cardamage/` pipeline, schema, renderer (demo only), CLI, API
- `models.yaml` model registry: weights dir, label mapping, licence status
- `scripts/safe_convert.py` untrusted `.pt` -> safetensors without executing pickle
- `cardamage/backends/` model backends (`ultralytics`, `rfdetr`) behind one interface
- `cardamage/dataset.py` dataset layout + validator; `scripts/prepare_coco_splits.py`, `finetune_rfdetr.py`, `eval_model.py`
- `scripts/compare.py` run every candidate on a folder, write overlays + latency
- `models/` converted weights (git-ignore these), `samples/` CC-licensed test photos (`SOURCES.json`)
- `docs/MODEL_SELECTION.md` **read this** - model comparison, cost, and the licensing blockers

## Production track: fine-tune RF-DETR-Seg on our data

```bash
# 1. split ONE annotated COCO-seg export by claim (no leakage between splits)
python scripts/prepare_coco_splits.py --coco all.json --images photos/ --out data/damage_v1 --group-key claim_id
#    fill in data/damage_v1/PROVENANCE.md (source + licence of every image)  -- validation fails without it
# 2. validate + train (GPU node; effective batch 16 = 4 x 4 on a T4)
python scripts/finetune_rfdetr.py --dataset-dir data/damage_v1 --size medium --epochs 60 --device cuda
# 3. compare against the community baseline on the held-out split, get per-class thresholds
python scripts/eval_model.py --dataset-dir data/damage_v1 --split test --models abdullahg7_v8s_seg rfdetr_seg_m
# 4. paste the printed class_conf into models.yaml, then run it:
python -m cardamage run --mode demo --damage-model rfdetr_seg_m --input samples --out outputs/demo_rf
```

Dataset rules (enforced by `cardamage/dataset.py`): COCO polygons (no RLE), categories from the schema taxonomy, `PROVENANCE.md`
present, no group/file overlap across splits, >=5% damage-free photos recommended (the community models' main failure
was false positives on healthy cars), warnings below 50 train instances per class. Use `--dry-run` to only validate.
`eval_model.py` reports mask/box AP, class-agnostic AP50, per-class P/R/F1 at IoU 0.5, and the false-alarm rate on
damage-free photos. Train only on `train/valid`; keep `test/` untouched until the final comparison.

Not verified here (no GPU, no real data): training quality, fp16 serving (`half` is ignored by the RF-DETR backend),
and TensorRT/ONNX export. Only the CPU pipeline was smoke-tested (2-epoch nano fine-tune on pseudo-labels, which proves the
plumbing, not accuracy).

## Sample dataset and what we measured (dev only)

Until client data arrives we use **Humans in the Loop "Car Parts and Car Damages"** (CC0 1.0 per the creator): 814 damage photos,
8 classes (scratch, dent, broken_part, paint_chip, missing_part, flaking, corrosion, crack), polygon masks. Its 998 car-part
photos (21 part classes, e.g. front/back bumper, hood, doors) are in `data/raw/hitl_parts` - a licence-clean start for the
"damage side" parts model. Photo origin is undisclosed by the creator: see `data/hitl_damage_v1/PROVENANCE.md` before shipping anything trained on it.

```bash
python scripts/convert_supervisely.py --root "data/raw/hitl_car_damages/Car parts dataset/File1" --out data/work/hitl_damage_all.json
python scripts/prepare_coco_splits.py --coco data/work/hitl_damage_all.json --images "data/raw/hitl_car_damages/Car parts dataset/File1/img" \
       --out data/hitl_damage_v1 --group-key group --link     # 652 / 81 / 81 images
python scripts/finetune_rfdetr.py --dataset-dir data/hitl_damage_v1 --size nano --name rfdetr_seg_n --epochs 6 --device cpu
python scripts/eval_model.py --dataset-dir data/hitl_damage_v1 --split test --models abdullahg7_v8s_seg rfdetr_seg_n
```

Held-out test (81 images, 917 instances), `rfdetr_seg_n` = RF-DETR-Seg-nano, 6 epochs, ~20 min on a laptop CPU:

| model | mask AP | mask AP50 | box AP | class-agnostic AP50 |
|---|---|---|---|---|
| abdullahg7_v8s_seg (community, zero-shot) | 0.004 | 0.011 | 0.005 | 0.024 |
| rfdetr_seg_n (fine-tuned on the sample set) | 0.045 | 0.084 | 0.054 | 0.079 |

Read this carefully: the baseline is zero-shot on a different label set and photo style, so the gap shows the value of in-domain
training, not that RF-DETR beats YOLO. Absolute numbers are low because this is a nano model after 6 CPU epochs (validation mAP50
was still rising each epoch: 0.5 -> 1.4 -> 2.5 -> 4.1 -> 5.1 -> 6.0%). Visually it already localises crushed hoods/bumpers, side dents and missing
grille areas on wide shots (outputs/demo_rf/montage.jpg) but emits overlapping duplicate regions and some wrong `broken_part` labels.
There are no damage-free photos in this set, so the false-alarm rate is not measured. Real numbers need a GPU run (`--size medium`,
50+ epochs) and the client's data.

Operational lessons: on Apple Silicon do **not** train on MPS (wired memory grew to ~15 GB and the machine thrashed); `--device cpu`
now forces CPU (Lightning's accelerator used to stay on "auto"). A 12 GB-free-memory watchdog is a good idea for any long local run.

## Stronger local run (CPU, plugged in) with live progress

```bash
caffeinate -i python scripts/finetune_rfdetr.py --dataset-dir data/hitl_damage_v1 --size small --name rfdetr_seg_s \
    --epochs 30 --batch-size 4 --grad-accum 4 --device cpu --num-workers 4 --early-stopping    # tqdm bar is on by default (--progress none to silence)
python scripts/training_status.py outputs/train/rfdetr_seg_s --total-epochs 30 --watch 30      # second terminal: per-epoch table + ETA
```
Measured on this Mac (CPU, full 652 train images, rough): nano ~8 min/epoch, small ~11, medium ~17 (+~1-2 min validation each).

## Threshold calibration and duplicate suppression

- `eval_model.py` now picks per-class thresholds on the **valid** split (`--calibrate-split`, default) and scores **test** with them frozen,
  so the printed `class_conf` block is honest. `--calibrate-split none` restores tuning on the scored split (optimistic; don't report it).
- `cardamage/postprocess.py` merges same-label overlapping detections (mask IoU >= 0.5 or >= 85% contained in a higher-confidence one;
  `dedupe_iou` / `dedupe_containment` in `Settings`, 0 disables). RF-DETR is NMS-free, so this matters for it. Eval applies the same step
  (`--no-dedupe` to compare). Measured on rfdetr_seg_s: metrics are essentially unchanged (mask AP50 15.9 -> 15.5%), the demo shows ~22% fewer
  regions (483 -> 378, together with class_conf), i.e. cleaner output rather than higher accuracy.
- Frozen thresholds cost the rare classes dearly (crack F1 0.38 -> 0.11, flaking 0.04 -> 0.00): the earlier numbers were inflated and
  classes with < ~50 valid instances cannot be calibrated at all.

## Car-region filter (optional, off by default)

The damage model never saw a damage-free photo, so on street scenes it labelled pedestrians as `paint_damage` (0.7-0.9). `--vehicle-filter`
(or `CARDAMAGE_VEHICLE_FILTER=1`, also for the website) runs the COCO-pretrained RF-DETR-Seg once per image (`cardamage/vehicle.py`, same safe
loader, ~0.5 s/image on CPU) and drops damage that lies on a person and not on a vehicle. The result JSON then carries `vehicle_model`
(schema 0.4.0, additive). Measured with `scripts/measure_vehicle_filter.py` (v2 model):

- 31 street/accident samples: 135 -> 111 regions; all 24 removed were `paint_damage` on pedestrians (checked in `outputs/vehicle_filter_people/samples_removed.jpg`); no other class touched.
- v2 test split (81 close-up photos): 0 removed, mask AP50 unchanged (0.178; every class identical).
- Not shipped: an "outside any vehicle" rule (`vehicle_min_overlap` > 0). It removed 13 more regions on the samples, mostly real damage on crushed
  cars that COCO does not recognise as cars. Treat any such rule as unsafe for the wrecks that matter most.
- Limits: only people are filtered; signs, shadows or road marks are not. The filter hides the symptom of having no negatives in training; it is
  not a substitute for adding damage-free and street-scene photos.

## Hard negatives (damage-free photos) and the false-alarm rate

`scripts/openimages_negatives.py` (select + download) -> `scripts/screen_negatives.py` (run a model, write review montages) -> `scripts/build_negatives.py`
(plan + build) turn Open Images V7 validation photos (per-image licence CC BY 2.0; attribution in `ATTRIBUTION.csv`) into **images with no
annotations**: `data/hitl_damage_v4` = v3 + 150 train / 30 valid negatives (v3 test unchanged), and `data/neg_test` = 60 held-out negatives.
Reviewed by eye: 3 damaged/render images and 11 non-exterior images were dropped; train negatives outside the reviewed set may still hide a few damaged cars.

```bash
python scripts/false_alarm_rate.py --dataset-dir data/neg_test --model rfdetr_seg_s_v2       # share of damage-free photos flagged, +/- person filter
python scripts/finetune_rfdetr.py --dataset-dir data/hitl_damage_v4 --size small --name rfdetr_seg_s_v4 --epochs 25 --batch-size 4 --grad-accum 4 --device cpu --num-workers 4 --early-stopping
python scripts/false_alarm_rate.py --dataset-dir data/neg_test --model rfdetr_seg_s_v4 --model-dir rfdetr_seg_s_v4=models/rfdetr_seg_s_v4
```

Baseline, v2 on `neg_test` (60 images): flagged at conf 0.30 / 0.50 / 0.70 = 87% / 65% / 38%; with `--vehicle-filter` 78% / 35% / 10%.
The negatives are car-show, vintage and street photos, not claims photos: they teach "people/bikes/healthy cars are not damage" only.

## GPU run

`deploy/` has a Dockerfile, entrypoint and Kubernetes Job/PVC manifests for fine-tuning on the GPU cluster: see `deploy/README.md` (image build, data upload, run, resume, what is untested).

## Read before using in production

The demo weights are **research-only / AGPL** and are visibly noisy on wide accident shots (false `tire_flat`/`glass_shatter`, missed crushed fronts). Prod mode runs but logs a warning for them. See `docs/MODEL_SELECTION.md`.

## Add a model

```bash
python scripts/safe_convert.py path/to/community.pt models/<name>   # never load the .pt directly
# then add an entry to models.yaml (label -> taxonomy, licence) and run with --damage-model <name>
```

Tests: `python -m pytest -q tests`.
