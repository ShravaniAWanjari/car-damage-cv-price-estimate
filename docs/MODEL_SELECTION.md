# Model selection: car damage detection / segmentation

**Status:** demo model chosen; production architecture recommended; production weights **do not exist yet** (see "The real blocker").

## TL;DR

| | Choice | Why |
|---|---|---|
| **Demo (now)** | `abdullahg7/cardd-yolov8s` v2.0 (YOLOv8s-seg, 11.8M params, 6 damage classes, 1024 px) | Only candidate that is segmentation + multi-class + documented metrics (self-reported mask mAP50-95 57.0 on CarDD test) + small. Best output on the sample photos. |
| **Production architecture** | **RF-DETR-Seg (M or L)**, Apache-2.0, fine-tuned by us. Fallback: YOLO11/YOLOv8-seg **with an Ultralytics Enterprise licence** | Licence-clean, fast on GPU, trivially cheap to serve. |
| **Production weights** | Fine-tune on licensed/own claims photos | No pretrained car-damage weights I found are both good *and* usable commercially. |

## The real blocker is licensing and data, not architecture

1. **CarDD (the dataset behind nearly every public car-damage model) is research-only.** Its licence (`CarDD_license.pdf`) allows "statistical and scientific research"; "any commercial use ... including **testing commercial systems**" needs prior approval from the PIC Lab (CAS), and no redistribution to third parties. Weights trained on it (abdullahg7, harpreetsahota, probably michelgi) inherit that risk. The 4,000 images also come from Flickr/Shutterstock (per the abdullahg7 card). Some HF datasets relabel CarDD-derived data as "CC-BY-4.0" (e.g. `tugberkkalay/autodamageiq...` lists CarDD as a source) - do not trust those labels.
   - For this reason **I did not use CarDD images to evaluate**; test photos are CC/public-domain from Wikimedia Commons (`samples/SOURCES.json` has author + licence per image).
   - Option: ask PIC Lab for a commercial licence (wangxk0624@mail.ustc.edu.cn per the licence PDF).
2. **Ultralytics YOLO (v5/8/11/12) is AGPL-3.0**, including trained weights. Their licence page says an Enterprise licence is needed for closed-source, SaaS/API and *internal business tools*. A "Apache-2.0" tag on a Hugging Face YOLO fine-tune (e.g. `vineetsarpal/yolov11n-car-damage`) does not remove that. Price is quote-only.
3. **Apache-2.0 detectors exist:** RF-DETR (N/S/M/L detection, N-2XL segmentation; XL/2XL *detection* and Atto/Femto/Pico are under Roboflow's PML 1.0), RT-DETR/RT-DETRv2, D-FINE/DEIM, Mask R-CNN (Detectron2), RTMDet (MMDetection). Not legal advice - have counsel confirm.
4. **Untrusted weights are code.** Community `.pt` files are pickles. `scripts/safe_convert.py` rebuilds tensors with an allow-list unpickler and writes safetensors, so nothing from the file executes (and `picklescan` flagged one file's `getattr`; it was benign, which I verified). Keep this step in the CI path.

## Candidates found and tested

Measured here: Apple M5 CPU, native `imgsz` per model, conf 0.25, 31 CC-licensed photos, **no ground truth**. Latency is only a relative guide; it is not GPU latency.

| Model | Arch / task | Params | GFLOPs @imgsz | CPU ms/img | Images with detections (of 31) | Observed behaviour |
|---|---|---|---|---|---|---|
| abdullahg7 v2.0 | YOLOv8s seg, 6 cls, 1024 | 11.8M | 103 | 142 | 24 | Best: fine per-type masks on close/mid shots. FPs: `tire_flat` on intact wheels, `glass_shatter` on intact windshields. |
| harpreetsahota | YOLO11 seg (large), 6 cls, 1280 | 62.1M | 1194 | 950 | 15 | 8x slower, no better; produced a large spurious mask on a pole/background. |
| michelgi yolo12m-seg | YOLO12m seg, 6 cls (Dent/Scratch/Crack/Broken/Paint_chip/Wreck), 640 | 22.4M | 116 | 176 | 12 | No model card. Tends to one big "Broken" blob over the whole front end. |
| shanexf yolo11m | YOLO11m **box only**, 6 cls, 640 | 20.1M | 68 | 73 | 16 | Conservative; boxes only (no area from masks). |
| vineetsarpal yolo11n | YOLO11n **box only**, 14 panel-aware cls, 640 | 2.6M | 6.5 | 22 | 16 | Gives panel in the label (bonnet/door/bumper...) which is nice for "damage side"; reported mAP50 79.7 on its own Roboflow set; false "bonnet-dent" on a road. |
| aitools damage / parts | YOLOv8m seg, 1 cls "item" / 5 part cls | 27.2M | 105 | 98 | 20 / 25 | Damage is class-less. Parts model mislabelled a front as `rear_bumper` -> **unreliable for side**, so the parts hook is off by default. |
| tumul31 RT-DETRv2-r50 | HF RT-DETR | - | - | - | - | Checked: it is a **car-parts** detector (Bonet, Bumper, Door...), not damage. Not loaded. |
| Naiscorp r101-DC5 Mask R-CNN | Detectron2 | - (764 MB file) | - | - | - | 7 classes, "proprietary-weights" licence, no metrics. Skipped. |

Not benchmarked: `michelgi` RF-DETR-large checkpoint (no card), VLM approaches (Florence-2 / PaliGemma / Llama-vision fine-tunes found on HF) - they output text/coarse polygons and are far more GPU per image than a 12M-param segmenter, so they don't fit a per-photo inline step.

### What I saw (honest read of the sample outputs)

- Close and mid-range damage photos: usable (scratch/lamp/crack/dent masks line up with visible damage).
- Wide accident scenes: crushed fronts are often **missed** (e.g. BMW with displaced bumper, Toyota into a pole) by every candidate.
- Precision: notable false positives (`tire_flat`, `glass_shatter`, a whole-image mask on a non-car scene). Per-class thresholds, a vehicle/photo-quality gate and a fine-tune on our own data are needed before JEV should trust it.
- This is a visual check on ~24 of 31 images, not a metric. Real numbers need a labelled hold-out set of our own claim photos (mask AP, per-class recall at fixed FP rate). The self-reported CarDD figure is the only number I have, and is unverified by me.

## Performance vs cost in production

Per-photo GPU cost is small compared to the licence and data costs, so **pick the most accurate licence-clean model, not the smallest**.

Published T4, TensorRT FP16, batch 1 (RF-DETR README): Seg-N 3.4 ms (COCO 40.3), Seg-S 4.4 (43.1), **Seg-M 5.9 (45.3)**, **Seg-L 8.8 (47.1)**, Seg-XL 13.5 (48.8). (The README does not say whether AP is box or mask.)

Illustrative arithmetic (assumptions: Seg-L at 8.8 ms, 50% real utilisation incl. pre/post-processing => ~55 img/s per T4-class GPU; GPU cost X $/hr): 1M photos = 1M / 55 / 3600 = **~5 GPU-hours**. At an assumed $0.50/GPU-hr that is ~$2.5 per million photos; with your own cluster substitute your internal rate. A claim is typically tens of photos, so inference spend rounds to zero. Memory: not measured here; models of this size (tens of M params) should be comfortable on a single mid-range GPU, so I'd expect GPU sharing and autoscaling on queue depth to be enough - verify on your cluster.

Implication: choose on accuracy per labelled-data dollar. Expect a larger/higher-res model (Seg-L, 1024 px) to pay for itself in fewer false positives on thin scratches/cracks.

## Recommendation / next steps

1. **Decide licensing path** (pick one): (a) commercial licence for CarDD from PIC Lab; (b) collect/label our own claim photos (best, also fixes domain shift to our wide-shot, real-world photos); (c) Ultralytics Enterprise licence if we keep YOLO.
2. **Fine-tune RF-DETR-Seg (M and L)** - tooling is in place (decision taken): `scripts/prepare_coco_splits.py` -> `scripts/finetune_rfdetr.py` -> `scripts/eval_model.py`, `backend: rfdetr` entries `rfdetr_seg_m/_l` in `models.yaml` (see README). Verified end to end on the CC0 Humans-in-the-Loop sample (README, "Sample dataset"): nano/6 CPU epochs reached mask AP50 8.4% on its held-out split; needs the client's data and a GPU run for real numbers. Baseline to beat: `abdullahg7_v8s_seg` on the same held-out split, judged by mask AP, per-class F1 and the false-alarm rate on damage-free photos.
3. **Add gating/filters:** per-class thresholds (`class_conf` in `models.yaml`, calibrated by `eval_model.py`; implemented), a "is this a car?" check, minimum photo quality - addresses the FP patterns seen above.
4. **Damage side / vehicle-relative size:** needs a reliable car-parts (or vehicle) segmenter; none of the open ones I tried were good enough. Plan to train parts + damage jointly on our data. Then `area_ratio_vehicle` can be added to the schema.
5. **Export for serving:** ONNX/TensorRT FP16 behind Triton or a thin FastAPI worker; the `/analyze` endpoint here is the integration shape.
