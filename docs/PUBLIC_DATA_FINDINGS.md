# Damage detection on public data: what it supports and what needs client data

Model: RF-DETR-Seg small (Apache-2.0), fine-tuned on the Humans in the Loop CC0 sample set (814 photos), taxonomy v2
(6 classes: dent, paint_damage, corrosion, crack, broken_part, missing_part). Held-out test: 81 photos, split by
near-duplicate group so none leak from training. Per-class thresholds were chosen on the validation split and frozen
before scoring test. 95% intervals come from resampling test images 200 times.

## Bottom line

Public data does **not** show that the model performs well enough overall. It supports large structural damage
(`broken_part`, borderline `missing_part`). Dents, paint damage and any-damage detection fall short of the acceptance bar
we set beforehand, and that gap is the evidence for needing the client's labelled photos.

## Result against the pre-set acceptance bar

Metric: mask AP50 (predicted region overlaps the labelled region by IoU >= 0.5). Bar for each major class: lower 95% bound >= 0.30.

| criterion | result | 95% interval | verdict |
|---|---|---|---|
| broken_part AP50 | 0.385 | 0.31 - 0.48 | pass (just) |
| missing_part AP50 | 0.312 | 0.20 - 0.44 | fail (point estimate is above 0.30) |
| dent AP50 | 0.119 | 0.08 - 0.19 | fail |
| paint_damage AP50 | 0.046 | 0.02 - 0.10 | fail |
| any damage (class-agnostic) AP50, bar >= 0.50 | 0.221 | 0.19 - 0.25 | fail |
| any-damage recall, bar >= 0.95 | 1.00 | | not informative: the model shows ~4.7 regions per photo, and the test set has no damage-free photos |
| crack (n=8), corrosion (n=12) | 0.126, 0.077 | very wide | reported only, too few to judge |

Overall mask AP 0.098 (0.078 - 0.152), mask AP50 0.178 (0.151 - 0.258).

## Why it falls short (diagnostics, not a new bar)

- **Extent disagreement explains part of it.** At a looser overlap (IoU 0.3), dent AP rises 0.12 -> 0.29, paint_damage
  0.05 -> 0.20, any-damage 0.22 -> 0.37. The model often finds the damage but outlines it differently from annotators.
  Even so, none reach the bar.
- **Post-processing is not the fix.** About half of the shown regions overlap another region, mostly across labels (e.g. a
  dent inside a broken part), and removing them lowers AP. Stricter duplicate suppression changes nothing.
- **Merging classes helped size, not accuracy.** v2 raised the median damage size at model input from ~5 px to 18 px, yet
  paint_damage stays low. Diffuse regions with soft edges are hard to match; annotation consistency is likely a factor
  (not yet measured).
- **Training was not exhausted but is slowing.** Validation mask mAP50 reached 0.166 at epoch 25 of 25 (0.156 at epoch 20).
  Test (0.178) agrees with validation, so no sign of overfitting. More epochs would add points, not tens of points.
- v1 (8 classes) and v2 numbers are not comparable: the task and labels changed. v1 small run: mask AP50 0.155 on its test split.

## Limits of this evidence

- One public dataset, 814 photos; test is 81 photos. Intervals are wide for every class.
- Photo origin is undisclosed by the dataset creator, so weights trained on it are marked `production_ok: false`.
- **False alarms are bad (measured later, on separate data).** The training set has no damage-free photos. On 60 reviewed damage-free
  photos (Open Images, CC BY; healthy cars and street scenes), the v2 model flags 87% of images at its default threshold (95% CI 76-93%),
  65% at 0.5 and 38% at 0.7; the person filter lowers these to 78% / 35% / 10%. Pedestrians were labelled `paint_damage` (up to 0.9)
  and bikes/karts `broken_part`. Healthy cars alone: 33/40 flagged at 0.30 even with the filter. The any-damage "recall 1.00" above is
  therefore meaningless: v2 flags nearly everything. Fix attempted in the v4 dataset (150 negatives in training); result pending.
- A small model at 384 px trained on a laptop CPU. A larger model or resolution may score higher; not tried.
- This shows feasibility on public photos. It is not a prediction of accuracy on the client's photos.

## What needs client data

1. **Photos from the real claims flow**, including damage-free vehicles (to measure false alarms) and the camera/angle
   mix the pipeline will actually see.
2. **The client's damage classes**, so we stop guessing a taxonomy. Our v2 classes are a placeholder based on common
   assessment practice.
3. **Consistent labelling for dent and paint damage**, using the rules in `docs/ANNOTATION_GUIDELINES.md`, with a double-labelled
   subset to measure annotator agreement.
4. **Enough examples per rare class** (crack, corrosion): the validator warns below 50 training instances per class;
   we had 59 and 117.
5. **Legal clearance** of the training data and sign-off on the licence of any weights shipped.
6. **Parts labels** (CC0 parts photos are ready) for "which panel" and vehicle-relative area, and the JEV input schema agreed with that team.

## Reproduce

```
python scripts/eval_model.py --dataset-dir data/hitl_damage_v2 --split test --models rfdetr_seg_s_v2 --bootstrap 200 --out outputs/eval_v2 --device cpu
python scripts/diagnose_model.py --dataset-dir data/hitl_damage_v2 --split test --model rfdetr_seg_s_v2 --device cpu
```
