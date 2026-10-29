# Damage taxonomy v2 and annotation guidelines

Status: working draft for the public-data proof. The client has not named damage classes, so this follows the categories commonly
used in claims triage. **Confirm with the client's assessors before labelling their photos.** Machine-readable form: `configs/taxonomy_v2.yaml`.

## Why v2 (evidence, from `scripts/analyze_labels.py` on the train+valid splits, 384 px input)

| v1 class | instances | median area (% of image) | share narrower than 6 px at input | what the model did (valid/test AP50) |
|---|---|---|---|---|
| scratch | 2890 | 0.06 | 43% | 1.8% |
| paint_chip | 1193 | 0.04 | 60% | 5.4% |
| flaking | 302 | 0.12 | 29% (median 10 px) | 0.3% |
| corrosion | 248 | 0.11 | 31% (median 10 px) | 5.4% |
| dent / broken_part / missing_part | 1493 / 1348 / 568 | 0.5 - 1.0 | 2 - 5% | 14% / 34% / 41% |

The weak classes are weak because the marks are a few pixels wide (RF-DETR works on 12 px patches): a resolution ceiling, not a labelling or
training-length problem. Scratch, chip and flaking are also visually indistinguishable at that size and are repaired the same way (refinish).
Overlap in the annotations is modest (small marks mostly sit *inside* dents / broken parts, e.g. 27% of scratches), so no class was merged because of mislabelling.

## Classes

| Class | Label it when | Do NOT label |
|---|---|---|
| `dent` | Panel deformed without a break in the material (surface pushed in/out). One instance per visually separate dent. | Creases that run into a break (-> `crack`/`broken_part`). |
| `paint_damage` | Damage limited to the paint/coating: scratches, scuffs, paint chips, flaking or peeling paint. **One region per cluster** of such marks (see rules). | Marks on glass or lamps; rust (-> `corrosion`). |
| `corrosion` | Visible rust/oxidation, including rust bleeding through paint. Kept separate because it is usually pre-existing wear, which matters for the coverage decision. | Dirt, stains, brake dust. |
| `crack` | A line fracture in a rigid part (bumper, lamp lens, glass, panel) where the part is still in one piece. | Scratches in paint. |
| `broken_part` | A part that is fractured, deformed beyond a dent, detached in pieces or hanging, but still in the picture. | A part that is simply absent (-> `missing_part`). |
| `missing_part` | The location where a part should be and is gone (grille, lamp, trim, mirror, bumper section). Label the empty location. | Parts that are present but damaged. |

Out of scope for now (add when the client asks): tyre damage, glass shatter, lamp broken as separate classes. The schema already reserves
`glass_shatter`, `lamp_broken`, `tire_flat`.

## Rules (adapted from the CarDD annotation protocol and common claims practice)

1. **Region-level, not mark-level, for surface damage.** Same-class marks closer than ~3% of the image's long side are ONE region
   (`region_merge` in the spec reproduces this automatically for the existing labels). Assessors scope a repair area, not every scratch.
2. **Mixed damage priority.** If a region has several damage types, the most severe wins where they coincide:
   `missing_part > broken_part > crack > dent > corrosion > paint_damage`. Paint damage *inside* a dent is still labelled; do not delete it - overlap between classes is allowed.
3. **Damage across panels.** Split at the panel boundary: one instance per panel. (Needs the parts model to automate; until then split by eye.)
4. **Outline:** tight polygon around the damaged area, not the whole panel. Minimum size: skip marks under ~0.03% of the image.
5. **Do label pre-existing damage** (rust, old dents) - the class tells JEV what it is. Do not guess whether it is from the current accident.
6. **Photos:** include undamaged vehicles (>=10%) and photos where damage is only on the far side - without them false-alarm rates cannot be measured.
7. **QA:** two annotators on 10% of photos; report agreement (mask IoU >= 0.5 match rate per class). A class below ~0.6 agreement needs a clearer definition before more labels are bought.

## What v2 does and does not claim

v2 numbers are **not comparable** to v1 numbers: classes are coarser and surface damage is region-level, for ground truth as well as predictions.
Report both and say so. The change makes the task closer to how a repair is scoped; it does not make the model better at seeing a 5 px scratch.

## Proposed acceptance bar for "good enough on public data" (decide before looking at v2 test results; adjust to the client's needs)

On the held-out **test** split, thresholds frozen from **valid**, with 95% bootstrap CIs (`eval_model.py --bootstrap 200`):

1. Each major class (dent, paint_damage, broken_part, missing_part): mask AP50 **lower CI bound** >= 0.30.
2. Class-agnostic ("is there damage here?") mask AP50 point estimate >= 0.50.
3. Any-damage image recall >= 0.95 (already 0.99 on v1).
4. False-alarm rate on damage-free photos. Originally not testable (no such photos); now measured on `data/neg_test` (60 reviewed
   Open Images photos, `scripts/false_alarm_rate.py`). **Proposed, set after seeing the v2 baseline (87% of images flagged at conf 0.30,
   65% at 0.5, 38% at 0.7) and before any v4 result:** at the production threshold, at most 10% of damage-free photos may be flagged
   (upper 95% bound reported alongside; n=60 gives a wide interval). This is a tightening motivated by a visibly bad baseline, not a
   pre-registered bar, so say so when reporting. It tests car shows/street scenes, not claims photos.
5. crack and corrosion (< ~15 test instances): report, do not gate - too few to measure.
This is a proof of *feasibility on public data*, not a performance guarantee on client photos (different cameras, lighting, damage mix).
