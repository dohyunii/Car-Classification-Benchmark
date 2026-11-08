# Cars196 external-data experiment: executed results

Goal: improve measured accuracy toward99% (198/200 validation images). Additional labeledCars196 data was authorized by the user. This experiment did not improve accuracy, so the expanded-data heads are not selected as replacements.

## Source and audit

Pinned public Cars196 mirrors supplied16185 images. A label-blind image audit compared them to all700training,200validation and700test images. No source labels were used to recover competition test labels. All1600 course images had near-match candidates; the mirror images had no byte-identical or decoded-RGB-identical matches, consistent with re-encoding. This is similarity evidence, not a proof of exact identity.

Conservative exclusions:8342 heldout-overlap/ambiguous images,217 training duplicates,37 redundant external-component members.7589 source images passed the screen. Only then were source labels interpreted for those survivors. Of these,7530 belonged to other classes; **59 belonged to the20 target classes**. Full class names, including body style/year, were matched exactly.

Train-only calibration:180/180 tested recompressed/resized/cropped/flipped positives detected;0/2000 sampled different-class pairs matched. This is not proof that every possible transformed duplicate is detectable. Ambiguous candidates were excluded, and the full screening thresholds/software/source hashes are saved.

## Training and comparison

Combined training used759 distinct images and3795 feature rows. All3500 original course feature rows were reused verbatim; only295 additional feature rows were extracted. The frozen DINOv2-S timm backbone,224pxP3 preprocessing,E4 head,augmentationseed123,headseeds42/123/2026,AdamWlr/decay.001/.001,batch64,max35epochs andpatience7 stayed fixed. Each head was initialized afresh. Normalization was refitted only on combined training rows; validation remained200images.

| Head seed | Original700 accuracy | Expanded759 accuracy | Original macro-F1 | Expanded macro-F1 |
|---|---:|---:|---:|---:|
| 42 | 97.5% | 97.0% | 0.9749 | 0.9700 |
| 123 | 96.0% | 96.5% | 0.9598 | 0.9646 |
| 2026 | 97.5% | 97.0% | 0.9749 | 0.9699 |

| Training set | Mean accuracy ± sample SD | Mean macro-F1 ± sample SD |
|---|---:|---:|
| Original700 | 97.00% ± 0.87pp | 0.9699 ± 0.0087 |
| Expanded759 | 96.83% ± 0.29pp | 0.9682 ± 0.0031 |

Expanded training changed the three-seed mean from97.00% to96.83% (−0.17pp). Fixed headseed2026 changed from97.5% to97.0% (one fewer correct image). These small differences are not evidence of a general benefit or harm on future data. Your historical98.0% F0 checkpoint remains the stronger reported result; it is not present locally and was not reverified.

All starting models and course files were preserved. No test inference or Kaggle submission was performed for these expanded-data models. No additional optimization against validation scores was run.

## Artifacts

- Audit: `outputs/Cars196_audit/2772c5434f1a`
- Approved manifest and class mapping: `outputs/Cars196_manifest/86e6a2008d95`
- Training runs/checkpoints/statistics: `outputs/Cars196_training/3db6eb7c0166`
- Easy comparison: `outputs/Cars196_training/3db6eb7c0166/metrics/easy_comparison.csv`

The7530 surviving source images in other classes were not fed into a20-class head. Using them would require a separately designed supervised transfer stage, rather than silently reassigning their labels.
