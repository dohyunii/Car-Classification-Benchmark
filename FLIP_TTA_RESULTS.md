# Fixed flip-TTA: measured local results

Model: available J0 DINOv2-S/14 timm checkpoint, head seed2026. This is **not the original F0 model**; its Colab checkpoint remains unavailable locally. The original model was independently reproduced at195/200 correct before TTA. Original center-crop and horizontal-flip **probabilities** were averaged equally, with no fitted weights, training, normalization updates or test-label use.

| Inference | Validation accuracy | Macro-F1 | Correct/200 |
|---|---:|---:|---:|
| Original | 97.5% | 0.9749 | 195 |
| Original + flip TTA | 97.0% | 0.9698 | 194 |

Only3 validation predictions changed:1 corrected and2 regressed. Exact paired McNemar/binomial p=1.0; the small sample provides little evidence about the direction of a population difference. This is not an equivalence test or evidence of a Kaggle improvement. A slight validation loss can coexist with a test gain or loss under finite samples/distribution shift; neither outcome is guaranteed.

Both700-row test CSVs were generated in original sample order. They differ on11 predicted labels. Test ground truth was never used. The original CSV was verified identical to the previously generated J0 baseline CSV. Nothing was submitted to Kaggle. Compare J0 original against J0 TTA to isolate the TTA effect; comparing directly against an older F0 leaderboard score confounds models.

Artifacts: `outputs/Flip_TTA/8e52a200b93f/` contains the two candidateCSV files, paired validation predictions, probability arrays, checkpoint checksum and summary JSON. Four focused synthetic tests passed: probability averaging, eval behavior, labels ignored, paired statistics, sample ordering/overwrite protection and the portable F0 reproduction gate.

For actual F0, paste/run `F0_flip_tta_colab_cell.py` inside the existing verified F0 notebook and call `run_verified_f0_flip_tta(...)` as documented in `FLIP_TTA_README.md`. It uses that model's own backbone/head/statistics and requires reproduction of its starting accuracy before test prediction. No repository helper imports are required for that standalone cell.
