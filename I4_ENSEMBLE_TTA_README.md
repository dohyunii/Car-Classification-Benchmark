# I4: full-ensemble horizontal-flip TTA

[Open I4 in Colab](https://colab.research.google.com/github/dohyunii/Car-Classification-Benchmark/blob/main/I4_Ensemble_FlipTTA.ipynb). Select **Runtime → Change runtime type → T4 GPU**, enable the Colab secret `KAGGLE_API_TOKEN`, and run all cells. Authorize Drive. The original successful F0, I1 and I2 folders are prefilled; keep them unchanged unless moved. I3 fine-tuned checkpoints are not used.

The experiment averages original/flip **probabilities**, then preserves the equal S224/B336/L336 backbone weights. F0's three heads remain equal within the S branch. All heads retain their own training-only feature mean/std. No fitting, augmentation search, branch weight search, pseudo-labeling or model updates occur. Validation is diagnostic only: a single fixed candidate exports even if validation decreases. Training data are inspected only for source integrity.

Original B336 and L336 validation/test probabilities reuse their existing feature caches. Original S predictions are replayed. Source checkpoint hashes, dataset content, cache metadata/signatures, dimensions, labels and class mapping must match; every label in the original 0.97959 CSV must reproduce before flip inference starts. Each official backbone loads at the pinned repository revision and matches its source weight fingerprint. Head checkpoint files are rechecked before export.

TTA flips the deterministic normalized center-crop tensor horizontally: S uses resize256/center224; B/L use resize384/center336, BICUBIC and ImageNet normalization. One backbone loads at a time. Default inference batches are 8 for S/B and 2 for L. If GPU memory is insufficient, restart the runtime and reduce the relevant batch size (L to 1). The L flip pass still processes 200 validation and 700 test images and can be slow on T4, but does not re-extract the original 3500 training views. Logs show elapsed and estimated remaining time every ten batches; early estimates may fluctuate.

Completed flip probabilities are cached per branch/split under `student_outputs_I4_Ensemble_FlipTTA/flip_cache/`. Cache identity includes checkpoint hashes, backbone weights, dataset manifest, mapping, test order, transform, software version, device and batch size. Metadata and finite normalized probability tensors are validated. Reruns reuse completed compatible splits; an interrupted split restarts. Corrupt caches stop rather than silently produce a candidate. A changed batch size creates a separate cache.

Every run creates `student_outputs_I4_Ensemble_FlipTTA/<uuid>/` containing:

- **`I4_S_B336_L336_flip_TTA_submission.csv`** — manually submit this one candidate.
- `CurrentBest_097959_control.csv` — labels verified equal to the winning original CSV.
- `test_prediction_changes.csv` — original/TTA labels and change flags, without test labels.
- `probabilities.npz` — original/TTA validation and test probabilities, plus validation targets.
- `summary.json` — validation comparison, corrected/regressed counts, changed test predictions and provenance.

The last cell prints the exact candidate path. Download it from Drive or the Colab Files panel, then use **Submit Prediction** on the course competition. Suggested description: `I4: equal S224 B336 L336, original + horizontal-flip probability TTA`. Preserve the whole output directory and your original 0.97959 submission. Equal Kaggle scores do not prove identical predictions; the change CSV reports the actual disagreement count. No automatic Kaggle submission occurs and no accuracy gain is guaranteed.

Synthetic tests cover horizontal orientation, per-head statistics, cache hits and invalid values, unchanged sources, fixed probability averaging, ordered exports, end-to-end candidate export despite worse validation, and self-contained notebook definitions. They do not measure real DINOv2/GPU performance; inference runs in Colab.
