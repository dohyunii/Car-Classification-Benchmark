# Final car classifier: unchanged winning I6 ensemble

[Open the final notebook in Colab](https://colab.research.google.com/github/dohyunii/Car-Classification-Benchmark/blob/main/Final_Car_Classifier.ipynb).

Select **Runtime → Change runtime type → T4 GPU**, enable notebook access to the Colab secret `KAGGLE_API_TOKEN`, and choose **Runtime → Run all**. Authorize Drive. The successful source paths are prefilled:

- I6 winner: `student_outputs_I6_LinearBlend/659a3f761309/`.
- Original F0 summary: `student_outputs_F0_Ensemble/20261108T091250Z_4ed04fc8/ensemble_summary.json`.
- Original B336/L336 feature caches are found automatically under the existing `student_outputs_I*/cache_336/` folders.

The first run builds **`MyDrive/Car-Classification-Benchmark/final_model_I6/car_ensemble.pt`**. It requires I6's complete fifteen checkpoints, `summary.json`, `probabilities.npz`, scored CSV and the source feature caches. Keep the original experiment folders as backups. Later runs load the single bundle directly; they do not need those experiment folders. The final notebook contains all implementation code and does not need Python helper uploads.

## What stays identical

Three frozen official DINOv2 backbones at pinned revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, original FP32 heads, training-derived feature statistics, deterministic BICUBIC resize/center crop and ImageNet normalization. No layers are fused, classifiers removed, weights retrained or temperatures changed.

```
S = mean of the original S224 MLP heads (seeds42,123,2026)
B = .75 * mean of B336 MLP heads + .25 * mean of B336 linear heads
L = .75 * mean of L336 MLP heads + .25 * mean of L336 linear heads
probabilities = (S + B + L) / 3
```

Every B/L head group uses seeds42,123,2026 equally. This is the same fifteen-head ensemble reported to score0.98376, packaged for simpler use. It is not a smaller or distilled model. Fresh-image inference still uses three backbones sequentially, sharing each backbone's extracted features across its heads.

## One artifact and one prediction call

The bundle contains the original head state dictionaries, per-head normalization, class mapping, pretrained backbone fingerprints and transforms, fixed blend configuration, provenance hashes, reference image hashes/IDs and a small reference feature cache. Full pretrained backbone weights are not embedded. They are downloaded from official sources on demand only for fresh-image inference.

The notebook automatically calls:

```python
submission, verification = predict_submission(
    BUNDLE_PATH, DATA_ROOT, STORAGE_ROOT / 'final_model_I6/predictions'
)
```

For the original competition test images, the bundle reuses the saved B/L frozen features and I6's saved S branch mean. B/L heads run again and the final probability blend is recomputed. It does not simply copy the final reference CSV or return its labels. Image content/order, packaged tensors, source checkpoint hashes, final probability agreement and all700 predicted labels are checked. This avoids another lengthy backbone extraction. Validation scores do not select or change anything.

Only after every label matches does a new `final_model_I6/predictions/<uuid>/` receive:

- **`submission.csv`** — identical image-ID/label pairs to the scored I6 submission.
- `verification.json` — matching count, changed count, bundle hash and prediction mode.

The model directory also has `ensemble_config.json` for inspection and `bundle_sha256.txt`. The `.pt` bundle itself is sufficient for loading; these two companion files are not required. Do not change the configuration to force a match. Any mismatch stops export, and original source files are not overwritten.

For genuinely new images, use the same artifact:

```python
model = CarEnsemble(BUNDLE_PATH)
probabilities = model.predict(image_paths)
labels = probabilities.argmax(1)
```

This runs all three official frozen backbones with their recorded weight fingerprints and original preprocessing. Original heads reuse each backbone's feature vector. There is no TTA or training. GPU batches default to8 for S/B and2 for L. For a full fresh-inference check of the original test images, `predict_submission(..., fresh_image_inference=True)` also requires all700 labels to match before export. It is optional and slower; finite-precision/library differences may cause a strict gate failure, which should be diagnosed rather than bypassed.

## Verification status

Targeted synthetic tests exercise packaging fifteen source heads, source-hash rejection, reference replay, artifact portability after removing all source experiment/cache files, leading-zero sample IDs, changed-image rejection, changed-head/reference mismatch rejection, fresh inference through three fake backbones, and notebook schema/self-contained/no-training checks. They do not execute the user's real checkpoints or independently measure Kaggle accuracy. The actual bundle and all700-label check are built in the user's Colab run, where the source files live. Identical submitted labels preserve the score on the same evaluation; no new Kaggle measurement is claimed and no automatic submission happens.
