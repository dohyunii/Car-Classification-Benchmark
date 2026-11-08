# J3: one controlled CLS + mean-patch representation candidate

J3 keeps J0's frozen pretrained DINOv2-S/14 backbone, 224px P3 preprocessing, fixed augmentation seed 123 and E4 training recipe. It changes only the extracted representation and the head input width. The purpose is to test whether spatial patch information helps fine-grained car classification. A larger representation is not assumed to improve accuracy.

No external labeled images, pseudo-labels, validation/test training, unrestricted hyperparameter search or automatic Kaggle submission is used. This is the final predeclared representation candidate after J0/J1/J2.

## Representation and exact paired control

The timm backbone's `forward_features(images)` returns the final-normalized prefix and patch tokens. For each image J3 concatenates:

```python
cls = tokens[:, 0]
mean_patch = tokens[:, model.num_prefix_tokens:].mean(dim=1)
features = torch.cat([cls, mean_patch], dim=1)
```

Every prefix token is excluded from the patch average. The same pretrained weights remain frozen in evaluation mode, with inference mode enabled during extraction. The resulting representation is **768-D**: 384 CLS dimensions followed by 384 mean-patch dimensions. This follows the common DINOv2 CLS-plus-average-patch representation idea; it does not modify the backbone or train any additional image model.

The E4 head becomes `Linear(768,128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128,20)`. Only its input width changes from J0's 384-D E4; hidden width, normalization, activation and dropout are unchanged. Every head starts fresh; no 384-D head weights are reused.

J3 requires the completed J0 **224px** raw feature cache, regardless of which resolution J0 selected. It verifies the original model weight fingerprint, dataset manifest, mapping, transforms, augmentation seed, view layout and extraction batch size. Extraction matches J0's DataLoader order and global augmentation RNG sequence: one deterministic original pass, four augmented passes, then one deterministic validation pass; zero workers and no shuffling. Expected tensors are `X_train=(3500,768)`, `y_train=(3500,)`, `X_val=(200,768)`.

Before any head training, the first 384 columns of **every** J3 training and validation row must match the corresponding J0 CLS features with **rtol=0, atol=1e-5**. This verifies that the original images and four augmented views are paired rather than introducing different random crops alongside pooling. A mismatch stops execution. Diagnose source weights, software versions, paths/order, extraction batch size and augmentation RNG; do not bypass or loosen the prefix check to obtain a passing result.

Cache signatures include pooling definition, feature width, actual weights, model/source, P3 transforms, augmentation seed/view count, class mapping, ordered content-hashed manifest, source J0 cache ID, package versions, device and batch size. Compatible caches are shared across UUID runs; metadata and paired-prefix checks are repeated before reuse. Classifier seeds do not affect the feature cache.

## Training and predeclared model selection

Train three fresh E4 heads with seeds **[42,123,2026]**, all sharing the same augmentation-seed-123 embeddings. Each run resets the classifier seed and independently seeds its shuffled loader. Standardization uses the complete augmented training set only: per-feature mean and population SD, clamped at 1e-6. Validation data never fits these statistics or updates the head.

The fixed recipe is AdamW lr **0.001**, weight decay **0.001**, embedding batch **64**, maximum **35 epochs**, early-stopping patience **7** and cross-entropy. Select the first epoch with strictly highest validation accuracy; an accuracy tie does not reset patience or select a new checkpoint on macro-F1.

Report individual seed results and **mean ± sample SD** for accuracy and macro-F1. These seed summaries describe classifier-initialization variability on the same validation split. They do not replace a separate holdout or establish general superiority.

Use **head seed 2026** as the single-model pooling candidate. Retain that candidate only if its validation accuracy strictly exceeds the verified J0 baseline **97.5% (195/200)**. Otherwise keep the baseline. Do not select the luckiest individual seed after seeing results.

Exactly one additional candidate is predeclared: the **equal probability average of all three J3 heads**. Compute each head's softmax probabilities using its best checkpoint and saved training-derived statistics, then take their arithmetic mean. There are no learned weights, temperature tuning, subset searches or combinations with other experiments.

Retain the three-head ensemble only if its validation accuracy **strictly exceeds both** the fixed-2026 J3 candidate and the 97.5% J0 baseline. Ties keep fewer heads. Macro-F1 is a secondary reported metric and does not override these strict accuracy rules. With only 200 validation images, one additional correct prediction is 0.5 percentage points; a small gain remains provisional.

## Execute

From the repository using the same PyTorch/TorchVision/timm environment as J0:

```bash
python pooled_accuracy_experiment.py \
  --j0-run outputs/J0/c8aab8360ba2 \
  --data-root /path/to/the/extracted/course/dataset \
  --output-root outputs/J3 \
  --batch-size 8 \
  --threads 4
```

Use your actual completed J0 run directory and identical extracted course dataset. The J0 cache must be preserved in `<J0-run>/cache/` or shared `<J0-run-parent>/cache/`. The default extraction batch size is 8 and must match J0's cache metadata. `--force` explicitly regenerates this J3 representation while still enforcing the paired-prefix control.

A new UUID directory under `--output-root` protects all previous artifacts. Shared J3 caches live in `<output-root>/cache/`. Dependencies are the same as J0: PyTorch, TorchVision, timm, NumPy, pandas, Pillow, scikit-learn and matplotlib, plus the repository's `accuracy_experiments.py` and `preprocessing_benchmark.py` helpers. The backbone weights may need their official download if not cached. No new labeled dataset or external API key is required for extracted local data.

## Artifacts and test inference

Each head saves a distinct full E4 state, normalization statistics, mapping and feature metadata in `checkpoints/`; its metric JSON, history CSV and aligned per-image validation predictions live in `metrics/`. Checkpoint names include `J3_S14_224_CLSmeanpatch_aug123_head<seed>`.

Each `*_validation_probabilities.npz` contains validation paths, targets, class names, logits and softmax probabilities in the same order. These aligned arrays support the single fixed equal-probability three-head candidate; alignment and normalization must be checked before averaging. `J3_runs.csv`, `J3_seed_summary.csv` and `J3_summary.json` record completed runs and sample-SD summaries. No fabricated score is bundled with the implementation.

**The training script currently performs no test inference and no Kaggle upload.** A separate exporter will generate a local submission only after the predeclared comparison identifies a measured accuracy gain. It must restore the retained checkpoint(s), use the same frozen pooled features, fixed saved statistics and original sample ID order, and perform no test-driven model selection. The actual Kaggle score remains unknown until a separate manual submission is evaluated.

## Lightweight verification

```bash
python -m unittest discover -s tests -p test_pooled_accuracy_experiment.py -v
```

Three synthetic tests verify patch averaging excludes all prefix tokens, 768-D output, exact J0-prefix pairing across all view blocks, label alignment, distinct augmented views, cache reuse, hard failure for mismatched prefixes/weights/batch size, train-only statistics, fresh head seeds, strict checkpoint restoration and aligned normalized probability/logit exports. No real pretrained extraction or held-out validation run is performed by these tests.


## Executed result

Actual CPU run passed full paired-prefix verification. Seeds42/123/2026:96.5%,96.0%,96.5%; fixed ensemble96.0%. Both candidates rejected; no J3 test predictions generated. See `ACCURACY_IMPROVEMENT_RESULTS.md` for the full comparison.
