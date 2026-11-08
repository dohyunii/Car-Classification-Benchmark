# J1: one bounded RBF-SVM classifier candidate

J1 tests one additional classifier family on J0's existing frozen DINOv2-S/14 features. It does not extract new features, change the backbone, search additional resolutions, train on validation/test data, use external labeled images or pseudo-labels, or upload anything to Kaggle.

J0 must finish first. J1 reads the selected resolution from its `summary.json` and uses that resolution's saved raw feature cache. The previously selected J0 checkpoint uses the fixed E4 head seed 2026; J1 does not select the luckiest E4 seed.

## Exact training-only selection procedure

1. Use only the original 700 training images and their class labels to construct `StratifiedKFold(n_splits=3, shuffle=True, random_state=123)`.
2. Interpret the cached 3,500 training rows as five view-major blocks: deterministic originals first, then four augmented passes. An original image at index `i` has sibling rows `i + view * 700`, for views 0–4.
3. For each fold, fit on **all five views of its fitting original images**. Exclude all five views of each held-out original image from fitting. Score only the held-out images' deterministic block-zero rows. Never split embedding rows independently: that would put sibling views on both sides of a fold.
4. Evaluate only the three predeclared values **C = [0.1, 1, 10]**, using `Pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale'))`. Each fit learns its scaler from that fold's fitting rows only; no globally fitted normalization is reused inside CV. `gamma='scale'` is determined by the fitting data.
5. Rank C by mean CV accuracy, then mean CV macro-F1, then smaller C for an exact tie. Fold standard deviations use sample SD; they describe training-fold variation, not a new independent test set.
6. Persist the chosen C and selection metadata before predicting the provided validation set. Refit the selected pipeline on all 3,500 training views. Its final scaler also uses training rows only.
7. Evaluate the selected pipeline on the provided 200 validation images **once**. The provided validation labels never enter C selection, scaler fitting, or SVM fitting. No SVM seed, C value, preprocessing variant or additional resolution is tuned afterward.

Fold diagnostics assert that no held-out original image or augmented sibling appears in fitting rows. All fitting label vectors follow the same view-major layout. The cache signature, model/resolution, augmentation configuration, class mapping, ordered dataset manifest, feature shapes and label alignment are checked before use.

## Execute

Use the same Python environment as J0, with NumPy, pandas, PyTorch, scikit-learn and joblib installed. No GPU is needed. From the repository:

```bash
python svm_accuracy_experiment.py \
  --j0-run outputs/J0/c8aab8360ba2 \
  --data-root /path/to/the/extracted/course/dataset \
  --output-root outputs/J1
```

`--j0-run` must point to a completed UUID run directory containing `summary.json` and the selected-resolution metric record. Replace the example with your actual run path. `--data-root` must contain the same train/validation/test files and original `sample_submission.csv` used by J0. These files are hashed to confirm identity; test images are not passed to the SVM.

By default resolution is taken from J0's final selection. An optional `--resolution 224` or `--resolution 336` is accepted only when it matches that selection. This is not an additional resolution search.

The loader finds the feature file by its exact J0 cache identifier in either `<J0-run>/cache/` (initial J0 layout) or `<J0-run-parent>/cache/` (shared layout). Keep that directory when moving runs. J1 writes into a new UUID directory under `--output-root` and leaves all J0/F0/I0 outputs unchanged. Do not repeatedly rerun J1 to tune against the provided validation set.

## Artifacts

Each J1 UUID directory contains:

| File | Contents |
|---|---|
| `J1_cv_folds.csv` | One row per C/fold: fitting original-image count, fitting view-row count, scored original-image count, sibling overlap, accuracy and macro-F1. |
| `J1_cv_summary.csv` | Mean and sample SD of accuracy/macro-F1 per C across the three folds. |
| `J1_original_image_folds.csv` | Original training path, class label and held-out fold assignment; siblings inherit that original image's assignment. |
| `J1_selection.json` | Chosen C, fixed candidate list, CV/ranking rules, resolution, cache identifier/metadata and scikit-learn version. Written before provided-validation prediction. |
| `J1_svm.joblib` | The final fitted StandardScaler + RBF-SVM pipeline, trained on all training views. Load only models produced locally by this experiment; do not load unknown joblib files. |
| `J1_metrics.json` | Single provided-validation accuracy, macro-F1, correct count and provenance flags. |
| `J1_validation_predictions.csv` | Provided-validation image path, true label and predicted label in original validation-image order. |

J1 deliberately generates no test predictions, submission CSV or Kaggle upload. Test inference is a separate next step only if the candidate is retained.

## Predeclared comparison rule

Retain J1 only if its **single provided-validation accuracy strictly exceeds** the already selected J0 resolution's **fixed seed-2026 E4 head accuracy**. If accuracy ties or decreases, keep E4. Macro-F1 is reported as a secondary diagnostic and does not override this strict comparison rule. Do not compare J1 with the worst E4 seed or with a different resolution chosen after seeing J1's validation result.

A small validation gain is provisional: one additional correct image changes accuracy by 0.5 percentage points. This comparison does not establish a Kaggle leaderboard gain. No leaderboard scores are used for tuning, and no score is claimed until a separate manual submission is evaluated.

## Lightweight verification

```bash
python -m unittest discover -s tests -p test_svm_accuracy_experiment.py -v
```

The focused tests use synthetic features only. They exercise sibling-safe folding, deterministic-view scoring, fold-local scaling, fixed C candidates and tie ranking, training-only final statistics, independence of C from provided-validation labels, cache identity/layout checks, output ordering and local model round-tripping. They do not constitute a real J1 accuracy measurement.


## Executed result

Actual J1 validation:95.0%, macro-F1 .9497, selected C=10. Rejected; no test predictions generated. See `ACCURACY_IMPROVEMENT_RESULTS.md` for the full comparison.
