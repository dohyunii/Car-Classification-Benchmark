# J0 measured accuracy comparison

Actual CPU runs using the provided course train/validation split. No external labeled images, original Stanford Cars labels, pseudo-labels or validation/test training were used. Test inference ran only after resolution selection; no Kaggle submission was made.

| Resolution | Head seed | Validation accuracy | Macro-F1 | Correct / 200 | Best epoch |
|---|---:|---:|---:|---:|---:|
| 224 | 42 | 97.5% | 0.9749 | 195 | 6 |
| 224 | 123 | 96.0% | 0.9598 | 192 | 14 |
| 224 | 2026 | 97.5% | 0.9749 | 195 | 7 |
| 336 | 42 | 96.5% | 0.9640 | 193 | 2 |
| 336 | 123 | 96.0% | 0.9593 | 192 | 7 |
| 336 | 2026 | 96.5% | 0.9644 | 193 | 13 |

| Resolution | Mean accuracy ± sample SD | Mean macro-F1 ± sample SD |
|---|---:|---:|
| 224 | 97.00% ± 0.87 pp | 0.9699 ± 0.0087 |
| 336 | 96.33% ± 0.29 pp | 0.9626 ± 0.0028 |

336px changed mean validation accuracy by -0.67 percentage points against the matched 224px baseline.
The predeclared mean-accuracy/mean-F1 selection chose 224px. The submission candidate uses fixed head seed 2026, scoring 97.5% validation accuracy and 0.9749 macro-F1. It was generated locally, **not submitted**.

Historical F0 ViT-S/14=98.0% and user-reported I0 ViT-B/14=97.5% were not reverified in these exact implementations. J0 uses the timm port of public self-supervised DINOv2-S weights with dynamic positional interpolation; the controlled 224px baseline is therefore the main comparison. All six checkpoints share the same actual backbone-weight fingerprint.

Settings: P3 original+four geometric views, augmentation seed123, E4 head, AdamW lr/decay .001, batch64, maximum35 epochs, accuracy-only best checkpoint, patience7. Standardization uses training embeddings only. View count and all head settings remain fixed between resolutions.

Three seeds reuse one 200-image validation split, so their SD measures training randomness, not uncertainty across new datasets. One validation image changes accuracy by0.5 percentage points. This bounded two-resolution experiment does not establish Kaggle improvement or justify an unrestricted validation search.

Submission CSV: `outputs/J0/c8aab8360ba2/submissions/J0_S14_224_head2026_submission.csv`. Run directory: `outputs/J0/c8aab8360ba2`. Histories, predictions, paired error changes, checkpoints, features, environment metadata and comparison plot are saved there.


## J1: training-only-selected RBF SVM

The sole additional classifier candidate used the J0-selected224px features. C∈{0.1,1,10} was selected by3fold stratified CV on original training images only. All5 sibling views stayed with each fitting image, and heldout scoring used deterministic original views only. Each fold fit its own scaler. C=10 was selected before accessing provided-validation predictions.

Measured validation accuracy: **95.0%** (190/200); macro-F1 **0.9497**. This is below the97.5% E4 comparator, so J1 was rejected and no J1 test predictions were generated. Artifacts: `outputs/J1/8199bfcca5ee/`.


## J2: conservative last-two-block fine-tuning

The actual starting checkpoint was independently verified at97.5% (195/200), macro-F1 0.9749. Only blocks10/11, final normalization and the E4 head were trainable. Backbone stayed in eval mode; head BatchNorm running statistics and train-derived feature statistics stayed fixed. AdamW rates were2e-6 backbone and3e-5 head, decay1e-4, batch8, gradient clip1. Only provided training images received fresh geometric augmentation.

Epoch1:96.5%; epoch2:97.5%; epoch3:96.0%. Training stopped after3 non-improving epochs. Strict accuracy selection retained epoch0 at97.5%; restored head and normalization tensors were verified identical to the source. No J2 test predictions or Kaggle submission were generated. Artifacts: `outputs/J2/a00177a18e98/`.


## J3: CLS plus mean patch-token features

All3500 training and200 validation CLS prefixes matched the paired J0 features at rtol0/atol1e-5, confirming unchanged weights, image order and augmentation stream. The new representation concatenated the final normalized384-D CLS token with384-D mean normalized patch tokens. Only E4 input width changed to768; all other head/training settings stayed fixed.

| Head seed | Validation accuracy | Macro-F1 | Best epoch |
|---|---:|---:|---:|
| 42 | 96.5% | 0.9650 | 10 |
| 123 | 96.0% | 0.9596 | 6 |
| 2026 | 96.5% | 0.9649 | 11 |

Three-seed mean: 96.33% ± 0.29 percentage points (sample SD). The sole predeclared equal-probability average of all3 heads scored 96.0%, macro-F1 0.9597. No weights or seed subsets were tuned. Neither pooled seed2026 nor the ensemble exceeded97.5%, so the original baseline was retained; no J3 test predictions were generated. Artifacts: `outputs/J3/b6841bc5b82a/`.

## Final disposition

No tested change improved on the matched97.5% J0 checkpoint. Your historical/user-reported98.0% F0 remains the stronger reference; its original Colab checkpoint/cache are not present locally and were not reproduced exactly by the timm backend. Keep that model rather than replace it with a lower-scoring candidate. The local J0 97.5% checkpoint is reproducible and retained; J1/J2/J3 candidates are rejected. No additional labeled data, original Stanford Cars labels, pseudo-labels, validation/test training, or automatic Kaggle submissions were used. Every attempted configuration and failed result is preserved.

Further targeted work should begin with the original F0 checkpoint, source notebook/package versions, and ideally feature cache, so the actual98.0% model can be verified before further modifications. Continuing an unrestricted search on the same200-image validation set would make small apparent gains less trustworthy.
