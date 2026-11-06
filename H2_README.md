# Experiment H2: Colab execution

`H2_DINOv2_FineTuning.ipynb` is a separate, self-contained notebook. Existing
experiments, checkpoints, and submissions are unchanged. The implementation was
checked against the GitHub F0/H0/H1 notebook at commit
`5a393fa56ae33a183a2268253528a795b5a450fe`; the older local DINOv1 notebook was
also inspected and left unchanged.

## Architecture

The model applies `head((backbone(images) - feature_mean) / feature_std)` using
the original F0 head and its saved normalization buffers. Only transformer blocks
10 and 11, final LayerNorm, and the head receive parameter updates. The head is
`Linear(384,128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128,20)`.
Backbone evaluation behavior is retained with autograd enabled. Head Dropout is
active during training; BatchNorm running statistics stay fixed for small batches,
while its affine parameters remain trainable.

Training uses fresh image augmentation, AdamW with backbone/head rates 2e-6/3e-5,
batch size 8, up to 10 epochs, patience 3, clipping at 1.0, and CUDA AMP. Evaluation
uses the original deterministic BICUBIC transforms in FP32. No TTA, ensemble, or
cached training features are used.

## Exact Colab steps

1. In Colab, choose **File → Open notebook → GitHub**, enter this repository, select
   the H2 branch (or main after merging), and open `H2_DINOv2_FineTuning.ipynb`.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Edit section 1. For persistent storage set `USE_DRIVE = True`, and put the
   original checkpoint at:
   `/content/drive/MyDrive/Car-Classification-Benchmark/student_outputs/dinov2_p3_e4_aug123_head2026.pt`.
   Edit `F0_CHECKPOINT` if needed. Without Drive, upload the checkpoint through
   Colab's Files panel into `/content/student_outputs/`.
4. Keep `DOWNLOAD_FROM_KAGGLE = True` to use the original competition download.
   Accept its rules on Kaggle and add Colab Secret `KAGGLE_API_TOKEN` with notebook
   access. An existing environment credential also works. For extracted data,
   set that flag to `False` and edit `DATA_ROOT` to contain `train/`, `val/`,
   `test/`, and `sample_submission.csv`.
5. Choose **Runtime → Run all** and authorize Drive if enabled. Section 7 must
   reproduce 98.00% validation accuracy and macro-F1 within 0.0005 of the approximate
   published 0.9803 before training can start.
6. Retain the run folder printed in section 12. Drive mode writes beneath
   `MyDrive/Car-Classification-Benchmark/student_outputs_H2/<run-id>/`. Without
   Drive, download `/content/student_outputs_H2/<run-id>/` before the runtime ends.

Missing or incompatible F0 checkpoints stop execution. There is no random-weight
replacement or automatic retraining of F0. Use the original checkpoint containing
head weights, feature statistics, class mapping, and the documented metadata.

## Outputs and selection

Every run creates its own directory containing the full best checkpoint
`H2_dinov2_last2_seed2026.pt`, `H2_history.csv`, `H2_validation_comparison.csv`,
`H2_classification_report.csv`, `H2_training_curves.png`, `H2_run_config.json`,
and `H2_summary.json`. A confusion matrix is optional.

The starting F0 model is saved as epoch 0 before training. Selection uses validation
accuracy first, then macro-F1. The best saved checkpoint is restored at the end.
Only a fine-tuned epoch that beats epoch 0 by this ranking produces
`H2_dinov2_last2_seed2026_submission.csv`. An accuracy tie with higher F1 qualifies
and is reported explicitly. If epoch 0 stays best, no H2 submission is generated.
The notebook writes a CSV; it does not upload it to Kaggle.

## Assumptions and troubleshooting

- The original split has exactly 700 training images, 200 validation images, and
  20 matching ImageFolder classes prefixed `000_` through `019_`.
- Test images are in a flat directory. Filename stems match the original sample
  CSV's string `image_id` values. The `image_id,label` sample CSV is mandatory;
  leading zeros and row ordering are preserved.
- Setup, pretrained weights, and optional Kaggle download need internet access.
  Colab's PyTorch/TorchVision are retained; use PyTorch 2.4 or newer. Supporting
  dependencies are installed in section 0. DINOv2 source is pinned to an upstream
  commit; immutable-SHA loading skips hub's branch-head validation.
- If DataLoader processes fail, set `NUM_WORKERS = 0` and rerun from the top.
- If baseline validation fails, inspect the checkpoint, dataset version, mapping,
  BICUBIC transforms, saved feature statistics, and diagnostics. Resolve the cause
  before training; do not bypass the baseline gate.
- The submission cell refuses to overwrite a CSV if executed twice in one run.
- Keep datasets and model weights out of Git.

## Verification and results to return

Notebook JSON/schema validation and Python compilation passed. Synthetic CPU
integration checks exercised sequential execution, exact head restoration,
normalization, baseline-failure diagnostics, gradient flow, nonzero-learning-rate
updates, frozen weights/BatchNorm statistics, checkpoint round-tripping, epoch-0
restoration, early stopping, evaluation artifacts, and submission validation/gates.

**The actual experiment was not run:** the development environment has no F0
checkpoint and no CUDA device. No H2 performance results are claimed. The dependency
installation cell, real pretrained-weight download, Drive/Kaggle authentication,
and CUDA AMP path still need execution in Colab.

Send back `H2_summary.json`, `H2_history.csv`, `H2_validation_comparison.csv`, and
`H2_training_curves.png`. They provide baseline/best accuracy and macro-F1,
percentage-point changes, best epoch, and trainable parameter counts. If the baseline
fails, send its diagnostics. If you submit a candidate manually, also send its actual
Kaggle score for comparison with F0's reported 0.94693.
