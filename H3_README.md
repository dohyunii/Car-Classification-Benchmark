# H3: frozen DINOv2 S/B/L comparison

Open `H3_DINOv2_Backbone_Comparison.ipynb` in Google Colab, select an **A100 GPU**,
configure section 1, and choose **Runtime → Run all**. The notebook runs all three
candidates sequentially. No F0 checkpoint is required.

## What is compared

| Backbone | Feature dimension | Image size | Extraction batch |
|---|---:|---:|---:|
| DINOv2-S/14 | 384 | 336 | 32 |
| DINOv2-B/14 | 768 | 336 | 16 |
| DINOv2-L/14 | 1024 | 336 | 8 |

All backbones use pinned original pretrained weights and remain frozen in evaluation
mode. Features are extracted in FP32 with TF32 disabled. The experiment retains F0's
original training image plus four augmented views, geometry/flip augmentation, and
seeds 123/2026. Each candidate receives its own freshly trained MLP-BN-Dropout head
and training-only feature normalization. Evaluation uses BICUBIC Resize(384)/CenterCrop(336).
Dedicated loader generators control augmentation comparisons; this does not claim
exact replay of historical F0 crops. No external labeled data, pseudo-labeling, TTA,
or ensemble is included. A 448-pixel follow-up is deliberately separate.

Head training uses hidden size 128, BatchNorm, ReLU, Dropout(0.3), AdamW LR 1e-3 and
weight decay 1e-3, batch 64, maximum 35 epochs, and patience 7. Fresh head BN learns its
statistics, as in F0. Checkpoint selection uses accuracy then macro-F1; F0 originally
selected by accuracy alone. Each epoch's head and validation probabilities are saved,
including a separate lowest-validation-loss checkpoint.

## Colab setup

1. Set `USE_DRIVE=True` to retain results in Drive (the default), and authorize
   the mount. Without Drive, download the run folder before ending the runtime.
2. Accept the competition rules in Kaggle. Add Colab Secret `KAGGLE_API_TOKEN`
   and enable notebook access. An existing environment credential also works.
3. Alternatively set `DOWNLOAD_FROM_KAGGLE=False` and set `DATA_ROOT` to extracted
   data containing `train/`, `val/`, `test/`, and `sample_submission.csv`.
4. Run from the top. It checks the original 700/200 split, 20 matching class folders,
   and exact sample submission IDs. Test images are used for inference only.
5. If GPU memory runs out, reduce the relevant candidate's `batch_size`. Completed
   candidate artifacts remain saved. A new top-to-bottom execution creates a new run.

`NUM_WORKERS=0` avoids the worker cleanup errors seen in H2. One backbone is resident
on the GPU at a time. Features are stored on CPU; content-validated caches can be
reused between runs. Setup uses Colab's existing compatible PyTorch/TorchVision and
installs supporting packages; internet is needed for setup, weights, and Kaggle.

## Optional F0 reference

By default the notebook searches these Drive project folders for
`dinov2_p3_e4_aug123_head2026.pt`:

- `student_outputs/`
- `student_outputs_F0_recovery/` and its run subfolders.

It uses the most recently modified match. Set `F0_CHECKPOINT` explicitly to choose
another file. If found, F0 is evaluated with its original 224-pixel transforms and
saved normalization, and its per-image predictions enter the disagreement table.

If no checkpoint is found, H3 still runs. The user's previously measured 98% accuracy
and 0.980278595 macro-F1 are labeled as a historical reference, not a newly measured
baseline. This distinction is recorded in `H3_summary.json`.

If F0 comparison fails, candidate results are already saved. Diagnose the checkpoint,
or set `F0_CHECKPOINT=None` and `AUTO_FIND_F0=False` and rerun section 7 onward to use
the historical reference. Do not bypass errors in H3's own model/data checks.

## Artifacts

Each execution writes a new folder under
`MyDrive/Car-Classification-Benchmark/student_outputs_H3/<run-id>/` containing:

- `H3_summary.json` and `H3_comparison.csv` with measured candidate metrics.
- `H3_disagreements.csv` counting complementary validation errors.
- `H3_training_curves.png`, plus an error gallery when the selected model has errors.
- `H3_run_config.json` and `dataset_manifest.json`.
- One directory per backbone with `best_head.pt`, `lowest_loss_head.pt`, `history.csv`,
  classification report, and validation/test probability files.
- Per-epoch head checkpoints and validation probabilities under each `epochs/` directory.

Validation IDs are relative class/image paths; probability columns follow class
indices 0–19. Test IDs retain leading zeros and the original sample row order. Head
checkpoints include saved normalization, mapping, preprocessing, source revision,
seeds, metrics, and environment metadata. Full frozen backbone weights are reloaded
from the pinned original pretrained source, rather than duplicated in each checkpoint.
Do not add datasets, caches, or model weights to Git.

Only a selected H3 model that beats the F0 reference by accuracy then macro-F1 produces
a separate H3 submission CSV. Tiny floating-point differences are ignored. If no model
qualifies, test probabilities are still saved for later analysis. Submission generation
can be disabled; nothing is automatically uploaded to Kaggle. An observed validation
gain is not a claim of a Kaggle gain.

## Verification and what to send back

Notebook schema validation, Python compilation, and synthetic CPU integration checks
cover the pipeline mechanics. Actual DINOv2 S/B/L training and feature extraction on
the user's A100 are not claimed as completed: the development workspace has CPU-only
PyTorch and no exposed NVIDIA GPU. Dependency installation, real pretrained downloads,
Drive/Kaggle authentication, and GPU execution require the Colab run.

Send `H3_summary.json`, `H3_comparison.csv`, `H3_disagreements.csv`, and
`H3_training_curves.png`. Retain the full run directory and feature caches so the
winning representation and complementary predictions can be reused later.
