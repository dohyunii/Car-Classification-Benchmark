# I0: frozen DINOv2 ViT-B/14 + P3 + E4

Open `I0_DINOv2_ViTB14.ipynb` in Colab using **File → Open notebook → GitHub** after these new files are pushed, or upload the notebook directly. Choose **Runtime → Change runtime type → T4 GPU**. It is self-contained; no helper module, previous classifier or API key is needed.

1. Run section 0. Retain Colab's installed torch/torchvision pair.
2. In section 1, set `DATA_ROOT` to the course export containing `train/`, `val/`, flat `test/`, and original `sample_submission.csv`. Data is not automatically downloaded. The data used here has exactly 700 training images, 200 validation images, and 20 matching class folders prefixed `000_` through `019_`. Test length follows the sample CSV (700 for the original export).
3. For persistence, set `USE_DRIVE=True`; adjust `STORAGE_ROOT` and `DATA_ROOT` for your mounted Drive. Default local Colab storage is temporary.
4. Keep `HEAD_SEEDS=[2026]`, augmentation seed 123, extraction batch size 8, and run all cells top to bottom. Official model source and weights require Internet access. No real pretrained results are embedded in the notebook.
5. Optionally rerun from section 1 with `HEAD_SEEDS=[123,42,2026]`. A new run directory protects earlier artifacts, and the compatible shared feature cache is reused. All three heads are fresh; augmentation stays fixed at 123. The notebook computes mean and sample SD only when all three requested classifier seeds complete in that execution.
6. Download or retain the printed run directory. CSV submissions are generated locally, never uploaded to Kaggle. Decide manually whether to submit based on validation results.

## Implementation and comparison

Remote repository commit `9464fdfb6075d8908fa05b95a3654671f1fea55e` was fetched for read-only inspection without checking it out over existing local work. Its `baseline.ipynb` is an F0 **recovery** notebook, not proof that the original F0 weights or metrics were reproduced. It contains the P3 transforms, E4 construction, training-only standardization, seeded shuffle, AdamW training and accuracy-based early stopping used as references. H2 and H3 notebooks exist remotely. H2's README explicitly says actual fine-tuning was not run; no improvement is assumed. Existing local DINOv1 preprocessing functions and tests were also inspected. Historical F0 scores (98%, macro-F1 0.9803, Kaggle 0.94693) are user-reported references, not reverified results.

I0 uses official `torch.hub.load` DINOv2 `dinov2_vitb14` with pretrained weights, frozen in evaluation mode. Source is pinned to official upstream SHA `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, matching H2's source pin; branch-head validation is skipped for this immutable SHA. A real forward pass must return 768-D features. Hub caches source/weights; there is no random-weight fallback.

P3 stores one deterministic bicubic center crop plus four independently sampled geometric views, with no ColorJitter. Random crop scale .8–1, ratio .9–1.1, horizontal flip .5. ImageNet normalization is used. Extraction is FP32 with inference mode, no gradients, deterministic ordering and zero workers. Validation/test each have one deterministic view and no TTA. Labels are repeated in view-major order. Expected tensors: **(3500,768)** training features, **(3500,)** training labels, **(200,768)** validation features and **(700,768)** test features (or actual sample length).

E4 is `Linear(768,128) → BatchNorm1d(128) → ReLU → Dropout(.3) → Linear(128,20)` with 101,268 trainable parameters. No old head weights are loaded. AdamW uses lr .001, weight decay .001, batch size 64, at most 35 epochs and patience 7. First strictly best validation accuracy wins, matching inspected F0; macro-F1 is recorded, not used as a new tie-break. Seed resets immediately before each fresh head/loader; shuffling uses an independent seeded generator. Deterministic algorithms are enabled and TF32 disabled. Software/hardware differences can still affect comparison with historical F0.

Training-only per-feature mean and population SD (clamp 1e-6) standardize every split. All five training views contribute; validation/test do not fit statistics or train the head. Duplicate image contents across splits are rejected.

## Cache and artifacts

New caches live exclusively in `student_outputs_I0/cache_vitb14/`. Keys include backbone/source SHA, actual weight digest, feature width, P3 transforms, view count, augmentation seed, class mapping, ordered content-hashed image manifests, test IDs, pipeline/package versions, device and extraction batch size. Head seed is excluded. Shape, dtype, finite values, labels and exact metadata must match. Invalid caches regenerate, and `FORCE_REEXTRACT=True` explicitly refreshes the same I0 key. Increase `PIPELINE_VERSION` when changing extraction logic.

Each execution creates `student_outputs_I0/<timestamp_uuid>/`. Every head saves a uniquely located `I0_dinov2_vitb14_p3_e4_aug123_head<seed>.pt` containing weights, normalization, mapping, upstream/cache metadata, seeds, training settings, best epoch and metrics. Re-executing training/submission cells refuses to overwrite their checkpoint or CSV; rerun from section 1 to start a new directory. Feature signatures and actual backbone weights are checked before training to catch stale in-memory inputs. Existing F0/H2 notebooks, outputs, checkpoints and submissions remain unchanged.

Results include per-epoch CSVs/plots, metrics JSON, classification reports/confusion matrices, `I0_results.csv`, `I0_F0_comparison.csv`, optional `I0_multiseed_summary.csv` and `I0_dinov2_vitb14_seed<seed>_submission.csv`. Prediction generation reloads the chosen checkpoint and its saved statistics; it does not use validation labels. Original sample ID order, uniqueness, count and integer class bounds are checked.

## Validation and results to return

Run `python -m unittest discover -s tests -p test_i0_notebook.py -v` with notebook dependencies and nbformat installed. The test schema-validates and compiles cells, executes sequentially on synthetic data with the real split sizes and a tiny fake 768-D encoder, then checks head restoration, view-label alignment, train-only statistics, safe cache reuse/rejection/regeneration, seed changes, frozen eval behavior and exact submission ordering/overwrite protection. It does not download or train DINOv2.

CUDA is unavailable in the Codex environment. Actual official ViT-B/14 weight loading, T4 memory/runtime, Drive integration and real I0 accuracy **were not executed or measured**. Model loading is statically checked and mocked in integration tests. No synthetic score is saved as an experimental result. Real execution remains for Colab; lower extraction batch size to 4 if necessary, which deliberately creates a separate cache key.

Afterward report validation accuracy, macro-F1, correct predictions out of 200, best epoch, trainable head/frozen backbone parameter counts and training curves. For explicitly configured three-seed evaluation also report mean ± sample SD. Send the metrics JSONs, `I0_results.csv`, comparison CSV and optional summary. Kaggle should remain “not submitted” until you manually submit and obtain an actual score. A larger backbone is not assumed to improve performance.
