# Car preprocessing benchmark

Compare P0–P4 using pretrained **DINOv1 ViT-S/16**, frozen in evaluation mode, 384-D embeddings, and a fresh `Linear(384, 20)` classifier. AdamW, cross-entropy, 35 epochs, learning rate 0.001, weight decay 0.001, batch size 64, and patience 7 stay fixed. Validation accuracy selects the best epoch; macro-F1 is secondary. This phase implements preprocessing only, not all assignment requirements.

The untouched `Baseline_Submission.ipynb` uses bilinear resizing and seed 42. It lacks controlled augmentation comparisons, separate seeds and experiment records. Its cache uses paths, file sizes and modification times. No historical score has been imported or verified. New P0 uses bicubic; it is not an exact reproduction of the old interpolation setting.

## Data and environment

Python 3.10+ with compatible PyTorch/TorchVision, timm, NumPy, pandas, Pillow, scikit-learn, matplotlib and Jupyter. Colab T4 is supported; CPU is slower. Do not reinstall CPU PyTorch in Colab. The notebook prints package versions, which are included in cache metadata. Reproducibility assumes the same software/hardware; deterministic algorithms are enabled and unsupported operations fail explicitly.

```text
data/
  train/000_class_name/*.jpg ... 019_class_name/*.jpg
  val/000_class_name/*.jpg   ... 019_class_name/*.jpg
  test/image_id.jpg
  sample_submission.csv       # image_id,label; recommended, order preserved
```

Expected counts are explicitly 700/200/700; if the actual export differs, set `EXPECTED_COUNTS` to its verified counts. Splits are never changed. Duplicate content across splits is rejected. Data is not downloaded; obtain the course export yourself. Model weights require Internet access to Hugging Face and its download hosts. Do not put credentials in the notebook.

## Run P0 in Google Colab

1. Push these implementation files to your GitHub repository, or upload `preprocessing_benchmark.py` alongside the new notebook in Colab's `/content`. Open `preprocessing_experiments.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Run the dependency and environment cells. They use Colab's installed torch/torchvision and install only notebook dependencies. The import cell expects the helper in `/content` or the notebook working directory; alternatively clone your repository and change into it before that cell.
3. Mount Drive with the optional cell if needed. Set `DATA_ROOT` to your extracted dataset, e.g. `/content/drive/MyDrive/cars`, and `OUTPUT_ROOT` to `/content/drive/MyDrive/car_preprocessing_outputs` for persistence.
4. Keep `EXPERIMENT_ID = 'P0'`, all seeds 123, `MODE = 'single'`, `FORCE_REEXTRACT = False`, and `MAKE_SUBMISSION = False`. Run all remaining cells top to bottom. Model loading may download pretrained weights once. The notebook reports actual validation metrics; no scores are prepopulated.

## Controlled experiments

| ID | Original views | Random views | ColorJitter | Embeddings/image |
|---|---:|---:|---|---:|
| P0 | 1 | 0 | No | 1 |
| P1 | 0 | 1 | No | 1 |
| P2 | 0 | 1 | Yes | 1 |
| P3 | 1 | 4 | No | 5 |
| P4 | 1 | 4 | Yes | 5 |

All evaluation images use Resize(256, bicubic), CenterCrop(224), tensor conversion and ImageNet normalization. Random views use RandomResizedCrop(224, scale=.8–1, ratio=.9–1.1, bicubic) and horizontal flip .5. P2/P4 additionally apply ColorJitter(.2,.2,.15,.02) with probability .7. P3/P4 store **view-major rows**, repeat the full label vector for each view, and fit standardization to all training views. Their embeddings are correlated views, not extra independent images. More views also mean more optimizer steps per epoch under the fixed training recipe; interpret P1/P3 and P2/P4 comparisons accordingly.

Change only `EXPERIMENT_ID` to P1, P2, P3 or P4 and rerun configuration and execution cells. Functions create fresh local state; no stale notebook feature variables are used. Set `MODE = 'initial'` to run all five with augmentation/head seed 123 sequentially and one loaded backbone. Inspect the table before choosing a small `PROMISING_IDS` list; set `MODE = 'repeat'` to test that list on [42,123,2026]. Do not tune on test predictions. On 200 validation images one correct prediction is 0.5 percentage points; consider sample SD and macro-F1, not tiny single-seed differences.

## Cache and seed safety

Global, augmentation and head seeds are separate. Set the augmentation seed once before extraction; it advances across all four views. The head seed is reset immediately before a fresh head, optimizer and seeded shuffle loader. Workers are zero. Changing only the head seed reuses features. Feature extraction runs under inference mode with the frozen backbone in eval mode.

Caches are separate P0–P4 files identified by a SHA-256 of preprocessing settings, view count/layout, augmentation seed, class mapping, ordered content-hashed dataset manifest/test IDs, actual model weight fingerprint, pretrained configuration, pipeline/package versions, device and extraction batch size. Head-only settings are excluded. Metadata, labels, shape, dtype and finite values are checked on load; invalid caches regenerate. `FORCE_REEXTRACT = True` forces regeneration. To avoid accidental reuse after implementation changes, increment `PIPELINE_VERSION` when changing feature logic. Cache IDs are printed. Train-only means and standard deviations are recomputed per run and saved with the head.

## Artifacts and interpretation

`OUTPUT_ROOT` contains `cache/`, `checkpoints/`, `metrics/`, `plots/`, and `submissions/`. Run identifiers include configuration digest, augmentation seed and head seed. Repeating the exact same run replaces its artifacts intentionally. JSON records include configurations, counts, metrics, best epoch, cache and checkpoint paths, software/device and timings. History CSVs contain loss/accuracy each epoch; confusion CSVs expose per-class behavior. Comparison CSVs and plots identify the exact selected runs, and summary CSVs report mean and **sample** SD (`ddof=1`, undefined for one run). Runtime includes hashing and loading; feature preparation may be a cache hit. No results are fabricated.

Enable `MAKE_SUBMISSION` only when desired; each file contains exactly `image_id,label`, valid class IDs and original sample ordering. Validation results are not Kaggle results. Do not submit synthetic smoke outputs. Persistent Drive output is recommended because Colab runtime storage is temporary.

## Local validation

Run `python -m unittest discover -s tests -v`. Tests use synthetic images and a tiny fake frozen 384-D encoder, not pretrained DINO; they check augmentation reproducibility, view/label order, content-based invalidation, cache corruption recovery, head-seed cache reuse, standardization, and end-to-end artifacts. Neither the private dataset, full pretrained training, T4 execution nor accuracy claims are covered. Run the actual P0–P4 benchmark in Colab to obtain those results. Original notebook remains unchanged.
